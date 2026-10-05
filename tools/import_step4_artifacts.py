"""Audit completed finals on CPU, then import evidence without another model forward."""
from __future__ import annotations
import hashlib
import io
import json
import shutil
import sys
import tempfile
import zipfile
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import inference, step3, step4, train


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(actual, expected, where):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), where
        for key in expected:
            close(actual[key], expected[key], where + '/' + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for index, value in enumerate(expected):
            close(actual[index], value, where + '/' + str(index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10, err_msg=where)
    else:
        assert actual == expected, where


def unpack(archive, stage):
    with zipfile.ZipFile(archive) as bundle:
        names = [i.filename for i in bundle.infolist() if not i.is_dir()]
        assert len(names) == len(set(n.casefold() for n in names)), 'Duplicate ZIP paths'
        assert bundle.testzip() is None, 'Corrupt ZIP'
        for name in names:
            target = (stage / name).resolve()
            assert target.is_relative_to(stage.resolve()), 'ZIP path escapes staging'
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as supplied, target.open('wb') as destination:
                shutil.copyfileobj(supplied, destination)
    return names


def main():
    directory = ROOT / 'runs/step4_import'
    artifacts = directory / 'step4_artifacts.zip'
    weights = directory / 'step4_best_checkpoints.zip'
    labels_dir = ROOT / 'runs/step0_validation/labels'
    hashes = {}
    with tempfile.TemporaryDirectory(prefix='deepweeds_final_audit_') as temporary:
        stage = Path(temporary)
        names = unpack(artifacts, stage)
        weight_names = unpack(weights, stage)
        expected_weights = {f'runs/{g}/seed{s}/best.pt' for g in step4.GROUPS for s in step4.SEEDS}
        assert set(weight_names) == expected_weights, 'Need exactly six best checkpoints'
        assert (stage / 'eval.py').read_bytes() == (ROOT / 'eval.py').read_bytes(), 'Evaluator changed'
        for source in (stage / 'code').glob('*.py'):
            assert source.read_bytes() == (ROOT / 'code' / source.name).read_bytes(), source.name
        plan_path = stage / 'step4/frozen_plan.json'
        plan = read(plan_path)
        assert plan == read(ROOT / 'step4/frozen_plan.json'), 'Actual frozen choices differ'
        assert plan['code_sha256'] == step4.code_hash() and not plan['test_used']
        plan_hash = step3.sha256(plan_path)
        final_rows = read(stage / 'step4/final.json')
        assert {(r['exp_id'], r['seed']) for r in final_rows} == {(g, s) for g in step4.GROUPS for s in step4.SEEDS}
        assert len(final_rows) == 6
        assert len([n for n in names if n.startswith('predictions/')]) == 18
        assert len([n for n in names if n.startswith('curves/')]) == 6
        for group in step4.GROUPS:
            for seed in step4.SEEDS:
                run = stage / f'runs/{group}/seed{seed}'
                folder = stage / f'step4/{group}/seed{seed}'
                record = read(run / 'config.json')
                cfg = record['config']
                source_cfg = plan['training_source'][group]
                replaced = {'exp_id', 'seed', 'images_dir', 'labels_dir', 'out_dir', 'pred_dir', 'curves_dir', 'device', 'resume'}
                assert all(cfg[k] == v for k, v in source_cfg.items() if k not in replaced), 'Recipe changed'
                assert (cfg['exp_id'], cfg['seed'], cfg['epochs'], cfg['batch_size'], cfg['img_size']) == (group, seed, 10, 32, 224)
                assert cfg['save_test_predictions'] is False
                assert record['gpu'] == 'Tesla T4' and record['pretrained_cfg']['tag'] == 'fb_in1k'
                assert record['versions']['torch'] == plan['runtime']['torch']
                assert record['versions']['timm'] == plan['runtime']['timm']
                split = read(run / 'split_checks.json')
                assert split['n'] == {'train': 10501, 'val': 3501, 'test': 3507}
                assert split['union'] == 17509 and not any(split['overlap'].values()) and split['missing_files'] == 0
                summary = read(run / 'summary.json')
                history = pd.read_csv(run / 'history.csv')
                assert history['epoch'].tolist() == list(range(1, 11))
                best = int(history.loc[history['val_macro_f1'].idxmax(), 'epoch'])
                assert summary['best_epoch'] == best and summary['test_evaluated'] is False
                path = run / 'best.pt'
                checkpoint_hash = step3.sha256(path)
                hashes[f'{group}_seed{seed}'] = checkpoint_hash
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                assert checkpoint['epoch'] == best and checkpoint['ema'] is False
                assert abs(checkpoint['macro_f1'] - history['val_macro_f1'].max()) < 1e-10
                assert tuple(checkpoint['model']['head.fc.weight'].shape) == (9, 768)
                assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
                del checkpoint
                val = read(folder / 'val_done.json')
                signature = dict(group=group, seed=seed, checkpoint_sha256=checkpoint_hash,
                                 plan_sha256=plan_hash, runtime=plan['runtime'])
                assert val['signature'] == signature and val['test_used'] is False
                assert val['fit_split'] == ('val' if group == 'F01' else None)
                assert 0 < val['T'] < float('inf')
                assert read(folder / 'test_started.json') == val, 'Once-only ledger differs'
                done = read(folder / 'test_done.json')
                assert done['val'] == val and done['n'] == 3507 and done['test_forward_passes'] == 1
                timing = read(folder / 'latency_batch1.json')
                samples = np.asarray(timing['samples_ms'])
                assert timing['n'] == 100 and len(samples) == 100 and timing['warmup'] == 10
                assert timing['runtime'] == plan['runtime'] and timing['checkpoint_sha256'] == checkpoint_hash
                assert timing['batch'] == 1 and timing['dtype'] == 'fp32'
                assert timing['img_size'] == plan['inference'][group]['size'] and not timing['fused_bn']
                assert not timing['preprocessing_included'] and not timing['transfer_included']
                for key, q in [('p50', 50), ('p95', 95), ('p99', 99)]:
                    close(timing[key], np.percentile(samples, q), f'{group}/{seed}/{key}')
                close(timing['mean'], samples.mean(), 'latency mean')
                close(timing['images_per_s'], 1000 / samples.mean(), 'throughput')
                close(val['p95_ms'], timing['p95'], 'val p95')
                row = next(r for r in final_rows if (r['exp_id'], r['seed']) == (group, seed))
                for split_name, count in [('val', 3501), ('test', 3507)]:
                    filenames = read(folder / f'{split_name}_filenames.json')
                    y = np.load(folder / f'{split_name}_labels.npy', allow_pickle=False)
                    logits = np.load(folder / f'{split_name}_logits.npy', allow_pickle=False)
                    assert len(filenames) == count and logits.shape == (count, 9) and y.shape == (count,)
                    assert np.isfinite(logits).all()
                    for tag, temperature in [(group, val['T'])] + ([(group + '_uncal', 1.)] if group == 'F01' else []):
                        pred = train.ev.read_pred(str(stage / f'predictions/{tag}_seed{seed}_{split_name}.csv'))
                        train.ev.check_against_csv(pred, str(labels_dir / f'{split_name}_subset0.csv'), split_name)
                        np.testing.assert_array_equal(pred.filenames, filenames)
                        np.testing.assert_array_equal(pred.y_true, y)
                        probs = inference.apply_temperature(logits, temperature)
                        np.testing.assert_allclose(pred.probs, probs, rtol=0, atol=5e-9)
                        metric = train.ev.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
                        stored = (val['val_metrics'] if tag == group else val['val_uncal_metrics']) if split_name == 'val' else (done['metrics'] if tag == group else done['uncal_metrics'])
                        for key in train.ev.SCALARS:
                            close(stored[key], metric[key], tag + '/' + split_name + '/' + key)
                        if tag == group:
                            close(row[f'macro_f1_{split_name}'], metric['macro_f1'], 'Final F1')
                            close(row[f'top1_{split_name}'], metric['top1'], 'Final accuracy')
                            if split_name == 'test':
                                close(row['ece_test'], metric['ece'], 'Final ECE')
                    if group == 'F01' and split_name == 'val':
                        fitted = inference.fit_temperature(logits, y, split='val')
                        assert abs(fitted - val['T']) < 1e-6, 'Temperature not reproducible on own val'
                print(f'{group} seed {seed}: checkpoint, recipe, val/test CSV, temperature and raw timing verified.', flush=True)
        output = stage / 'independent_eval'
        common = ['--test-csv', str(labels_dir / 'test_subset0.csv'), '--labels', str(labels_dir / 'labels.csv'), '--out', str(output)]
        logs = []
        for tag in ('F00', 'F01', 'F01_uncal'):
            stream = io.StringIO()
            with redirect_stdout(stream), redirect_stderr(stream):
                status = train.ev.main(['score', '--pred', str(stage / f'predictions/{tag}_seed*_test.csv'), '--tag', tag, *common])
            assert status == 0, stream.getvalue()
            close(read(output / f'{tag}_summary.json'), read(stage / f'step4/eval_outputs/{tag}_summary.json'), 'Official summary ' + tag)
            for suffix in ('per_seed', 'per_class', 'confusion_sum'):
                pd.testing.assert_frame_equal(pd.read_csv(output / f'{tag}_{suffix}.csv'), pd.read_csv(stage / f'step4/eval_outputs/{tag}_{suffix}.csv'), check_exact=False, rtol=0, atol=1e-10)
            logs.append(stream.getvalue())
        stream = io.StringIO()
        with redirect_stdout(stream), redirect_stderr(stream):
            status = train.ev.main(['grade', '--final', str(stage / 'predictions/F01_seed*_test.csv'),
                '--baseline', str(stage / 'predictions/F00_seed*_test.csv'), '--uncal', str(stage / 'predictions/F01_uncal_seed*_test.csv'),
                '--final-val', str(stage / 'predictions/F01_seed*_val.csv'), '--val-csv', str(labels_dir / 'val_subset0.csv'),
                '--latency-p95-ms', str(max(r['p95_ms'] for r in final_rows if r['exp_id'] == 'F01')), *common])
        assert status == 0, stream.getvalue()
        assert read(output / 'grade_I.json') == read(stage / 'step4/eval_outputs/grade_I.json')
        logs.append(stream.getvalue())
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'independent_eval_output.md').write_text('\n\n'.join(logs), encoding='utf-8')
        backup = directory / f'before_import_{datetime.now():%Y%m%d_%H%M%S}'
        backup.mkdir()
        shutil.copyfile(ROOT / 'results.xlsx', backup / 'results.xlsx')
        selected = [n for n in names if n.startswith(('runs/F', 'step4/', 'predictions/F', 'curves/F'))] + weight_names
        for name in selected:
            target, supplied = ROOT / name, stage / name
            if target.exists():
                if step3.sha256(target) == step3.sha256(supplied):
                    continue
                assert name == 'step4/frozen_plan.json', f'Existing final evidence differs: {name}'
                previous = backup / name
                previous.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, previous)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(supplied, target)
        for group in step4.GROUPS:
            for seed in step4.SEEDS:
                for name in ('config.json', 'history.csv', 'summary.json', 'split_checks.json'):
                    source = ROOT / f'runs/{group}/seed{seed}/{name}'
                    target = ROOT / f'step4/training_records/{group}_seed{seed}/{name}'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        manifest = dict(artifacts_sha256=step3.sha256(artifacts), checkpoints_zip_sha256=step3.sha256(weights),
            frozen_plan_sha256=plan_hash, checkpoint_sha256=hashes, seeds=[0, 1, 2], groups=['F00', 'F01'],
            prediction_files=18, curves=6, gold_csv_checked=True, calibration_reproduced_on_val=True,
            official_evaluator_recomputed=True, once_only_ledgers_checked=True, additional_model_forwards=0,
            raw_latency_checked=True, test_images_per_seed=3507, grade_I=read(output / 'grade_I.json'))
        train.write_json(ROOT / 'step5/verification.json', manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
