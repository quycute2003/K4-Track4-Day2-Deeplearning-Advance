"""Controlled T00-based ablations on the selected ConvNeXt-Tiny; val only."""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import dataset, train, step1
else:
    import dataset, train, step1

EXPERIMENTS = {
    'T01': ('Initialization', 'Scratch', {'init': 'scratch'}),
    'T02': ('Initialization', 'Frozen backbone', {'init': 'frozen'}),
    'T03': ('Augmentation', 'ColorJitter', {'aug': 'color'}),
    'T04': ('Augmentation', 'CutMix alpha=1', {'mix': 'cutmix'}),
    'T05': ('Loss', 'Label smoothing epsilon=0.1', {'loss': 'ls', 'label_smoothing': 0.1}),
    'T06': ('Loss', 'Focal gamma=2', {'loss': 'focal'}),
}
COLUMNS = ['exp_id', 'backbone', 'seed', 'axis', 'variant', 'base_exp_id', 'changed_fields',
           'macro_f1_val', 'delta_f1_pp', 'top1_val', 'ece_val', 'nll_val',
           'balanced_acc_val', 'f1_chinee_apple', 'f1_snake_weed', 'best_epoch',
           'train_seconds_epoch', 'status', 'init', 'aug', 'mix', 'loss', 'label_smoothing',
           'focal_gamma', 'pretrained_tag', 'notes', 'config_path', 'curve_path']


def baseline_record(project):
    record = step1.read_json(Path(project) / 'runs/B03/seed0/config.json')
    cfg = record['config']
    required = dict(exp_id='B03', backbone='convnext_tiny', init='finetune', seed=0,
                    fold=0, epochs=10, batch_size=32, img_size=224, aug='basic',
                    mix=None, loss='ce', save_test_predictions=False)
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError('B03 does not match the chosen T00 baseline')
    if record['pretrained_cfg']['tag'] != 'fb_in1k':
        raise ValueError('T00 must use ConvNeXt-Tiny fb_in1k')
    return record


def spec(project, exp_id):
    if exp_id == 'T00':
        return 'Baseline', 'B03 reused; no new training', {}
    if exp_id in EXPERIMENTS:
        return EXPERIMENTS[exp_id]
    if exp_id == 'T07':
        path = Path(project) / 'step2/combination_plan.json'
        if not path.exists():
            raise FileNotFoundError('Finish T03–T06 and create the combination plan first')
        plan = step1.read_json(path)
        parts = plan.get('selected_components', [])
        if (len(parts) != 2 or parts[0] not in ('T03','T04') or parts[1] not in ('T05','T06')
                or plan.get('overrides') != {**EXPERIMENTS[parts[0]][2], **EXPERIMENTS[parts[1]][2]}
                or plan.get('test_used') is not False or plan.get('seed') != 0):
            raise ValueError('Invalid preregistered combination plan')
        return 'Combination', plan['variant'], plan['overrides']
    raise ValueError(f'Unknown recipe: {exp_id}')


def assert_controlled(reference, candidate, overrides):
    """Accept exactly the preregistered factor changes, with no other drift."""
    ignore = step1.IGNORE_FIELDS - {'backbone'}
    expected = dict(reference)
    expected.update(overrides)
    changes = {key: (expected.get(key), candidate.get(key)) for key in expected.keys() | candidate.keys()
               if key not in ignore and expected.get(key) != candidate.get(key)}
    if changes or candidate['save_test_predictions']:
        raise ValueError(f'Uncontrolled recipe changes: {changes}')


def make_config(project, images_dir, labels_dir, exp_id):
    reference = baseline_record(project)
    _, _, overrides = spec(project, exp_id)
    cfg = step1.make_config(reference, project, images_dir, labels_dir,
                            exp_id, reference['config']['backbone'])
    for key, value in overrides.items():
        setattr(cfg, key, value)
    assert_controlled(reference['config'], asdict(cfg), overrides)
    return cfg


def assert_run_record(reference, record, overrides, source_id):
    assert_controlled(reference['config'], record['config'], overrides)
    if record['gpu'] != reference['gpu'] or any(record['versions'][key] != reference['versions'][key]
                                              for key in ('torch','torchvision','timm')):
        raise ValueError('Recipe screening environment differs from T00')
    if record['pretrained_cfg']['tag'] != 'fb_in1k' or record['config']['exp_id'] != source_id:
        raise ValueError('Unexpected recipe experiment or pretrained tag')


def val_metrics(project, source_id):
    root = Path(project)
    folder = root / 'runs' / source_id / 'seed0'
    pred = train.ev.read_pred(str(root / 'predictions' / f'{source_id}_seed0_val.csv'))
    summary = step1.read_json(folder / 'summary.json')
    if summary['test_evaluated'] or summary['seed'] != 0 or summary['exp_id'] != source_id:
        raise ValueError('Invalid screening summary')
    logits = np.load(folder / 'val_logits.npy')
    labels = np.load(folder / 'val_labels.npy')
    names = step1.read_json(folder / 'val_filenames.json')
    if list(pred.filenames) != names or not np.array_equal(pred.y_true, labels):
        raise ValueError('Val filename/label/logit alignment mismatch')
    probs = train.probabilities(logits)
    if logits.shape != (len(labels), 9) or not np.allclose(probs, pred.probs, atol=1e-7):
        raise ValueError('Saved probabilities differ from saved val logits')
    metrics = train.ev.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
    for key, saved in [('macro_f1', 'macro_f1_val'), ('top1', 'top1_val'), ('ece', 'ece_val')]:
        if not math.isclose(metrics[key], summary[saved], abs_tol=1e-8):
            raise ValueError(f'Saved summary differs from val predictions: {key}')
    record = step1.read_json(folder / 'config.json')
    labels_path = Path(record['config']['labels_dir']) / 'val_subset0.csv'
    if labels_path.is_file():
        train.ev.check_against_csv(pred, str(labels_path), 'val')
    return pred, metrics


def collect_results(project):
    root = Path(project)
    reference = baseline_record(root)
    base_pred, base_metrics = val_metrics(root, 'B03')
    alias = root / 'curves/T00_convnext_tiny_seed0.png'
    if not alias.exists():
        history = pd.read_csv(root / 'runs/B03/seed0/history.csv')
        train.plot_curves(history.to_dict('records'), alias, 'T00 — ConvNeXt-Tiny (B03 reused, seed 0)')
    rows, class_rows = [], []
    for exp_id in ['T00', *EXPERIMENTS, 'T07']:
        source_id = 'B03' if exp_id == 'T00' else exp_id
        plan_missing = exp_id == 'T07' and not (root / 'step2/combination_plan.json').exists()
        axis, variant, overrides = ('Combination', 'Pending T03–T06', {}) if plan_missing else spec(root, exp_id)
        folder = root / 'runs' / source_id / 'seed0'
        row = dict.fromkeys(COLUMNS)
        row.update(exp_id=exp_id, backbone='convnext_tiny', seed=0,
                   axis={'Initialization':'A. Initialization','Augmentation':'B. Augmentation','Loss':'C. Loss'}.get(axis,axis),
                   variant=variant, base_exp_id='T00 (=B03)', notes='Single seed; val only',
                   changed_fields=', '.join(overrides) if overrides else ('pending' if plan_missing else ''),
                   status='Chưa chạy' if not plan_missing else 'Chờ T03–T06',
                   config_path=f'runs/{source_id}/seed0/config.json',
                   curve_path=f'curves/{exp_id}_convnext_tiny_seed0.png')
        cfg = dict(reference['config'], **overrides)
        if not plan_missing:
            for key in ('init', 'aug', 'mix', 'loss', 'label_smoothing', 'focal_gamma'):
                row[key] = cfg[key]
            row['pretrained_tag'] = 'none (scratch)' if cfg['init'] == 'scratch' else 'fb_in1k'
        if not (folder / 'summary.json').exists():
            rows.append(row)
            continue
        if plan_missing:
            raise ValueError('T07 results exist but their combination plan is missing')
        record = step1.read_json(folder / 'config.json')
        summary = step1.read_json(folder / 'summary.json')
        assert_run_record(reference, record, overrides, source_id)
        history = pd.read_csv(folder / 'history.csv')
        if len(history) != cfg['epochs'] or not (root / row['curve_path']).is_file():
            raise ValueError('Missing full history or training curve')
        pred, metrics = (base_pred, base_metrics) if exp_id == 'T00' else val_metrics(root, source_id)
        if list(pred.filenames) != list(base_pred.filenames) or not np.array_equal(pred.y_true, base_pred.y_true):
            raise ValueError('Ablation and T00 use different val images or labels')
        row.update(macro_f1_val=metrics['macro_f1'], delta_f1_pp=100*(metrics['macro_f1']-base_metrics['macro_f1']),
                   top1_val=metrics['top1'], ece_val=metrics['ece'], nll_val=metrics['nll'],
                   balanced_acc_val=metrics['balanced_acc'], f1_chinee_apple=float(metrics['f1'][0]),
                   f1_snake_weed=float(metrics['f1'][7]), best_epoch=summary['best_epoch'],
                   train_seconds_epoch=float(history.train_seconds.mean()), status='Đủ kết quả')
        class_rows.extend(dict(exp_id=exp_id, label=i, species=dataset.CLASS_NAMES[i],
                               support=int(metrics['support'][i]), f1=float(metrics['f1'][i]),
                               delta_f1_pp=100*float(metrics['f1'][i]-base_metrics['f1'][i])) for i in range(9))
        rows.append(row)
    out = root / 'step2'
    out.mkdir(parents=True, exist_ok=True)
    train.write_json(out / 'ablations.json', rows)
    train.write_json(out / 'workbook_schema.json', COLUMNS)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out / 'ablations.csv', index=False)
    pd.DataFrame(class_rows).to_csv(out / 'per_class_f1.csv', index=False)
    return rows


def plan_combination(project):
    root = Path(project)
    rows = {row['exp_id']: row for row in collect_results(root)}
    for exp_id in ('T03', 'T04', 'T05', 'T06'):
        if rows[exp_id]['status'] != 'Đủ kết quả':
            raise ValueError('Combination requires completed T03–T06')
    aug = max(('T03', 'T04'), key=lambda exp: rows[exp]['macro_f1_val'])
    loss = max(('T05', 'T06'), key=lambda exp: rows[exp]['macro_f1_val'])
    overrides = {**EXPERIMENTS[aug][2], **EXPERIMENTS[loss][2]}
    plan = dict(exp_id='T07', selected_components=[aug, loss], overrides=overrides,
                variant=f'{aug} + {loss}', base='T00 (=B03)', seed=0, test_used=False,
                rule='best non-baseline augmentation + best non-baseline loss by val macro-F1; earliest id breaks ties',
                source_macro_f1={exp: rows[exp]['macro_f1_val'] for exp in ('T00','T03','T04','T05','T06')},
                note='Combination is tested even if the alternatives do not beat T00; it is not assumed to improve.')
    path = root / 'step2/combination_plan.json'
    if path.exists():
        if step1.read_json(path) != plan:
            raise ValueError('Combination choices changed; do not overwrite or resume a different T07')
    else:
        train.write_json(path, plan)
    return plan


def run_recipe(project, images_dir, labels_dir, exp_id):
    reference = baseline_record(project)
    step1.assert_environment(reference)
    if exp_id == 'T00':
        return collect_results(project)[0]
    cfg = make_config(project, images_dir, labels_dir, exp_id)
    folder = train.run_dir(cfg)
    _, _, overrides = spec(project, exp_id)
    if (folder / 'config.json').exists():
        saved = step1.read_json(folder / 'config.json')
        assert_run_record(reference, saved, overrides, exp_id)
    if (folder / 'summary.json').exists():
        print(f'{exp_id}: reuse completed result, no retraining.', flush=True)
    else:
        cfg.resume = (folder / 'last.pt').exists()
        train.run(cfg)
    rows = collect_results(project)
    return next(row for row in rows if row['exp_id'] == exp_id)


def paired_delta_ci(y_true, base_pred, new_pred, *, repeats=200, seed=0):
    """Stratified paired bootstrap on val; does NOT estimate training-seed std."""
    if repeats < 2:
        raise ValueError('Need at least two bootstrap replicates')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y_true == c) for c in range(9)]
    if any(len(group) == 0 for group in groups):
        raise ValueError('All nine classes must be represented')
    deltas = []
    for _ in range(repeats):
        indexes = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        def f1(pred):
            cm = train.ev.confusion_matrix(y_true[indexes], pred[indexes])
            return float(train.ev.per_class(cm)['f1'].mean())
        deltas.append(100*(f1(new_pred)-f1(base_pred)))
    lo, hi = np.quantile(deltas, [.025, .975])
    return dict(lo_pp=float(lo), hi_pp=float(hi), bootstrap_std_pp=float(np.std(deltas, ddof=1)),
                repeats=repeats, seed=seed, scope='val resampling only; not training-seed variation')


def write_report(project):
    root = Path(project)
    rows = collect_results(root)
    if any(row['status'] != 'Đủ kết quả' for row in rows):
        raise ValueError('Finish all six ablations and T07 before final recipe selection')
    best = max(rows, key=lambda row: row['macro_f1_val'])
    selection = dict(selected_recipe=best['exp_id'], source_exp_id='B03' if best['exp_id']=='T00' else best['exp_id'],
                     backbone='convnext_tiny', overrides=spec(root,best['exp_id'])[2], seed=0,
                     macro_f1_val=best['macro_f1_val'], delta_f1_pp=best['delta_f1_pp'],
                     rule='highest val macro-F1; T00 wins exact ties, then earliest experiment id',
                     single_seed=True, test_used=False)
    train.write_json(root / 'step2/selection.json', selection)
    baseline, _ = val_metrics(root, 'B03')
    uncertainty = {}
    for row in rows[1:]:
        pred, _ = val_metrics(root, row['exp_id'])
        uncertainty[row['exp_id']] = paired_delta_ci(baseline.y_true, baseline.y_pred, pred.y_pred)
    train.write_json(root / 'step2/val_bootstrap.json', uncertainty)
    lookup = {row['exp_id']: row for row in rows}
    combo = step1.read_json(root / 'step2/combination_plan.json')
    components = combo['selected_components']
    additive = sum(lookup[exp]['delta_f1_pp'] for exp in components)
    actual = lookup['T07']['delta_f1_pp']
    lines = ['# Bước 2 — Công thức huấn luyện ConvNeXt-Tiny', '',
             'T00 là kết quả B03 đã chạy: ConvNeXt-Tiny fb_in1k, seed 0, fold 0, 10 epoch, batch 32. '
             'Không train lại T00. T01–T06 đều bắt đầu từ khởi tạo quy định (ImageNet hoặc scratch), '
             'không tiếp tục từ checkpoint B03 hay từ một thí nghiệm khác.', '',
             '## Thiết kế có kiểm soát', '',
             'Ba trục, mỗi trục ba giá trị tính cả T00: init (finetune/scratch/frozen), '
             'augmentation (basic/color/CutMix), loss (CE/label smoothing 0,1/focal gamma 2). '
             'Không thay nền theo kiểu tham lam. T01–T06 chỉ đổi một yếu tố so với T00; '
             'T05 đổi cả tên loss và epsilon để mô tả cùng một yếu tố label smoothing. '
             'T07 là ngoại lệ được khai báo trước để kiểm tra kết hợp hai yếu tố.', '',
             'Giữ backbone, tag, seed, kích thước ảnh, optimizer, LR, WD, warmup/cosine, epoch/batch '
             'và cách chọn checkpoint; frozen không cập nhật backbone. B01 dùng AdamW, LR backbone/head '
             '1e-4/1e-3, WD 0,05 trừ norm/bias; các lượt T kế thừa đúng công thức này. '
             'Train crop+lật, val resize 256/crop 224; mean/std ImageNet và nội suy bicubic. '
             'AMP train, FP32 val. Chỉ val; không đánh giá test.', '',
             '| exp_id | Trục | Biến thể | F1 val | Δ pp | Top-1 | ECE | NLL |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['exp_id']} | {row['axis']} | {row['variant']} | {row['macro_f1_val']:.2%} | "
                     f"{row['delta_f1_pp']:+.3f} | {row['top1_val']:.2%} | {row['ece_val']:.5f} | {row['nll_val']:.5f} |")
    lines += ['', '## Kết quả từng trục và các lớp khó', '']
    for axis in ('Initialization','Augmentation','Loss'):
        alternatives = [row for row in rows if row['axis'].endswith(axis)]
        winner = max([rows[0], *alternatives], key=lambda row: row['macro_f1_val'])
        lines.append(f"- {axis}: giá trị tốt nhất theo val là {winner['exp_id']} ({winner['variant']}), "
                     f"F1 {winner['macro_f1_val']:.2%}, Δ {winner['delta_f1_pp']:+.3f} pp.")
        for row in alternatives:
            lines.append(f"  - {row['exp_id']}: F1 Chinee Apple {row['f1_chinee_apple']:.2%}, "
                         f"Snake Weed {row['f1_snake_weed']:.2%}; T00 lần lượt "
                         f"{rows[0]['f1_chinee_apple']:.2%}, {rows[0]['f1_snake_weed']:.2%}.")
    lines += ['', 'Chi tiết cả chín lớp và Δ với T00 nằm trong [per_class_f1.csv](per_class_f1.csv). '
              'Loss focal không dùng alpha/trọng số lớp; do đó không giả định tự động cải thiện lớp hiếm. '
              'Color và CutMix phải được đánh giá qua val; train accuracy với nhãn trộn không được dùng để chọn model.', '',
              '## Kiểm tra kết hợp', '',
              f"T07 kết hợp {components[0]} và {components[1]}, được chọn giữa các biến thể mới của mỗi trục "
              'bằng macro-F1 val, giữ init=finetune. Có thể cả hai biến thể vẫn thua T00; '
              'thử kết hợp không đồng nghĩa đã chứng minh chúng tốt.',
              f"Tổng Δ của hai thí nghiệm riêng: {additive:+.3f} pp; Δ thực của T07: {actual:+.3f} pp; "
              f"phần khác tổng đơn giản: {actual-additive:+.3f} pp. Đây là mô tả tương tác trên một seed, "
              'không phải kiểm định hiệu ứng cộng dồn. Kế hoạch được lưu trước khi train trong combination_plan.json.', '',
              '## Δ và giới hạn nhiễu', '',
              'Mỗi cấu hình chỉ có một seed; std giữa các seed chưa được đo. Không thể kết luận ưu thế '
              'ổn định chỉ từ Δ nhỏ. Khoảng dưới đây là bootstrap ghép cặp/phân tầng trên cùng ảnh val '
              '(200 lần, seed 0), đo biến thiên theo mẫu val, không thay thế std huấn luyện. '
              'Các cấu hình đã được chọn trên val nên các khoảng này chỉ mang tính mô tả, không phải '
              'bằng chứng xác nhận sau lựa chọn. Mean ± std giữa seed được báo cáo ở Bước 4.', '',
              '| exp_id | Δ pp | Khoảng bootstrap 95% (pp) | Bootstrap std (pp) |', '|---|---:|---:|---:|']
    for row in rows[1:]:
        u = uncertainty[row['exp_id']]
        lines.append(f"| {row['exp_id']} | {row['delta_f1_pp']:+.3f} | [{u['lo_pp']:+.3f}, {u['hi_pp']:+.3f}] | {u['bootstrap_std_pp']:.3f} |")
    lines += ['', '## Hội tụ và khả năng tổng quát hóa', '',
              'Raw loss giữa CE/label smoothing/focal không cùng thang ý nghĩa; so sánh chất lượng '
              'bằng macro-F1, top-1, NLL chung từ xác suất và ECE. Đường cong loss dùng để theo dõi '
              'từng run, không dùng loss focal thấp hơn CE làm bằng chứng tốt hơn.']
    for row in rows:
        source = 'B03' if row['exp_id']=='T00' else row['exp_id']
        h = pd.read_csv(root / 'runs' / source / 'seed0/history.csv')
        lines.append(f"- {row['exp_id']}: checkpoint epoch {row['best_epoch']}; val loss "
                     f"{h.iloc[0].val_loss:.4f} → {h.iloc[-1].val_loss:.4f}; "
                     f"train {row['train_seconds_epoch']:.2f} s/epoch.")
    lines += ['', f"Chọn **{best['exp_id']} ({best['variant']})** cho bước suy luận: macro-F1 val "
              f"**{best['macro_f1_val']:.2%}**, Δ so với T00 **{best['delta_f1_pp']:+.3f} pp**. "
              'Đây là quyết định screening bằng điểm val, chưa có std giữa seed.', '',
              '## Bằng chứng', '', 'Ảnh T00 được vẽ từ log B03 với tiêu đề T00; không có lượt train T00 mới.']
    for row in rows:
        lines.append(f"![{row['exp_id']}](../{row['curve_path']})")
    lines += ['', 'Cấu hình, log, logits và checkpoint: runs/<source_exp_id>/seed0. '
              'Đường cong: curves/. Dự đoán chỉ val: predictions/. Không mở test để chọn cấu hình.']
    (root / 'step2/step2_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return selection


def export_colab_xlsx(project):
    rows = collect_results(project)
    return step1.write_colab_sheet(project, rows, COLUMNS, 'Training', [
        'T00 dùng lại B03. Seed 0, 10 epoch, batch 32; ba trục có kiểm soát, không chọn bằng test.',
        'Δ macro-F1 là điểm phần trăm so với T00; chưa có std giữa các seed.',
        'Nguồn: config/history/val logits/dự đoán trong runs/<exp_id>/seed0 và predictions/.'])


def export_artifacts(project, out_path):
    import zipfile
    root = Path(project)
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for folder in ('code','tests','starter','step0','step1','step2','runs','curves','predictions'):
            for path in sorted((root / folder).rglob('*')):
                if path.is_file() and path.suffix in ('.py','.md','.json','.csv','.png','.npy','.ipynb'):
                    bundle.write(path, path.relative_to(root))
        for name in ('eval.py','results.xlsx','tools/gpu_budget.py'):
            bundle.write(root / name, name)
    return Path(out_path)


def export_checkpoints(project, out_path, *, include_last=False):
    """Keep experiment paths in the ZIP so downloaded best.pt files cannot collide."""
    import zipfile
    root = Path(project)
    paths = [root / f'runs/{exp}/seed0/{name}.pt' for exp in [*EXPERIMENTS, 'T07']
             for name in (('best','last') if include_last else ('best',))]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Finish all recipes before checkpoint export: {missing}')
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_STORED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(root))
    return Path(out_path)
