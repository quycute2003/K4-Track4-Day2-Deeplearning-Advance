"""Validate the downloaded val-only inference evidence before importing it."""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code'))
import train,step3


def main():
    directory=ROOT/'runs/step3_import'
    archive=directory/'step3_artifacts.zip'
    with zipfile.ZipFile(archive) as bundle:
        names=bundle.namelist()
        if bundle.testzip() is not None or len(names)!=len(set(names)):
            raise ValueError('Corrupt or duplicate ZIP entries')
        for name in names:
            if not (ROOT/name).resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Archive path escapes project')
        if any(name.startswith('predictions/') and '_test' in name for name in names):
            raise ValueError('Step 3 must not contain test predictions')
        if bundle.read('eval.py')!=(ROOT/'eval.py').read_bytes():
            raise ValueError('Official evaluator changed')
        for name in names:
            if name.startswith('code/') and name.endswith('.py') and bundle.read(name)!=(ROOT/name).read_bytes():
                raise ValueError(f'Inference source differs: {name}')
            if name.startswith(('runs/B','runs/T','predictions/B','predictions/T','curves/B','curves/T')):
                if not (ROOT/name).is_file() or bundle.read(name)!=(ROOT/name).read_bytes():
                    raise ValueError(f'Screening evidence differs: {name}')
        source=json.loads(bundle.read('step3/source.json'))
        if source!=step3._source(ROOT):
            raise ValueError('Checkpoint/logit/protocol provenance differs')
        with tempfile.TemporaryDirectory(prefix='deepweeds_step3_import_') as temporary:
            stage=Path(temporary)
            for info in bundle.infolist():
                if info.is_dir():
                    continue
                path=(stage/info.filename).resolve()
                if not path.is_relative_to(stage.resolve()):
                    raise ValueError('Invalid staging path')
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(bundle.read(info))
            supplied=json.loads(bundle.read('step3/inference.json'))
            rows=step3.collect_results(stage)
            if rows!=supplied or len(rows)!=9 or any(row['status']!='Đủ kết quả' for row in rows):
                raise ValueError('Incomplete or inconsistent inference table')
            if len(step3.step1.read_json(stage/'step3/latency.json'))!=18:
                raise ValueError('Require nine methods measured at two batch sizes')
            labels=ROOT/'runs/step0_validation/labels/val_subset0.csv'
            for row in rows:
                pred=train.ev.read_pred(str(stage/row['prediction_path']))
                train.ev.check_against_csv(pred,str(labels),'val')
            saved=step3.step1.read_json(stage/'step3/selection.json')
            checked=step3.write_report(stage)
            if saved!=checked:
                raise ValueError('Selection differs from recomputed val results')
            if not any(row['p95_ms']<=100 for row in rows):
                raise ValueError('No configuration meets the declared p95 target')
            probe=step3.step1.read_json(stage/'step3/model_check.json')
            if (probe['n']!=3501 or not probe['numerically_aligned'] or probe['test_used']
                    or probe['runtime']!=saved['runtime'] or probe['checkpoint_sha256']!=source['checkpoint_sha256']):
                raise ValueError('Model reload verification is missing or inconsistent')
        selected=[name for name in names if name.startswith(('step3/','predictions/I')) and not name.endswith('/')]
        for name in selected:
            target=ROOT/name
            if ('latency_batch' in name or name=='step3/runtime.json') and target.exists() and target.read_bytes()!=bundle.read(name):
                raise FileExistsError(f'Another completed runtime/timing already exists: {name}')
        backup=directory/f'before_import_{datetime.now():%Y%m%d_%H%M%S}'
        copied=[]
        for name in selected:
            target=ROOT/name; content=bundle.read(name)
            if target.exists() and target.read_bytes()==content:
                continue
            if target.exists():
                previous=backup/name
                previous.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(target,previous)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
            copied.append(name)
        backup.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'results.xlsx',backup/'results.xlsx')
        train.write_json(directory/'import_manifest.json',dict(archive_sha256=step3.sha256(archive),
            imported=copied,selection=checked,methods=9,timing_records=18,val_csv_checked=True,test_used=False))
    print(f'Imported {len(copied)} files; all 9 val results and 18 raw timing records verified.')
    print(json.dumps(checked,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
