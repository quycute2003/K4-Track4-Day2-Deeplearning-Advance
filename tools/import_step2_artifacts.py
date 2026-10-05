"""Validate downloaded Step 2 archives, then import evidence without replacing code."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import train, step2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    archive_dir = ROOT / 'runs/step2_import'
    artifacts = archive_dir / 'step2_artifacts.zip'
    checkpoints = archive_dir / 'step2_best_checkpoints.zip'
    expected = {f'runs/T{i:02d}/seed0/best.pt' for i in range(1,8)}
    with zipfile.ZipFile(artifacts) as bundle, zipfile.ZipFile(checkpoints) as weights:
        for z in (bundle, weights):
            if z.testzip() is not None:
                raise ValueError('Corrupt downloaded archive')
            if len(z.namelist()) != len(set(z.namelist())):
                raise ValueError('Duplicate archive entries')
            for name in z.namelist():
                if not (ROOT/name).resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError('Archive path escapes the project')
        if set(weights.namelist()) != expected:
            raise ValueError('Expected precisely seven best checkpoints T01–T07')
        if any(name.startswith('predictions/') and '_test' in name for name in bundle.namelist()):
            raise ValueError('Unexpected test predictions in screening')
        if bundle.read('eval.py') != (ROOT/'eval.py').read_bytes():
            raise ValueError('Downloaded eval.py differs from the official evaluator')
        # Fully validate true evidence in isolation before writing project results.
        with tempfile.TemporaryDirectory(prefix='deepweeds_step2_import_') as temp:
            stage = Path(temp)
            for info in bundle.infolist():
                if info.is_dir():
                    continue
                target = (stage/info.filename).resolve()
                if not target.is_relative_to(stage.resolve()):
                    raise ValueError('Invalid staging path')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(info))
            rows = step2.collect_results(stage)
            if any(row['status'] != 'Đủ kết quả' for row in rows):
                raise ValueError('Not all recipes have finished')
            saved_selection = step2.step1.read_json(stage/'step2/selection.json')
            checked_selection = step2.write_report(stage)
            if saved_selection != checked_selection:
                raise ValueError('Recipe selection differs from recomputed val results')
            labels = ROOT/'runs/step0_validation/labels/val_subset0.csv'
            if not labels.is_file():
                raise FileNotFoundError('Need original local fold-0 val CSV for verification')
            for source in ['B03',*[f'T{i:02d}' for i in range(1,8)]]:
                pred, _ = step2.val_metrics(stage,source)
                train.ev.check_against_csv(pred,str(labels),'val')
            # B-stage evidence must agree byte-for-byte with the existing originals.
            for name in bundle.namelist():
                if name.startswith(('runs/B','predictions/B','curves/B')):
                    target = ROOT/name
                    if not target.is_file() or target.read_bytes() != bundle.read(name):
                        raise ValueError(f'Existing backbone evidence differs: {name}')
        selected = []
        for name in bundle.namelist():
            if name.startswith(('runs/T','curves/T','predictions/T','step2/')):
                selected.append((name,bundle.read(name)))
        # Refuse replacing a previous completed experiment with different results.
        for name, content in selected:
            target=ROOT/name
            if name.startswith(('runs/T','predictions/T')) and target.exists() and target.read_bytes()!=content:
                raise FileExistsError(f'Another result already exists: {target}')
        for name in expected:
            target=ROOT/name
            if target.exists():
                with weights.open(name) as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=digest(target):
                        raise FileExistsError(f'Another checkpoint already exists: {target}')
        backup=archive_dir/f'before_import_{datetime.now():%Y%m%d_%H%M%S}'
        copied=[]
        for name,content in selected:
            target=ROOT/name
            if target.exists() and target.read_bytes()==content:
                continue
            if target.exists():
                old=backup/name
                old.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(target,old)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
            copied.append(name)
        backup.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'results.xlsx',backup/'results.xlsx')
        checkpoint_hashes={}
        for name in sorted(expected):
            target=ROOT/name
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                with weights.open(name) as source, target.open('wb') as output:
                    shutil.copyfileobj(source,output)
            checkpoint_hashes[name]=digest(target)
        manifest=dict(archives={p.name:digest(p) for p in (artifacts,checkpoints)},
                      imported_evidence=copied,checkpoint_sha256=checkpoint_hashes,
                      selection=checked_selection,official_val_checked=True,test_used=False)
        train.write_json(archive_dir/'import_manifest.json',manifest)
    print(f'Imported {len(copied)} evidence files and 7 checkpoints. Original code and B-stage preserved.')
    print(json.dumps(checked_selection,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
