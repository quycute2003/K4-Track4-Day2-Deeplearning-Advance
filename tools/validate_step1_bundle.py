"""Run the exported notebook's tests from its bundle in a fresh directory.

Windows: py -X utf8 tools/validate_step1_bundle.py
Only unpack sources/results and execute CPU tests; no data downloads or training.
"""
from __future__ import annotations

import ast
import base64
import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'step1_colab.ipynb'
    nb = json.loads((ROOT / 'notebooks' / name).read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')
    bootstrap = next(c for c in nb['cells'] if c['id'] == 'bootstrap')
    source = ''.join(bootstrap['source'])
    is_light = name in ('step2_colab_light.ipynb','step3_colab_light.ipynb','step3_lightning.ipynb')
    is_step2 = name in ('step2_colab.ipynb','step2_colab_light.ipynb')
    is_step3 = name in ('step3_colab_light.ipynb','step3_lightning.ipynb')
    if name=='step3_lightning.ipynb':
        if 'google.colab' in json.dumps(nb) or '/content' in json.dumps(nb):
            raise ValueError('Lightning notebook must not use Colab APIs/paths')
    if is_light:
        tree = ast.parse(source)
        expected = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == 'EXPECTED_BUNDLE_SHA256')
        archive = ROOT / 'notebooks' / ('step3_bundle.zip' if is_step3 else 'step2_bundle.zip')
        with archive.open('rb') as stream:
            actual = hashlib.file_digest(stream,'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Light notebook and bundle ZIP do not match')
        with zipfile.ZipFile(archive) as bundle:
            sources = {info.filename:base64.b64encode(bundle.read(info)).decode()
                       for info in bundle.infolist() if not info.is_dir() and not info.filename.endswith('.pt')}
            checkpoint_names = [info.filename for info in bundle.infolist() if info.filename.endswith('.pt')]
            if is_step3 and checkpoint_names != ['runs/T05/seed0/best.pt']:
                raise ValueError('Step 3 requires exactly the T05 best checkpoint')
            if is_step2 and checkpoint_names:
                raise ValueError('Step 2 light bundle must not include checkpoints')
    else:
        first = ast.parse(source).body[0]
        encoded = ast.literal_eval(first.value.args[0].args[0].args[0])
        sources = json.loads(zlib.decompress(base64.b64decode(encoded)))
    if is_step2 or is_step3:
        for relative, content in sources.items():
            if relative.startswith(('code/','tests/','starter/','tools/')) or relative == 'eval.py':
                if base64.b64decode(content) != (ROOT / relative).read_bytes():
                    raise ValueError(f'Stale embedded source: {relative}; rebuild notebook')
        if any(relative.endswith('.pt') for relative in sources):
            raise ValueError('Checkpoints must not inflate the standalone notebook')
    with tempfile.TemporaryDirectory(prefix='deepweeds_step1_bundle_') as temp:
        target_root = Path(temp).resolve()
        if is_light:
            # Run the real ZIP loader, stopping before ML imports/GPU checks.
            # This also exercises re-running it with existing identical evidence.
            base = target_root / 'content'
            base.mkdir()
            shutil.copyfile(archive,base / archive.name)
            project = target_root / 'project'
            project.mkdir()
            namespace = dict(BASE_DIR=base,PROJECT_DIR=project,json=json,base64=base64,
                             hashlib=hashlib,Path=Path,shutil=shutil,os=os,NOTEBOOK_DIR=base)
            prefix = source[:source.index('CODE_DIR =')]
            exec(compile(prefix,'light-loader','exec'), namespace)
            exec(compile(prefix,'light-loader-repeat','exec'), namespace)
            target_root = project
        else:
            for relative, content in sources.items():
                target = (target_root / relative).resolve()
                if not target.is_relative_to(target_root):
                    raise ValueError(f'Invalid bundle path: {relative}')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(content))
        # Extra local ML dependencies may be installed in runs/, but no local
        # source folders are put on sys.path: all project imports use the bundle.
        command = [sys.executable, '-X', 'utf8', '-c',
                   'import torch,unittest; torch.set_num_threads(2); '
                   's=unittest.defaultTestLoader.discover("tests"); '
                   'r=unittest.TextTestRunner(verbosity=1).run(s); '
                   'raise SystemExit(not r.wasSuccessful())']
        result = subprocess.run(command, cwd=target_root, env=os.environ.copy(),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)
        result.check_returncode()
        if is_step2:
            # Verify actual embedded B03 evidence in addition to tiny test fixtures.
            check = [sys.executable, '-X', 'utf8', '-c',
                     'import sys,torch; torch.set_num_threads(2); sys.path.insert(0,"code"); '
                     'import step2; rows=step2.collect_results("."); '
                     'assert rows[0]["status"]=="Đủ kết quả"; '
                     'assert rows[0]["macro_f1_val"]>0; '
                     'assert all(row["macro_f1_val"] is None for row in rows[1:]); '
                     'print("Embedded B03 predictions/logits/metrics aligned; T00 verified; new runs blank.")']
            subprocess.run(check, cwd=target_root, env=os.environ.copy(), check=True)
        if is_step3:
            check = [sys.executable,'-X','utf8','-c',
                'import sys,torch; torch.set_num_threads(2); sys.path.insert(0,"code"); '
                'import step3; rows=step3.prepare_offline("."); by={r["exp_id"]:r for r in rows}; '
                'assert by["I00"]["macro_f1_val"]==by["I07"]["macro_f1_val"]; '
                'assert by["I07"]["ece_val"]<by["I00"]["ece_val"]; '
                'assert by["I07"]["nll_val"]<by["I00"]["nll_val"]; '
                'assert all(r["p95_ms"] is None for r in rows); '
                'assert step3.sha256("runs/T05/seed0/best.pt")==step3.step1.read_json("step3/source.json")["checkpoint_sha256"]; '
                'print("Real T05 checkpoint/cache verified; calibration improved; all GPU timings remain blank.")']
            subprocess.run(check,cwd=target_root,env=os.environ.copy(),check=True)
            # Re-run bootstrap after cached output is written, including tiny
            # diagnostic float changes and an enriched previous report.
            report=target_root/'step2/step2_report.md'
            report.write_text(report.read_text(encoding='utf-8')+'\nLocal analysis note\n',encoding='utf-8')
            temperature=target_root/'step3/temperature.json'
            diagnostic=json.loads(temperature.read_text(encoding='utf-8'))
            diagnostic['after']['ece']+=1e-14
            temperature.write_text(json.dumps(diagnostic),encoding='utf-8')
            exec(compile(prefix,'light-loader-after-cache','exec'),namespace)
            assert json.loads(temperature.read_text(encoding='utf-8'))['after']['ece']==diagnostic['after']['ece']
    print('Fresh-directory bundle validation passed; no dataset or GPU training used.')


if __name__ == '__main__':
    main()
