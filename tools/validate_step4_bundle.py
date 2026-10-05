"""Check the actual Lightning bootstrap in an isolated folder, without GPU/test data."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    nb = json.loads((ROOT / 'notebooks/step4_lightning.ipynb').read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and not cell['outputs']
            compile(''.join(cell['source']), cell['id'], 'exec')
    bootstrap = ''.join(next(c for c in nb['cells'] if c['id'] == 'bootstrap')['source'])
    prefix = bootstrap.split('CODE_DIR=')[0]
    with tempfile.TemporaryDirectory(prefix='deepweeds_step4_check_') as temp:
        directory = Path(temp)
        project = directory / 'deepweeds_step3_lightning/project'
        for supplied, name in [(ROOT / 'runs/step3_import/step3_artifacts.zip', 'step3_artifacts.zip'),
                               (ROOT / 'notebooks/step4_bundle.zip', 'step4_bundle.zip')]:
            shutil.copyfile(supplied, directory / name)
        namespace = dict(NOTEBOOK_DIR=directory, PROJECT_DIR=project, Path=Path,
                         hashlib=hashlib, json=json, shutil=shutil, sys=sys)
        # The same bootstrap must work on a fresh Studio and when its cell is rerun.
        exec(prefix, namespace)
        exec(prefix, namespace)
        frozen = project / 'step4/frozen_plan.json'
        plan = json.loads(frozen.read_bytes())
        # Simulate freeze_plan() serializing the Windows ZIP's CRLF JSON on Linux.
        linux_bytes = json.dumps(plan, ensure_ascii=False, indent=2).encode('utf-8')
        frozen.write_bytes(linux_bytes)
        before = hashlib.sha256(frozen.read_bytes()).hexdigest()
        exec(prefix, namespace)
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before
        # Formatting differences are harmless; actual protocol changes must stop.
        changed = dict(plan, seeds=[0, 1, 3])
        frozen.write_bytes(json.dumps(changed, ensure_ascii=False).encode('utf-8'))
        try:
            exec(prefix, namespace)
        except FileExistsError:
            assert json.loads(frozen.read_bytes()) == changed
        else:
            raise AssertionError('Changed seed protocol was silently accepted')
        frozen.write_bytes(linux_bytes)
        print('CRLF/LF resume keeps the plan hash; genuine seed changes are rejected.')
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'runs/step0_dependencies'))
        command = [sys.executable, '-X', 'utf8', '-c',
                   "import torch,unittest; torch.set_num_threads(2); "
                   "s=unittest.defaultTestLoader.discover('tests'); "
                   "r=unittest.TextTestRunner(verbosity=1).run(s); raise SystemExit(not r.wasSuccessful())"]
        result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True)
        print(result.stdout, result.stderr)
        result.check_returncode()
        check = f"""import hashlib,json,shutil,sys
from pathlib import Path
NOTEBOOK_DIR=Path({str(directory)!r})
PROJECT_DIR=Path.cwd()
before=hashlib.sha256((PROJECT_DIR/'step4/frozen_plan.json').read_bytes()).hexdigest()
# Run the actual full bootstrap, omitting only the GPU runtime probe.
exec({bootstrap.replace("print('Runtime:',step4.runtime_check(PROJECT_DIR))", '')!r})
assert hashlib.sha256((PROJECT_DIR/'step4/frozen_plan.json').read_bytes()).hexdigest()==before
p=step4.plan_at('.')
assert p['seeds']==[0,1,2] and p['groups']==['F00','F01'] and p['test_used'] is False
for group in step4.GROUPS:
    for seed in step4.SEEDS:
        cfg=step4.make_config('.','images','labels',group,seed)
        assert (cfg.seed,cfg.epochs,cfg.batch_size,cfg.img_size)==(seed,10,32,224)
        assert cfg.save_test_predictions is False
assert not list(Path('predictions').glob('F*_test.csv'))
print('Frozen plan and six paired configs verified; no GPU or real test was run.')
"""
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', check], cwd=project,
                                env=env, capture_output=True, text=True)
        print(result.stdout, result.stderr)
        result.check_returncode()
    print('Clean notebook, ZIP hashes, fresh/repeated bootstrap and isolated suite passed.')


if __name__ == '__main__':
    main()
