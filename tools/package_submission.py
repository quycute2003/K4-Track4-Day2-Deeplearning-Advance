"""Package the six deliverables and their small evidence; exclude data and checkpoints."""
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    directories = {'code', 'curves', 'predictions', 'notebooks', 'step0', 'step1', 'step2', 'step3', 'step4', 'step5', 'tools', 'tests', 'starter'}
    root_files = {'report.md', 'results.xlsx', 'REPRODUCE.md', 'eval.py', 'GPU_BUDGET.md', 'GUIDE.md', 'RUBRIC.md', '.gitattributes', '.gitignore'}
    names = [name for name in tracked if name and (name.split('/')[0] in directories or name in root_files)]
    assert names and all(Path(name).suffix.lower() not in ('.pt', '.zip', '.ckpt', '.pth', '.safetensors') for name in names)
    out = ROOT / 'runs/step5_validation/deepweeds_submission.zip'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / 'REPRODUCE.md', 'README.md')
        for name in names:
            archive.write(ROOT / name, name)
    with zipfile.ZipFile(out) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert {'README.md', 'report.md', 'results.xlsx', 'eval.py'} <= set(archive.namelist())
        assert len([n for n in archive.namelist() if n.startswith('predictions/F') and n.endswith('_test.csv')]) == 9
    print(f'Submission ZIP: {out.name}; {len(names) + 1} files; {out.stat().st_size / (1 << 20):.1f} MiB; no dataset or checkpoints.')


if __name__ == '__main__':
    main()
