"""Build the one-file Colab notebook from the checked-in implementation.

Windows: py tools/build_step0_notebook.py
Rebuild after modifying any code/test files so the embedded source is current.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cell(kind, identity, text):
    result = {"cell_type": kind, "id": identity, "metadata": {},
              "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
        compile(text, identity, "exec")
    return result


def notebook(cells, name):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}, "colab": {"name": name}},
            "cells": cells}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    intro = cell("markdown", "intro", """# Bước 0 — Code, EDA và kiểm tra pipeline
Upload notebook này lên Colab, chọn Runtime → Change runtime type → T4 GPU, rồi Run all.
Notebook tự tạo code, dùng lại dữ liệu đo ngân sách nếu còn (kể cả đường dẫn /kaggle/working/data cũ).
Nó chạy unit tests, EDA, loss ban đầu, overfit một batch, rồi **B01 ResNet-50 10 epoch**, batch 32, seed 0.
Không đánh giá test. Chỉ chạy thí nghiệm thật nếu các kiểm tra pipeline đều đạt.
Sau khi xong, tải **step0_artifacts.zip** trong thanh Files. Checkpoint tải riêng nếu cần giữ lại.
""")
    install = cell("code", "install", """import sys, subprocess, json, hashlib, importlib.util
from pathlib import Path
result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'timm==1.0.30', 'fvcore==0.1.5.post20221221', 'matplotlib', 'pandas', 'pillow', 'scikit-learn'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(result.stdout)
result.check_returncode()
import torch
assert torch.cuda.is_available(), 'Bật GPU trong Runtime → Change runtime type.'
print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__)
""")
    paths = cell("code", "paths", """# Colab có thể chứa /kaggle, nên ưu tiên /content.
BASE_DIR = Path('/content') if Path('/content').is_dir() else Path('/kaggle/working')
PROJECT_DIR = BASE_DIR / 'deepweeds_step0'
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR = PROJECT_DIR / 'step0'
EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 0
RESUME = False  # Đặt True nếu lần B01 trước bị ngắt và last.pt còn tồn tại.
""")
    imports = cell("code", "imports", """# Các module top-level này cũng import được trong DataLoader worker (spawn/forkserver).
CODE_DIR = (PROJECT_DIR / 'code').resolve()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
for module_name in ['dataset', 'losses', 'model', 'train', 'step0']:
    sys.modules.pop(module_name, None)
import dataset, losses, model, train, step0
""")
    tests = cell("code", "unit-tests", """# Tests chạy CPU trên fixture nhỏ; các con số test không phải kết quả bài lab.
command = [sys.executable, '-X', 'utf8', '-c',
    "import torch,unittest; torch.set_num_threads(2); "
    "s=unittest.defaultTestLoader.discover('tests'); "
    "r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(not r.wasSuccessful())"]
result = subprocess.run(command, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True)
print(result.stdout)
result.check_returncode()
""")
    data = cell("code", "data", """IMAGES_DIR, LABELS_DIR = step0.prepare_data(BASE_DIR)
eda = step0.analyze_data(IMAGES_DIR, LABELS_DIR, EDA_DIR, seed=SEED)
from IPython.display import display, Image as DisplayImage
for filename in ['class_distribution.png', 'samples_3_per_class.png',
                 'augmentation_examples.png', 'mixup_examples.png', 'cutmix_examples.png']:
    display(DisplayImage(filename=str(EDA_DIR / filename)))
""")
    checks = cell("code", "pipeline-checks", """checks = step0.pipeline_checks(IMAGES_DIR, LABELS_DIR, EDA_DIR,
    backbone='resnet50', seed=SEED)
assert checks['passed'], 'Dừng lại nếu pipeline không qua kiểm tra.'
display(DisplayImage(filename=str(EDA_DIR / 'overfit_curve.png')))
print(json.dumps(checks, ensure_ascii=False, indent=2))
""")
    baseline = cell("code", "baseline", """# B01 là lần train chính thức đầu tiên, không dùng trọng số từ phép overfit.
assert checks['passed'] and eda['reference_filenames_match']
cfg = train.Config(exp_id='B01', backbone='resnet50', seed=SEED,
    epochs=EPOCHS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    images_dir=str(IMAGES_DIR), labels_dir=str(LABELS_DIR),
    out_dir=str(PROJECT_DIR / 'runs'), pred_dir=str(PROJECT_DIR / 'predictions'),
    curves_dir=str(PROJECT_DIR / 'curves'), save_test_predictions=False, resume=RESUME)
summary = train.run(cfg)
assert not summary['test_evaluated']
prediction = train.ev.read_pred(summary['val_predictions'])
train.ev.check_against_csv(prediction, str(LABELS_DIR / 'val_subset0.csv'), 'val')
print(json.dumps(summary, ensure_ascii=False, indent=2))
display(DisplayImage(filename=summary['curve']))
""")
    output = cell("code", "export", """step0.write_report(EDA_DIR, eda, checks, summary)
archive = step0.export_artifacts(PROJECT_DIR, BASE_DIR / 'step0_artifacts.zip')
print('Tải file này trong thanh Files:', archive)
print('Checkpoint nếu cần lưu riêng:', train.run_dir(cfg) / 'best.pt',
      'và', train.run_dir(cfg) / 'last.pt')
""")
    ending = cell("markdown", "next", """## Đọc kết quả và bước tiếp theo
ZIP gồm code, unit tests, step0_report.md, bảng đếm lớp, ảnh EDA/augmentation/overfit,
history.csv, config.json, val_logits.npy, file dự đoán val và đường cong B01.
Đọc ảnh mẫu rồi bổ sung nhận xét trong báo cáo: Chinee Apple/Snake Weed, Negatives, và crop/CutMix.
Chưa có kết luận backbone nào tốt nhất; Bước 1 chạy 4 backbone còn lại với **cùng** công thức, seed và số epoch.
Không chạy notebook đo ngân sách lần nữa. Không bật save_test_predictions ở Bước 0–3.
Nếu phiên bị ngắt mà last.pt còn tồn tại, giữ nguyên cấu hình, đặt RESUME=True và chạy lại ô baseline.
Nếu đổi batch/epoch cho B01, các backbone sau phải dùng cùng cấu hình để so sánh công bằng.
""")
    # The repo notebook expects the full source tree, so it needs no embedded bundle.
    repo_paths = cell("code", "paths", """PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / 'eval.py').is_file():
    PROJECT_DIR = PROJECT_DIR.parent
assert (PROJECT_DIR / 'code/train.py').is_file(), 'Chạy notebook trong repo hoặc chỉnh PROJECT_DIR.'
BASE_DIR = Path('/content') if Path('/content').is_dir() else PROJECT_DIR
EDA_DIR = PROJECT_DIR / 'step0'
EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 0
RESUME = False
""")
    repo_cells = [intro, install, repo_paths, imports, tests, data, checks, baseline, output, ending]
    save(ROOT / "code/lab_day2.ipynb", notebook(repo_cells, "lab_day2.ipynb"))
    source_paths = [ROOT / "eval.py", ROOT / "tools/gpu_budget.py", ROOT / "starter/lab_day2.ipynb",
                    ROOT / "code/lab_day2.ipynb", ROOT / "code/README.md"]
    for folder in ("code", "starter", "tests"):
        source_paths.extend(sorted((ROOT / folder).glob("*.py")))
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): p.read_text(encoding="utf-8")
               for p in source_paths}
    encoded = base64.b64encode(zlib.compress(json.dumps(sources, ensure_ascii=False).encode("utf-8"))).decode("ascii")
    digest = hashlib.sha256((ROOT / "eval.py").read_bytes()).hexdigest()
    # Preserve original eval bytes, including line endings, not just the text content.
    eval_bytes = base64.b64encode((ROOT / "eval.py").read_bytes()).decode("ascii")
    bootstrap_text = f'''# Tạo các file Python có thể đọc/sửa trong Files → deepweeds_step0/code.
import base64, zlib
sources = json.loads(zlib.decompress(base64.b64decode({encoded!r})))
for relative, source in sources.items():
    target = PROJECT_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding='utf-8')
(PROJECT_DIR / 'eval.py').write_bytes(base64.b64decode({eval_bytes!r}))
assert hashlib.sha256((PROJECT_DIR / 'eval.py').read_bytes()).hexdigest() == {digest!r}
print('Code đã sẵn sàng; eval.py khớp SHA-256 của bản gốc.')
'''
    wrapper_cells = [intro, install, paths, cell("code", "bootstrap", bootstrap_text),
                     imports, tests, data, checks, baseline, output, ending]
    save(ROOT / "notebooks/step0_colab.ipynb", notebook(wrapper_cells, "step0_colab.ipynb"))
    print("Built notebooks/step0_colab.ipynb and code/lab_day2.ipynb")


if __name__ == "__main__":
    main()
