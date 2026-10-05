"""Build a standalone Colab notebook reusing the real B01 artifacts."""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

from build_step0_notebook import cell, notebook, save

ROOT = Path(__file__).resolve().parent.parent


def main():
    files = [ROOT / 'eval.py', ROOT / 'code/README.md', ROOT / 'code/lab_day2.ipynb',
             ROOT / 'starter/lab_day2.ipynb', ROOT / 'tools/gpu_budget.py']
    for folder in ('code', 'tests', 'starter'):
        files.extend(sorted((ROOT / folder).glob('*.py')))
    # Embed real, small B01 results; checkpoints remain separate downloads/uploads.
    files.extend(sorted(p for p in (ROOT / 'runs/B01/seed0').glob('*')
                        if p.suffix in ('.json', '.csv', '.npy')))
    files.extend([ROOT / 'curves/B01_resnet50_seed0.png', ROOT / 'predictions/B01_seed0_val.csv'])
    data = {p.relative_to(ROOT).as_posix(): base64.b64encode(p.read_bytes()).decode()
            for p in files}
    # The full test suite also reads the original starter notebook and imports
    # the budget utility; a fresh runtime has no repository files to fall back on.
    required = {'starter/lab_day2.ipynb', 'tools/gpu_budget.py', 'eval.py',
                'code/_package.py', 'tests/test_starter.py', 'tests/test_gpu_budget.py'}
    if not required.issubset(data):
        raise ValueError(f'Incomplete standalone bundle: {sorted(required - data.keys())}')
    packed = base64.b64encode(zlib.compress(json.dumps(data).encode())).decode()
    cells = [cell('markdown', 'intro', '''# Bước 1 — So sánh 5 backbone trên val
Giữ B01 ResNet50 đã chạy; train B02 ResNeXt50, B03 ConvNeXt-Tiny, B04 DeiT-Small,
B05 EfficientNet-B0. Cùng T00: seed 0, 10 epoch, batch 32, input 224.
ConvNeXt dùng tag fb_in1k rõ ràng, không lấy mặc định ImageNet-12k của timm.

Upload notebook này, chọn **T4 GPU**, chạy từng ô từ trên xuống.
Notebook đã kèm kết quả B01 thật. Nếu phiên mới chưa có checkpoint, ô B01 sẽ yêu cầu
upload **best.pt của B01** (file bạn đã tải ở Bước 0), chỉ để đo latency, không train lại.
Nếu phiên hiện tại còn `/content/deepweeds_step0/runs/B01/seed0/best.pt`, dùng lại luôn.

Mỗi model có một ô train riêng. Chạy lại ô sẽ dùng kết quả đã hoàn thành hoặc resume
checkpoint của lượt bị ngắt với cùng cấu hình. File nằm trên đĩa phiên Colab;
tải checkpoint/ZIP trước khi phiên bị xóa. Không có lượt đánh giá test.
'''), cell('code', 'setup', '''import sys, subprocess, json, base64, zlib, hashlib
from pathlib import Path
result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'timm==1.0.30', 'fvcore==0.1.5.post20221221', 'matplotlib', 'pandas',
    'pillow', 'scikit-learn', 'openpyxl'], capture_output=True, text=True)
print(result.stdout, result.stderr)
result.check_returncode()
BASE_DIR = Path('/content')
PROJECT_DIR = BASE_DIR / 'deepweeds_step0'
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
'''), cell('code', 'bootstrap', f'''# Khôi phục code và kết quả B01; không ghi đè kết quả khác nội dung.
sources = json.loads(zlib.decompress(base64.b64decode({packed!r})))
for relative, encoded in sources.items():
    target = PROJECT_DIR / relative
    content = base64.b64decode(encoded)
    if relative.startswith(('runs/', 'curves/', 'predictions/')) and target.exists():
        if target.read_bytes() != content:
            raise FileExistsError(f'Kết quả cũ khác bundle: {{target}}')
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
assert hashlib.sha256((PROJECT_DIR / 'eval.py').read_bytes()).hexdigest() == {hashlib.sha256((ROOT / 'eval.py').read_bytes()).hexdigest()!r}
CODE_DIR = PROJECT_DIR / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
for name in ['dataset', 'losses', 'model', 'train', 'step0', 'step1']:
    sys.modules.pop(name, None)
import dataset, losses, model, train, step0, step1
reference = step1.baseline_config(PROJECT_DIR)
step1.assert_environment(reference)
print('GPU/phiên bản khớp B01; T00 giữ nguyên.')
'''), cell('code', 'tests', '''command = [sys.executable, '-X', 'utf8', '-c',
    "import torch,unittest; torch.set_num_threads(2); "
    "s=unittest.defaultTestLoader.discover('tests'); "
    "r=unittest.TextTestRunner(verbosity=1).run(s); raise SystemExit(not r.wasSuccessful())"]
result = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True)
print(result.stdout, result.stderr)
result.check_returncode()
'''), cell('code', 'data', '''IMAGES_DIR, LABELS_DIR = step0.prepare_data(BASE_DIR)
frames = dataset.load_split(LABELS_DIR)
split_checks = dataset.check_split(*frames, IMAGES_DIR)
print('Dùng CSV fold 0 gốc; chỉ train/val trong các lượt chạy.')
'''), cell('code', 'b01', '''# Nếu phiên mới: chọn best.pt của B01 trong máy bạn.
checkpoint = PROJECT_DIR / 'runs/B01/seed0/best.pt'
if not checkpoint.exists():
    from google.colab import files
    print('Upload best.pt của B01 (không cần last.pt).')
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('Chỉ chọn một file best.pt của B01.')
    filename, content = next(iter(uploaded.items()))
    if not filename.endswith('.pt'):
        raise ValueError('Cần file checkpoint .pt.')
    checkpoint.write_bytes(content)
    del uploaded, content
summary = step1.run_backbone(PROJECT_DIR, IMAGES_DIR, LABELS_DIR, 'B01')
step1.export_colab_xlsx(PROJECT_DIR)
print('B01 giữ nguyên kết quả train; đã đo latency trên GPU hiện tại.')
''')]
    for exp_id, backbone, _ in [('B02', 'resnext50_32x4d', ''),
                              ('B03', 'convnext_tiny', ''),
                              ('B04', 'deit_small_patch16_224', ''),
                              ('B05', 'efficientnet_b0', '')]:
        cells.extend([cell('markdown', exp_id.lower()+'-title', f'## {exp_id} — {backbone}\nCùng công thức T00 với B01.\n'),
                      cell('code', exp_id.lower(), f'''summary = step1.run_backbone(PROJECT_DIR, IMAGES_DIR, LABELS_DIR, {exp_id!r})
step1.export_colab_xlsx(PROJECT_DIR)
from IPython.display import display, Image as DisplayImage
display(DisplayImage(filename=str(PROJECT_DIR / 'curves/{exp_id}_{backbone}_seed0.png')))
print(json.dumps(summary, ensure_ascii=False, indent=2))
''')])
    cells.extend([cell('code', 'compare', '''rows = step1.collect_results(PROJECT_DIR)
display(__import__('pandas').DataFrame(rows))
try:
    step1.capture_imagenet_reference(PROJECT_DIR)
except (OSError, ValueError) as exc:
    print('Chưa tải được bảng tham khảo ImageNet:', exc)
selection = step1.write_report(PROJECT_DIR)
step1.export_colab_xlsx(PROJECT_DIR)
print(json.dumps(selection, ensure_ascii=False, indent=2))
print((PROJECT_DIR / 'step1/step1_report.md').read_text(encoding='utf-8'))
'''), cell('code', 'export', '''archive = step1.export_artifacts(PROJECT_DIR, BASE_DIR / 'step1_artifacts.zip')
print('Tải:', archive)
print('Excel:', PROJECT_DIR / 'results.xlsx', '(đã nằm trong ZIP)')
print('Checkpoint tải riêng: runs/B02..B05/seed0/best.pt và last.pt')
'''), cell('markdown', 'next', '''## Lưu kết quả
Tải `/content/step1_artifacts.zip`; ZIP chứa `results.xlsx`, báo cáo, code,
log và biểu đồ của cả năm backbone, dự đoán val. Không chứa checkpoint.
Giữ riêng best.pt/last.pt mỗi model, và tải notebook có output qua File → Download → .ipynb.

Bảng còn thiếu số liệu hoặc chưa đủ năm backbone thì Bước 1 chưa hoàn thành.
Một seed/model chỉ dùng để screening; các chênh lệch nhỏ chưa chứng minh model tốt hơn.
Latency hiện là FP32/batch 1, một forward sau 10 warmup; Bước 3 mới đo p50/p95/p99.
''')])
    save(ROOT / 'notebooks/step1_colab.ipynb', notebook(cells, 'step1_colab.ipynb'))
    print('Built notebooks/step1_colab.ipynb')


if __name__ == '__main__':
    main()
