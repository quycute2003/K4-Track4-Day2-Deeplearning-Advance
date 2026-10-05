"""Standalone Colab Step 2, embedding real B-stage results and current code."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import zlib
import zipfile
from pathlib import Path

from build_step0_notebook import cell, notebook, save

ROOT = Path(__file__).resolve().parent.parent


def main():
    files = [ROOT / 'eval.py', ROOT / 'code/README.md', ROOT / 'code/lab_day2.ipynb',
             ROOT / 'starter/lab_day2.ipynb', ROOT / 'tools/gpu_budget.py', ROOT / 'results.xlsx']
    for folder in ('code','tests','starter'):
        files.extend(sorted((ROOT / folder).glob('*.py')))
    for exp, name in [('B01','resnet50'),('B02','resnext50_32x4d'),('B03','convnext_tiny'),
                      ('B04','deit_small_patch16_224'),('B05','efficientnet_b0')]:
        files.extend(sorted(p for p in (ROOT / f'runs/{exp}/seed0').glob('*')
                            if p.suffix in ('.json','.csv','.npy')))
        files.extend([ROOT / f'curves/{exp}_{name}_seed0.png', ROOT / f'predictions/{exp}_seed0_val.csv'])
    files.extend(p for p in (ROOT / 'step1').glob('*') if p.is_file())
    files.append(ROOT / 'step2/plan.md')
    data = {p.relative_to(ROOT).as_posix(): base64.b64encode(p.read_bytes()).decode() for p in files}
    required = {'starter/lab_day2.ipynb','tools/gpu_budget.py','eval.py','tests/test_step2.py',
                'runs/B03/seed0/config.json','runs/B03/seed0/val_logits.npy'}
    if not required.issubset(data):
        raise ValueError(f'Incomplete bundle: {sorted(required-data.keys())}')
    packed = base64.b64encode(zlib.compress(json.dumps(data).encode())).decode()
    cells = [cell('markdown','intro','''# Bước 2 — Công thức huấn luyện
Backbone: **ConvNeXt-Tiny fb_in1k**. T00 dùng lại B03: macro-F1 val **96,30%**,
seed 0, fold 0, 10 epoch, batch 32, input 224. Không cần upload checkpoint B03.

Chọn **T4 GPU**, chạy các ô từ trên xuống. Có **7 lượt train mới**: 6 ablation
đổi riêng từng yếu tố và T07 kết hợp augmentation + loss chọn bằng val.
Ba trục A/B/C đều có 3 giá trị tính cả T00. Mỗi lượt bắt đầu từ khởi tạo quy định;
không train tiếp từ B03. Chỉ dùng train/val, không đánh giá test.

B03 đã đo 78,16 giây/epoch gồm train + val: 7 × 10 epoch ≈ **1,52 giờ** trên T4.
Dự trù **2–2,5 giờ** cho tải dữ liệu, các phép biến đổi ảnh và dự phòng; frozen có
thể nhanh hơn. Đây là ước lượng từ B03, không phải đo thực tế của từng công thức.
Chạy lại ô dùng kết quả xong hoặc resume last.pt đúng cấu hình trong cùng phiên.
Trước khi phiên bị xóa, tải ZIP kết quả và checkpoint ở cuối notebook.
'''), cell('code','setup','''import sys, subprocess, json, base64, zlib, hashlib
from pathlib import Path
result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'timm==1.0.30', 'fvcore==0.1.5.post20221221', 'matplotlib', 'pandas',
    'pillow', 'scikit-learn', 'openpyxl'], capture_output=True, text=True)
print(result.stdout, result.stderr)
result.check_returncode()
BASE_DIR = Path('/content')
PROJECT_DIR = BASE_DIR / 'deepweeds_step0'
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
'''), cell('code','bootstrap',f'''# Code và bằng chứng B01–B05 thật, đã đóng gói trong notebook.
sources = json.loads(zlib.decompress(base64.b64decode({packed!r})))
for relative, encoded in sources.items():
    target = PROJECT_DIR / relative
    content = base64.b64decode(encoded)
    if relative == 'results.xlsx' and target.exists():
        continue  # Giữ các sheet/kết quả hiện có.
    if relative.startswith(('runs/','curves/','predictions/','step1/')) and target.exists():
        if target.read_bytes() != content:
            raise FileExistsError(f'Kết quả cũ khác bundle: {{target}}')
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
del sources, content
assert hashlib.sha256((PROJECT_DIR / 'eval.py').read_bytes()).hexdigest() == {hashlib.sha256((ROOT / 'eval.py').read_bytes()).hexdigest()!r}
CODE_DIR = PROJECT_DIR / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
for name in ['dataset','losses','model','train','step0','step1','step2']:
    sys.modules.pop(name, None)
import dataset, losses, model, train, step0, step1, step2
reference = step2.baseline_record(PROJECT_DIR)
step1.assert_environment(reference)
print('GPU/torch/torchvision/timm khớp B03; giữ công thức nền.')
'''), cell('code','tests','''command = [sys.executable, '-X', 'utf8', '-c',
    "import torch,unittest; torch.set_num_threads(2); "
    "s=unittest.defaultTestLoader.discover('tests'); "
    "r=unittest.TextTestRunner(verbosity=1).run(s); raise SystemExit(not r.wasSuccessful())"]
result = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True)
print(result.stdout, result.stderr)
result.check_returncode()
'''), cell('code','data','''IMAGES_DIR, LABELS_DIR = step0.prepare_data(BASE_DIR)
frames = dataset.load_split(LABELS_DIR)
split_checks = dataset.check_split(*frames, IMAGES_DIR)
print('CSV fold 0 gốc; test chỉ kiểm tra metadata, không đọc ảnh hay dự đoán.')
from IPython.display import display, Image as DisplayImage
'''), cell('code','t00','''baseline = step2.run_recipe(PROJECT_DIR, IMAGES_DIR, LABELS_DIR, 'T00')
step2.export_colab_xlsx(PROJECT_DIR)
print(json.dumps(baseline, ensure_ascii=False, indent=2))
display(DisplayImage(filename=str(PROJECT_DIR / 'curves/T00_convnext_tiny_seed0.png')))
''')]
    # Builder does not import ML libraries; experiment definitions are explicit.
    trials = [('T01','A — Khởi tạo từ đầu'), ('T02','A — Đóng băng backbone, train head'),
              ('T03','B — Thêm ColorJitter'), ('T04','B — CutMix alpha=1'),
              ('T05','C — Label smoothing epsilon=0.1'), ('T06','C — Focal gamma=2')]
    for exp, title in trials:
        cells.extend([cell('markdown',exp.lower()+'-title',f'## {exp} — {title}\nChỉ đổi yếu tố này so với T00.\n'),
                      cell('code',exp.lower(),f'''row = step2.run_recipe(PROJECT_DIR, IMAGES_DIR, LABELS_DIR, {exp!r})
step2.export_colab_xlsx(PROJECT_DIR)
print(json.dumps(row, ensure_ascii=False, indent=2))
display(DisplayImage(filename=str(PROJECT_DIR / 'curves/{exp}_convnext_tiny_seed0.png')))
''')])
    cells.extend([cell('markdown','t07-title','''## T07 — Kiểm tra kết hợp hai yếu tố
Chọn biến thể mới tốt nhất giữa T03/T04 và giữa T05/T06 theo macro-F1 val;
hòa điểm chọn ID trước. Lưu kế hoạch trước khi train. Giữ init=finetune.
Thử kết hợp cả khi hai biến thể mới thua T00 để đo tương tác; không giả định cải thiện.
'''), cell('code','t07','''plan = step2.plan_combination(PROJECT_DIR)
print(json.dumps(plan, ensure_ascii=False, indent=2))
row = step2.run_recipe(PROJECT_DIR, IMAGES_DIR, LABELS_DIR, 'T07')
step2.export_colab_xlsx(PROJECT_DIR)
display(DisplayImage(filename=str(PROJECT_DIR / 'curves/T07_convnext_tiny_seed0.png')))
'''), cell('code','compare','''rows = step2.collect_results(PROJECT_DIR)
display(__import__('pandas').DataFrame(rows)[['exp_id','axis','variant','macro_f1_val',
    'delta_f1_pp','top1_val','ece_val','f1_chinee_apple','f1_snake_weed','status']])
selection = step2.write_report(PROJECT_DIR)
step2.export_colab_xlsx(PROJECT_DIR)
print(json.dumps(selection, ensure_ascii=False, indent=2))
print((PROJECT_DIR / 'step2/step2_report.md').read_text(encoding='utf-8'))
'''), cell('code','export','''archive = step2.export_artifacts(PROJECT_DIR, BASE_DIR / 'step2_artifacts.zip')
checkpoints = step2.export_checkpoints(PROJECT_DIR, BASE_DIR / 'step2_best_checkpoints.zip')
print('Cần tải cả 2 file:', archive, checkpoints, sep='\\n')
print('results.xlsx, report, curves, log/val logits/code đã nằm trong artifacts ZIP.')
print('Checkpoint ZIP giữ đúng đường dẫn T01..T07/seed0/best.pt, tránh trùng tên.')
'''), cell('markdown','save','''## Tải kết quả
Chạy riêng hai ô dưới, mỗi ô tải một ZIP. Hoặc tải từ thanh **Files** bên trái.
ZIP checkpoint khoảng 0,8 GB; ZIP kết quả chứa workbook với sheet Backbones và Training.
Tải thêm notebook có output qua **File → Download → Download .ipynb** để lưu bằng chứng.

Nếu muốn resume sau khi phiên bị xóa, cần giữ thêm `last.pt` cùng config/log và `best.pt`.
Khi đã train xong, các `best.pt` đủ cho các thử nghiệm suy luận tiếp theo.
'''), cell('code','download-artifacts','''from google.colab import files
files.download(str(BASE_DIR / 'step2_artifacts.zip'))
'''), cell('code','download-checkpoints','''files.download(str(BASE_DIR / 'step2_best_checkpoints.zip'))
'''), cell('markdown','next','''Bước 2 hoàn thành khi có đủ T00–T07, đường cong tương ứng, sheet Training,
kết luận từng trục và selection.json. T00 có thể vẫn là tốt nhất.
Một seed/cấu hình chỉ dùng screening; khoảng bootstrap trên val không phải std giữa seed.
Test tiếp tục để riêng đến Bước 4. Sau khi tải, gửi hai ZIP về thư mục dự án để đối chiếu.
''')])
    save(ROOT / 'notebooks/step2_colab.ipynb', notebook(cells, 'step2_colab.ipynb'))
    print(f'Built notebooks/step2_colab.ipynb ({len(data)} embedded files)')
    # A long base64 code cell burdens browser editors. Keep the same evidence in
    # a separate ZIP for the lightweight notebook, without reducing experiments.
    archive = ROOT / 'notebooks/step2_bundle.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for relative, encoded in data.items():
            bundle.writestr(relative, base64.b64decode(encoded))
    bundle_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    light = copy.deepcopy(cells)
    light[0] = cell('markdown','intro','''# Bước 2 — Bản notebook nhẹ
Upload **step2_colab_light.ipynb** vào Colab, chọn **T4 GPU** rồi chạy từ trên xuống.
Ở ô thứ hai (bootstrap), chọn **step2_bundle.zip** từ máy khi được yêu cầu.
ZIP chứa code và kết quả B01–B05 thật; không cần upload checkpoint B03.

Giữ nguyên 7 lượt train mới T01–T07, ba trục khởi tạo/augmentation/loss.
T00 dùng lại B03 ConvNeXt-Tiny fb_in1k: seed 0, 10 epoch, batch 32, input 224.
Dự trù 2–2,5 giờ trên T4; chỉ train/val, không đánh giá test.
Khi xong tải step2_artifacts.zip, step2_best_checkpoints.zip và notebook có output.
''')
    original = ''.join(next(c for c in cells if c['id']=='bootstrap')['source'])
    continuation = original[original.index('for relative, encoded in sources.items():'):]
    loader = f'''# Upload ZIP riêng; ô code này không chứa chuỗi base64 khổng lồ.
import zipfile
EXPECTED_BUNDLE_SHA256 = {bundle_hash!r}
BUNDLE_PATH = BASE_DIR / 'step2_bundle.zip'
if not BUNDLE_PATH.exists():
    from google.colab import files
    print('Chọn file step2_bundle.zip trên máy bạn (không chọn .ipynb).')
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('Chỉ chọn một file step2_bundle.zip.')
    filename, payload = next(iter(uploaded.items()))
    if hashlib.sha256(payload).hexdigest() != EXPECTED_BUNDLE_SHA256:
        raise ValueError('ZIP không khớp notebook. Dùng step2_bundle.zip đi kèm bản nhẹ.')
    BUNDLE_PATH.write_bytes(payload)
    del uploaded, payload
if hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest() != EXPECTED_BUNDLE_SHA256:
    raise ValueError('step2_bundle.zip không khớp notebook; thay bằng ZIP đi kèm.')
sources = {{}}
with zipfile.ZipFile(BUNDLE_PATH) as bundle:
    for info in bundle.infolist():
        target = (PROJECT_DIR / info.filename).resolve()
        if not target.is_relative_to(PROJECT_DIR.resolve()):
            raise ValueError('Đường dẫn ZIP vượt ngoài thư mục dự án.')
        if not info.is_dir():
            sources[info.filename] = base64.b64encode(bundle.read(info)).decode()
'''
    index = next(i for i,c in enumerate(light) if c['id']=='bootstrap')
    light[index] = cell('code','bootstrap',loader+continuation)
    save(ROOT / 'notebooks/step2_colab_light.ipynb', notebook(light, 'step2_colab_light.ipynb'))
    print(f'Built lightweight notebook + ZIP: {archive.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
