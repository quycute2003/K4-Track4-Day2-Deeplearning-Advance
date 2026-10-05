"""Build a tiny notebook + separate ZIP with evidence and just the T05 checkpoint."""
from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path
from build_step0_notebook import cell,notebook,save

ROOT=Path(__file__).resolve().parent.parent


def lightning_notebook(cells):
    """Reuse the experiment cells; replace only cloud-specific setup/I/O."""
    updated=json.loads(json.dumps(cells))
    by_id={c['id']:c for c in updated}
    replacements={
        'intro': '''# Bước 3 — Suy luận và độ trễ trên Lightning AI
Chọn Studio với **1 GPU T4**, mở JupyterLab. Upload notebook này và **step3_bundle.zip**
vào cùng thư mục rồi chạy các ô từ trên xuống. Không cần PyTorch Lightning Trainer.
Notebook chỉ khoảng 15 KB; ZIP chứa duy nhất checkpoint T05, khoảng 117 MB.

Không train lại, không đánh giá test. Giữ nguyên 9 cấu hình: 1-view, hflip prob/logit,
5-crop, 256/288/320, temperature, AMP. Mỗi cấu hình đo batch 1/32,
10 warmup + 100 lần đồng bộ GPU. Input sẵn ở GPU; gồm TTA/model/softmax/T.

Dùng bộ torch/torchvision CUDA đang hoạt động trong Studio và timm==1.0.30.
Môi trường suy luận ghi riêng vào runtime.json, không sửa cấu hình train T05.
Đối chiếu **toàn bộ 3.501 logit val** với cache T05 trước khi tái sử dụng.
Chỉ gộp số đo trong cùng môi trường Lightning, không lấy số Colab lấp vào.

Dữ liệu/kết quả được lưu trong deepweeds_step3_lightning dưới thư mục làm việc.
Chạy lại ô sẽ dùng lại kết quả đã hoàn thành. Cuối notebook tải step3_artifacts.zip
qua liên kết hoặc Download trong Files của JupyterLab. Lưu thêm notebook có output.
''',
        'setup': '''import sys, subprocess, json, base64, hashlib, shutil, os
from pathlib import Path
if sys.version_info < (3,10):
    raise RuntimeError('Notebook yêu cầu Python >=3.10.')
# Keep the CUDA-enabled torch/torchvision supplied by the Studio.
try:
    import torch, torchvision
except ImportError as exc:
    raise RuntimeError('Thiếu torch/torchvision. Trong terminal Studio cài: '
        'pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126 '
        'rồi restart kernel và chạy lại.') from exc
if not torch.cuda.is_available():
    raise RuntimeError('Bật machine T4 và chọn kernel có CUDA rồi chạy lại. '
        'Nếu đang dùng torch CPU, cài bộ CUDA trong hướng dẫn và restart kernel.')
print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__,
      '| torchvision:', torchvision.__version__, '| CUDA:', torch.version.cuda)
result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'timm==1.0.30', 'fvcore==0.1.5.post20221221', 'matplotlib', 'pandas',
    'pillow', 'scikit-learn', 'openpyxl'], capture_output=True, text=True)
print(result.stdout, result.stderr)
result.check_returncode()
if 'timm' in sys.modules and sys.modules['timm'].__version__ != '1.0.30':
    raise RuntimeError('Đã cài timm 1.0.30; restart kernel rồi chạy lại từ đầu.')
NOTEBOOK_DIR = Path.cwd().resolve()
BASE_DIR = NOTEBOOK_DIR / 'deepweeds_step3_lightning'
PROJECT_DIR = BASE_DIR / 'project'
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
print('Thư mục lưu dữ liệu/kết quả:', BASE_DIR)
''',
        'export': '''archive = step3.export_artifacts(PROJECT_DIR, BASE_DIR / 'step3_artifacts.zip')
print('Đã tạo:', archive)
print('ZIP có Excel, report/scatter/reliability, T, runtime, logits/probs/dự đoán val và raw timing.')
from IPython.display import FileLink
display(FileLink(str(archive.relative_to(NOTEBOOK_DIR))))
print('Nếu liên kết không tải được: trong Files mở deepweeds_step3_lightning, '
      'chuột phải step3_artifacts.zip → Download.')
''',
        'next': '''Lưu notebook có output bằng Save, rồi Download từ Files của JupyterLab.
Tải **deepweeds_step3_lightning/step3_artifacts.zip** và gửi về thư mục dự án.
Không có checkpoint mới, không cần tải lại T05.

Bước 3 hoàn thành khi đủ 9 cấu hình, 18 số đo batch, ECE trước/sau T và scatter,
có ít nhất một cấu hình p95 ≤100 ms. Ngưỡng này chỉ áp dụng pipeline GPU đã khai báo.
Mọi quyết định trên val; test giữ đến Bước 4. Không cần chạy lại Bước 1/2 vì đổi nền tảng.
'''}
    for identity,source in replacements.items():
        by_id[identity].update(cell(by_id[identity]['cell_type'],identity,source))
    bootstrap=''.join(by_id['bootstrap']['source'])
    begin=bootstrap.index('BUNDLE_PATH =')
    end=bootstrap.index("with BUNDLE_PATH.open('rb') as stream:")
    bootstrap=bootstrap[:begin]+'''BUNDLE_PATH = Path(os.environ.get('DEEPWEEDS_BUNDLE', str(NOTEBOOK_DIR / 'step3_bundle.zip'))).expanduser().resolve()
if not BUNDLE_PATH.is_file():
    raise FileNotFoundError(f'Upload step3_bundle.zip vào thư mục {NOTEBOOK_DIR}, hoặc đặt biến DEEPWEEDS_BUNDLE bằng đường dẫn ZIP.')
'''+bootstrap[end:]
    bootstrap=bootstrap.replace("platform='Colab'","platform='Lightning AI'")
    by_id['bootstrap'].update(cell('code','bootstrap',bootstrap))
    result=notebook(updated,'step3_lightning.ipynb')
    result['metadata'].pop('colab',None)
    return result


def main():
    files=[ROOT/name for name in ('eval.py','results.xlsx','code/README.md','code/lab_day2.ipynb',
        'starter/lab_day2.ipynb','tools/gpu_budget.py','step3/plan.md','step3/source.json','step3/temperature.json')]
    for folder in ('code','tests','starter'):
        files.extend(sorted((ROOT/folder).glob('*.py')))
    for folder in ('step1','step2','curves','predictions'):
        files.extend(sorted(p for p in (ROOT/folder).glob('*') if p.is_file() and p.suffix in ('.json','.csv','.md','.npy','.png')))
    for folder in sorted((ROOT/'runs').glob('[BT]*/seed0')):
        files.extend(sorted(p for p in folder.glob('*') if p.is_file() and p.suffix in ('.json','.csv','.npy')))
    # Do not embed incomplete I-stage outputs. They are generated from cached T05.
    files=[p for p in files if not (p.parent==ROOT/'predictions' and p.name.startswith('I'))]
    files.append(ROOT/'runs/T05/seed0/best.pt')
    paths={p.relative_to(ROOT).as_posix():p for p in files}
    required={'starter/lab_day2.ipynb','tools/gpu_budget.py','eval.py','tests/test_step3.py',
              'code/inference.py','code/benchmark.py','code/step3.py','runs/T05/seed0/best.pt'}
    if not required.issubset(paths):
        raise ValueError('Incomplete standalone bundle')
    archive=ROOT/'notebooks/step3_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for relative,path in paths.items():
            bundle.write(path,relative,compress_type=zipfile.ZIP_STORED if path.suffix=='.pt' else zipfile.ZIP_DEFLATED)
    with archive.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    eval_hash=hashlib.sha256((ROOT/'eval.py').read_bytes()).hexdigest()
    cells=[cell('markdown','intro','''# Bước 3 — Suy luận và độ trễ (bản nhẹ)
Chọn **T4 GPU** rồi chạy các ô từ trên xuống. Ô bootstrap yêu cầu upload **step3_bundle.zip**.
ZIP kèm code, logit/kết quả B/T và **chỉ checkpoint T05**, khoảng 112 MB; notebook không có ô base64 dài.
Không train lại. Chỉ chạy val, không đánh giá test. Không cần upload từng best.pt.

9 cấu hình: 1-view, hflip (prob/logit), 5-crop, input 256/288/320, temperature và AMP.
Có ít nhất 4 nhóm khác mốc. Mỗi cấu hình đo batch 1/32, warmup 10 + 100 lần đồng bộ GPU.
Input đo đã ở GPU, không gồm đọc ảnh/PIL/transfer; có gồm tensor TTA + forward + softmax/T.

Dự trù **15–30 phút gồm setup/tải dữ liệu**, chưa phải thời gian thực đo cả loạt trên T4.
Chạy lại ô sẽ dùng kết quả/latency đã hoàn thành; lượt suy luận dở chạy lại riêng lượt đó.
Cuối notebook tải **step3_artifacts.zip** và notebook có output. Không có checkpoint mới.
'''),cell('code','setup','''import sys, subprocess, json, base64, hashlib, shutil
from pathlib import Path
result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'timm==1.0.30', 'fvcore==0.1.5.post20221221', 'matplotlib', 'pandas',
    'pillow', 'scikit-learn', 'openpyxl'], capture_output=True, text=True)
print(result.stdout, result.stderr)
result.check_returncode()
BASE_DIR = Path('/content')
PROJECT_DIR = BASE_DIR / 'deepweeds_step0'
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
'''),cell('code','bootstrap',f'''import zipfile
def stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b''):
        digest.update(chunk)
    return digest.hexdigest()
EXPECTED_BUNDLE_SHA256 = {digest!r}
BUNDLE_PATH = BASE_DIR / 'step3_bundle.zip'
if not BUNDLE_PATH.exists():
    from google.colab import files
    print('Chọn step3_bundle.zip trên máy bạn (không chọn notebook).')
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('Chỉ chọn một file step3_bundle.zip.')
    filename, payload = next(iter(uploaded.items()))
    if hashlib.sha256(payload).hexdigest() != EXPECTED_BUNDLE_SHA256:
        raise ValueError('ZIP không khớp notebook. Dùng ZIP đi kèm bản Bước 3 này.')
    BUNDLE_PATH.write_bytes(payload)
    del uploaded, payload
with BUNDLE_PATH.open('rb') as stream:
    if stream_sha256(stream) != EXPECTED_BUNDLE_SHA256:
        raise ValueError('ZIP không khớp notebook; thay bằng ZIP đi kèm.')
with zipfile.ZipFile(BUNDLE_PATH) as bundle:
    for info in bundle.infolist():
        if info.is_dir():
            continue
        target = (PROJECT_DIR / info.filename).resolve()
        if not target.is_relative_to(PROJECT_DIR.resolve()):
            raise ValueError('Đường dẫn ZIP vượt ngoài dự án.')
        if info.filename == 'results.xlsx' and target.exists():
            continue
        if target.exists() and info.filename == 'step3/temperature.json':
            current = json.loads(target.read_text(encoding='utf-8'))
            supplied = json.loads(bundle.read(info))
            if any(current[key] != supplied[key] for key in ('T','fit_split','n','test_used')):
                raise FileExistsError(f'Temperature đã lưu khác ZIP: {{target}}')
            # Diagnostic floats can vary slightly across NumPy versions; keep T.
            continue
        if target.exists() and info.filename.endswith('.md') and info.filename.startswith(('step1/','step2/')):
            # Local reports may have been enriched after the original export.
            # They are analysis documents; measured source files below are locked.
            target.write_bytes(bundle.read(info))
            continue
        if target.exists() and info.filename.startswith(('runs/','curves/','predictions/','step1/','step2/','step3/source.json')):
            with target.open('rb') as current, bundle.open(info) as supplied:
                if stream_sha256(current) != stream_sha256(supplied):
                    raise FileExistsError(f'Bằng chứng cũ khác ZIP: {{target}}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(info) as supplied, target.open('wb') as destination:
            shutil.copyfileobj(supplied, destination)
assert hashlib.sha256((PROJECT_DIR / 'eval.py').read_bytes()).hexdigest() == {eval_hash!r}
CODE_DIR = PROJECT_DIR / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
for name in ['dataset','losses','model','train','step0','step1','step2','inference','benchmark','step3']:
    sys.modules.pop(name, None)
import dataset, model, train, step0, step1, step2, inference, benchmark, step3
reference = step3.reference(PROJECT_DIR)
runtime = step3.assert_environment(PROJECT_DIR, platform='Colab')
print('Môi trường suy luận:', runtime)
print('Nguồn train T05 giữ nguyên; cần đối chiếu logits val trước khi suy luận.')
'''),cell('code','tests','''command = [sys.executable, '-X', 'utf8', '-c',
    "import torch,unittest; torch.set_num_threads(2); "
    "s=unittest.defaultTestLoader.discover('tests'); "
    "r=unittest.TextTestRunner(verbosity=1).run(s); raise SystemExit(not r.wasSuccessful())"]
result = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True)
print(result.stdout, result.stderr)
result.check_returncode()
'''),cell('code','cached','''from IPython.display import display, Image as DisplayImage
rows = step3.prepare_offline(PROJECT_DIR)
step3.export_colab_xlsx(PROJECT_DIR)
print(json.dumps(step1.read_json(PROJECT_DIR / 'step3/temperature.json'), ensure_ascii=False, indent=2))
display(DisplayImage(filename=str(PROJECT_DIR / 'step3/calibration_reliability.png')))
print('Đã chuẩn bị I00/I07 từ cache; latency vẫn để trống đến khi đo T4.')
'''),cell('code','data','''IMAGES_DIR, LABELS_DIR = step0.prepare_data(BASE_DIR)
network = step3.load_selected_model(PROJECT_DIR, device='cuda')
error = step3.verify_model(PROJECT_DIR, IMAGES_DIR, LABELS_DIR, network)
print('Reload T05: sai số logit toàn bộ val:', error)
print(step1.read_json(PROJECT_DIR / 'step3/model_check.json'))
print('ConvNeXt không có BN: fusion không áp dụng. Chỉ tạo val loader.')
''')]
    for exp,title in [('I00','Mốc FP32 1-view'),('I01','TTA lật ngang, gộp xác suất'),
                      ('I02','5 crop, gộp xác suất'),('I03','Cùng hai view I01, gộp logit'),
                      ('I04','Input 256'),('I09','Input 288'),('I10','Input 320'),
                      ('I07','Temperature scaling 1-view'),('I08','AMP FP16 1-view')]:
        cells.extend([cell('markdown',exp.lower()+'-title',f'## {exp} — {title}\nĐo batch 1 và 32, 100 lần sau 10 warmup.\n'),
            cell('code',exp.lower(),f'''row = step3.run_method(PROJECT_DIR, IMAGES_DIR, LABELS_DIR, {exp!r}, network)
step3.export_colab_xlsx(PROJECT_DIR)
print(json.dumps(row, ensure_ascii=False, indent=2))
''')])
    cells.extend([cell('code','compare','''rows = step3.collect_results(PROJECT_DIR)
display(__import__('pandas').DataFrame(rows))
selection = step3.write_report(PROJECT_DIR)
step3.export_colab_xlsx(PROJECT_DIR)
print(json.dumps(selection, ensure_ascii=False, indent=2))
display(DisplayImage(filename=str(PROJECT_DIR / 'step3/accuracy_latency.png')))
print((PROJECT_DIR / 'step3/step3_report.md').read_text(encoding='utf-8'))
'''),cell('code','export','''archive = step3.export_artifacts(PROJECT_DIR, BASE_DIR / 'step3_artifacts.zip')
print('Tải:', archive)
print('ZIP có Excel, report/scatter/reliability, T, logits/probs/dự đoán val và raw timing.')
from google.colab import files
files.download(str(archive))
'''),cell('markdown','next','''Tải thêm notebook có output qua **File → Download → Download .ipynb**.
Không cần tải checkpoint mới. Gửi `step3_artifacts.zip` về thư mục dự án để đối chiếu.
Bước 3 hoàn thành khi đủ 9 dòng, 18 số đo batch, ECE trước/sau T và scatter,
có ít nhất một cấu hình p95 ≤100 ms. Chỉ là latency pipeline GPU đã khai báo.
Test giữ riêng đến Bước 4; không dùng test để sửa phương pháp hoặc tìm T.
''')])
    save(ROOT/'notebooks/step3_colab_light.ipynb',notebook(cells,'step3_colab_light.ipynb'))
    save(ROOT/'notebooks/step3_lightning.ipynb',lightning_notebook(cells))
    print(f'Built light Step 3: {len(paths)} files; ZIP {archive.stat().st_size:,} bytes; only T05 checkpoint.')


if __name__=='__main__':
    main()
