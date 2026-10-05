"""Build a small Lightning notebook that reuses the user's Step 3 workspace."""
import hashlib
import json
import zipfile
from pathlib import Path
from build_step0_notebook import cell,notebook,save

ROOT=Path(__file__).resolve().parent.parent


def main():
    archive=ROOT/'notebooks/step4_bundle.zip'
    names=['code/step4.py','tests/test_step4.py','step4/plan.md','step4/frozen_plan.json']
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(ROOT/name,name)
    patch_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    previous_hash=hashlib.sha256((ROOT/'runs/step3_import/step3_artifacts.zip').read_bytes()).hexdigest()
    existing=json.loads((ROOT/'notebooks/step3_lightning.ipynb').read_text(encoding='utf-8'))
    setup=''.join(next(c for c in existing['cells'] if c['id']=='setup')['source'])
    cells=[cell('markdown','intro','''# Bước 4 — Chung kết trên Lightning AI
Giữ **Studio T4 đã chạy Bước 3**, upload notebook này và **step4_bundle.zip** vào
cùng thư mục với notebook Bước 3. Không cần upload lại trọng số T05.
Nếu Studio mới, upload thêm **step3_artifacts.zip** đã tải từ Bước 3 vào cùng thư mục.

Cần **6 lượt train mới**, 10 epoch/lượt: F00=T00/I00 và F01=T05/I09, mỗi nhóm seed 0/1/2.
Dự trù **1.5–2 giờ**, ước lượng từ các lượt trước, chưa phải thời gian đo Lightning.
F01 khớp T riêng trên val 288 trước test; F01 calibrated là kết quả chính.
Không lấy nhiệt độ I07 ở 224 áp cho 288.

Chạy sáu ô train/val trước, rồi mới chạy test. Test đúng một forward pass mỗi model/seed.
Chạy lại chỉ dùng cache test; nếu test dở không có cache hoàn chỉnh, notebook dừng.
Không đổi công thức/frozen_plan sau khi mở test.
Cuối phiên tải **step4_artifacts.zip**, **step4_best_checkpoints.zip** và notebook có output.
'''),cell('code','setup',setup),cell('code','bootstrap',f'''import zipfile
PATCH_SHA256 = {patch_hash!r}
PREVIOUS_SHA256 = {previous_hash!r}
def stream_sha256(stream):
    digest=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1<<20),b''):
        digest.update(chunk)
    return digest.hexdigest()
def checked_unpack(path,expected,*,patch=False):
    if not path.is_file():
        raise FileNotFoundError(f'Upload {{path.name}} vào {{NOTEBOOK_DIR}} rồi chạy lại.')
    with path.open('rb') as stream:
        if stream_sha256(stream)!=expected:
            raise ValueError(f'ZIP không khớp notebook: {{path.name}}')
    with zipfile.ZipFile(path) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            target=(PROJECT_DIR/info.filename).resolve()
            if not target.is_relative_to(PROJECT_DIR.resolve()):
                raise ValueError('Invalid ZIP path')
            if target.exists() and not (patch and info.filename.startswith(('code/','tests/'))):
                if patch and target.read_bytes()!=bundle.read(info):
                    raise FileExistsError(f'Existing frozen protocol differs: {{target}}')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.open(info) as supplied,target.open('wb') as destination:
                shutil.copyfileobj(supplied,destination)
if not (PROJECT_DIR/'step3/selection.json').is_file():
    checked_unpack(NOTEBOOK_DIR/'step3_artifacts.zip',PREVIOUS_SHA256)
checked_unpack(NOTEBOOK_DIR/'step4_bundle.zip',PATCH_SHA256,patch=True)
CODE_DIR=PROJECT_DIR/'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0,str(CODE_DIR))
for name in ['dataset','model','losses','train','inference','benchmark','step0','step1','step2','step3','step4']:
    sys.modules.pop(name,None)
import step0,step4
plan=step4.freeze_plan(PROJECT_DIR)
print(json.dumps(plan,ensure_ascii=False,indent=2))
print('Runtime:',step4.runtime_check(PROJECT_DIR))
'''),cell('code','tests','''command=[sys.executable,'-X','utf8','-c',
    "import torch,unittest; torch.set_num_threads(2); s=unittest.defaultTestLoader.discover('tests'); "
    "r=unittest.TextTestRunner(verbosity=1).run(s); raise SystemExit(not r.wasSuccessful())"]
result=subprocess.run(command,cwd=PROJECT_DIR,capture_output=True,text=True)
print(result.stdout,result.stderr); result.check_returncode()
'''),cell('code','data','''IMAGES_DIR,LABELS_DIR=step0.prepare_data(BASE_DIR)
print('Chỉ kiểm tra metadata test trong giai đoạn train. Chưa tạo dự đoán test.')
''')]
    for seed in (0,1,2):
        for group in ('F00','F01'):
            cells.extend([cell('markdown',f'{group.lower()}-{seed}-title',f'## {group} — seed {seed}\nTrain 10 epoch và khóa val/T/latency.\n'),
                cell('code',f'{group.lower()}-{seed}',f'''summary=step4.run_training(PROJECT_DIR,IMAGES_DIR,LABELS_DIR,{group!r},{seed})
val=step4.prepare_val(PROJECT_DIR,IMAGES_DIR,LABELS_DIR,{group!r},{seed})
print(json.dumps(dict(training=summary,val=val),ensure_ascii=False,indent=2))
''')])
    cells.extend([cell('markdown','test-warning','''## Test cuối cùng
Chỉ chạy sau khi sáu ô train/val đều hoàn thành. Không sửa cấu hình sau ô này.
Một lượt model mỗi nhóm/seed; bản calibrated/uncal lấy từ cùng logits đã lưu.
'''),cell('code','test-once','''for seed in step4.SEEDS:
    for group in step4.GROUPS:
        result=step4.test_once(PROJECT_DIR,IMAGES_DIR,LABELS_DIR,group,seed)
        print(group,seed,json.dumps(result['metrics'],ensure_ascii=False))
'''),cell('code','eval','''summary=step4.summarize(PROJECT_DIR,LABELS_DIR)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print((PROJECT_DIR/'step4/eval_outputs/eval_output.md').read_text(encoding='utf-8'))
from IPython.display import display,Image,FileLink
for group in step4.GROUPS:
    display(Image(filename=str(PROJECT_DIR/f'step4/{group}_confusion.png')))
'''),cell('code','export','''artifacts=step4.export(PROJECT_DIR,BASE_DIR/'step4_artifacts.zip')
weights=step4.export(PROJECT_DIR,BASE_DIR/'step4_best_checkpoints.zip',checkpoints=True)
for archive in (artifacts,weights):
    print('Download:',archive)
    display(FileLink(str(archive.relative_to(NOTEBOOK_DIR))))
print('Có thể tải bằng chuột phải file → Download trong Files. Lưu thêm notebook có output.')
'''),cell('markdown','next','''Sau khi tải hai ZIP, gửi về dự án để nhập/đối chiếu. Bước 5 làm trên CPU:
hoàn thiện Final/PerClass/Summary, mean ± std, report.md, ma trận nhầm lẫn, ảnh lỗi,
README tái lập và push Git. Không cần train thêm để tổng hợp Bước 5.
''')])
    nb=notebook(cells,'step4_lightning.ipynb'); nb['metadata'].pop('colab',None)
    save(ROOT/'notebooks/step4_lightning.ipynb',nb)
    print('Built Step 4:',archive.stat().st_size,'bytes, no checkpoint or embedded payload.')


if __name__=='__main__':
    main()
