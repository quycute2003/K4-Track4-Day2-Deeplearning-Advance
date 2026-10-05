# Tái lập bài làm DeepWeeds

Đã hoàn thành Bước 0–5. F01 test macro-F1 96,86 ± 0,27%, top-1 97,58 ± 0,19%
qua seed 0/1/2; mean ± std mẫu (ddof=1). Mốc F00 macro-F1 95,83 ± 0,38%.
Kết quả, ma trận nhầm lẫn và ảnh lỗi ở [report.md](report.md); bảng tổng hợp ở results.xlsx.
Không cần GPU để tính lại điểm từ các CSV đã commit.

## Môi trường

- Screening B/T: Colab Tesla T4, torch 2.11.0+cu130, torchvision 0.26.0+cu130, timm 1.0.30.
- Inference và chung kết: Lightning Tesla T4, Python 3.12.11, torch 2.8.0+cu128,
  torchvision 0.23.0+cu128, timm 1.0.30, CUDA12.8, cuDNN91002.
- Fold0 gốc, train10501/val3501/test3507. Seed screening0; chung kết0/1/2.

## Notebook

Mở .ipynb bằng Colab hoặc JupyterLab Lightning, không copy notebook vào ô code.

| Bước | Notebook | File ngoài notebook |
|---|---|---|
| Ngân sách | notebooks/gpu_budget_colab.ipynb | dữ liệu tự tải |
| 0 | [Mở trên Colab](https://colab.research.google.com/github/quycute2003/K4-Track4-Day2-Deeplearning-Advance/blob/main/notebooks/step0_colab.ipynb) | dữ liệu tự tải |
| 1 | [Mở trên Colab](https://colab.research.google.com/github/quycute2003/K4-Track4-Day2-Deeplearning-Advance/blob/main/notebooks/step1_colab.ipynb) | checkpoint B01 khi yêu cầu |
| 2 | [Notebook](notebooks/step2_colab_light.ipynb) | step2_bundle.zip tương ứng notebook |
| 3 | [Notebook Lightning](notebooks/step3_lightning.ipynb) | step3_bundle.zip tương ứng, gồm T05 best.pt |
| 4 | [Notebook Lightning](notebooks/step4_lightning.ipynb) | step4_bundle.zip; Studio mới cần thêm step3_artifacts.zip |

Chung kết: giữ Studio T4 Bước3, upload notebook4+ZIP4 cùng thư mục, chạy từ đầu.
Sáu ô train/val phải hoàn thành trước ô test. Cuối phiên tải step4_artifacts.zip và
step4_best_checkpoints.zip, lưu notebook có output. Xem step4/plan.md.
Phần report/Excel cuối cùng đã tổng hợp trên CPU sau khi nhập hai ZIP.

Các notebook có output và ZIP/checkpoint dùng tại từng phiên được giữ ngoàiGit.
Notebook sạch trong repo là bản để upload; không phải bản có output làm bằng chứng.
Link Colab trên trỏ tới notebook sạch trong GitHub; cần quyền truy cập repository.
Studio Lightning và notebook có output không được chia sẻ công khai.
Không commit dataset/checkpoint lớn theo quy định README bài lab.

## Code và kiểm tra

Implementation: code/dataset.py, model.py, losses.py, train.py, inference.py,
benchmark.py, step0/1/2/3/4.py. Giữ nguyên evaluator eval.py và starter pseudo-code.
Notebook cài timm/fvcore và các thư viện dữ liệu; giữ torch/torchvision CUDA của Studio.
Windows dùng `py -X utf8`; notebook dùng sys.executable để chọn đúng interpreter.

```powershell
py -X utf8 -m unittest discover -s tests
```

Local workspace hiện dùng thư viện ML riêng trong runs/step0_dependencies. Thư mục
đó được ignore; máy mới cần tự cài bộ torch/torchvision/timm phù hợp trước khi test.
Builder tools/build_step4_notebook.py tạo ZIP nhỏ chỉ có code/protocol mới;
không có checkpoint vì F00/F01 train mới từ ImageNet. Builder các bước trước cần
metadata/outputs và checkpoint tương ứng đã lưu ngoàiGit; không thể tái tạo checkpoint
huấn luyện từ repository code đơn thuần.

## Tính lại kết quả trên CPU

Tải nguyên metadata fold 0 từ tác giả, không chia lại/sửa nhãn. Trên Windows:

```powershell
py -m pip install numpy pandas
$labelsDir = Join-Path (Get-Location) 'data/labels'
New-Item -ItemType Directory -Path $labelsDir -Force | Out-Null
foreach ($name in @('labels','train_subset0','val_subset0','test_subset0')) {
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/$name.csv" -OutFile (Join-Path $labelsDir "$name.csv")
}
py -X utf8 eval.py score --pred "predictions/F00_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F00 --out eval_out
py -X utf8 eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out
py -X utf8 eval.py score --pred "predictions/F01_uncal_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01_uncal --out eval_out
py -X utf8 eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/F00_seed*_test.csv" --uncal "predictions/F01_uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" --val-csv data/labels/val_subset0.csv --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --latency-p95-ms 8.79023894965485 --out eval_out
```

Trong notebook/Linux dùng sys.executable hoặc interpreter của môi trường đó.
`score`/`grade` chỉ đọc dự đoán đã lưu, không chạy model test lần hai. Output chính thức
đã lưu ở step4/eval_outputs; grade phần I là 20/20, không phải điểm toàn bài.

Để dựng lại báo cáo/Excel trong workspace có đầy đủ dữ liệu, chạy
`py -X utf8 tools/finalize_report.py <đường_dẫn_images.zip>` rồi chạy Node với
`tools/export_backbones.mjs . final`. Final dùng AVERAGE/STDEV.S từ ba seed; Summary
liên kết Final. Cần runtime `@oai/artifact-tool` cho bước author Excel trên desktop.
Importer tools/import_step4_artifacts.py cần hai ZIP ở runs/step4_import và CSV gốc
ở runs/step0_validation/labels; kiểm tra hash/logits/CSV/T/latency, không forward model.
Config/history/summary/split_checks sáu lượt có bản nhỏ trong step4/training_records.
SHA-256 checkpoint và ZIP ở step5/verification.json; weights giữ ngoài Git.

## Chính sách test

Mọi lựa chọn trên val. F00 baseline=T00/I00; F01=T05/I09+T khớp riêng val288.
Test một lần/model/seed, calibrated và uncal từ cùng logits, không forward lần hai.
Ledger test_started và cache lưu trong step4/Fxx/seedN. Không refit T hoặc sửa protocol
sau khi mở điểm test; lượt dở không có cache đầy đủ phải giữ chẩn đoán và dừng.
eval.py score/grade chạy sau khi đủ test; không sửa evaluator để thay cách tính.
