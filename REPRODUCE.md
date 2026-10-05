# Tái lập bài làm DeepWeeds

Tiến độ: Bước 0–3 hoàn thành; Bước 4 cần sáu lượt train/test; Bước 5 chờ chung kết.
Không có điểm test hoặc mean ± std cuối cùng. Báo cáo hiện tại: report.md.

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
| 0 | notebooks/step0_colab.ipynb | dữ liệu tự tải |
| 1 | notebooks/step1_colab.ipynb | checkpoint B01 khi yêu cầu |
| 2 | notebooks/step2_colab_light.ipynb | step2_bundle.zip tương ứng notebook |
| 3 | notebooks/step3_lightning.ipynb | step3_bundle.zip tương ứng, gồm T05 best.pt |
| 4 | notebooks/step4_lightning.ipynb | step4_bundle.zip; Studio mới cần thêm step3_artifacts.zip |

Chung kết: giữ Studio T4 Bước3, upload notebook4+ZIP4 cùng thư mục, chạy từ đầu.
Sáu ô train/val phải hoàn thành trước ô test. Cuối phiên tải step4_artifacts.zip và
step4_best_checkpoints.zip, lưu notebook có output. Xem step4/plan.md.
Phần report/Excel cuối cùng tổng hợp trên CPU sau khi nhập hai ZIP.

Các notebook có output và ZIP/checkpoint dùng tại từng phiên được giữ ngoàiGit.
Notebook sạch trong repo là bản để upload; không phải bản có output làm bằng chứng.
Không có link Studio/notebook công khai được cung cấp, nên chưa điền link chạy công khai.
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

## Chính sách test

Mọi lựa chọn trên val. F00 baseline=T00/I00; F01=T05/I09+T khớp riêng val288.
Test một lần/model/seed, calibrated và uncal từ cùng logits, không forward lần hai.
Ledger test_started và cache lưu trong step4/Fxx/seedN. Không refit T hoặc sửa protocol
sau khi mở điểm test; lượt dở không có cache đầy đủ phải giữ chẩn đoán và dừng.
eval.py score/grade chạy sau khi đủ test; không sửa evaluator để thay cách tính.
