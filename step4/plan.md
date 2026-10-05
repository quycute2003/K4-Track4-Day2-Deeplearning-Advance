# Bước 4 — Chung kết và đánh giá test

Bước 0–3 đã xong. Bước 4 vẫn cần GPU; Bước 5 tổng hợp trên CPU sau khi nhận đủ test.
Không dùng kết quả một seed để thay mean ± std của ba seed. Chưa có kết quả test.

## Cấu hình chốt bằng val

- F00: ConvNeXt-Tiny `fb_in1k`, công thức T00 (CE), suy luận I00 FP32 ảnh 224, T=1.
- F01: cùng backbone, công thức T05 (label smoothing 0.1), suy luận I09 FP32 ảnh 288.
  Khớp một T dương riêng cho mỗi seed bằng NLL trên chính logits val 288 trước khi mở test.
  Đây là bổ sung hiệu chuẩn được khai báo trước test; không dùng T 224 của I07 cho ảnh 288.
  F01 calibrated là kết quả chính. F01_uncal tạo từ cùng logits test, không chạy model thêm.

Cả hai nhóm: seed 0/1/2, train 224, 10 epoch, batch 32, basic crop/flip, ImageNet norm,
AdamW LR backbone/head 1e-4/1e-3, weight decay 0.05 trừ norm/bias, warmup 1 epoch,
cosine, AMP train, checkpoint có macro-F1 val 224 tốt nhất (hòa lấy epoch sớm hơn).
Train lại cả sáu từ ImageNet pretrained; không tái dùng B03/T05 đã sàng lọc ở Colab.
Như vậy mốc và cấu hình cuối có cùng seed, môi trường Lightning và ngân sách epoch.

I09 hiện có F1 val 96.9838%, top-1 97.7149%, p95 9.068 ms trên T4 Lightning.
Nhiệt độ và latency F01 từng seed được khớp/đo lại trên val trước test.
Chênh lệch Stage 3 là một seed, chưa chứng minh độ ổn định.

## Thứ tự bắt buộc

1. Giữ cùng Studio/môi trường T4 đã chạy Bước 3.
2. Upload step4_lightning.ipynb + step4_bundle.zip vào cùng thư mục, mở JupyterLab.
3. Chạy lần lượt sáu ô train/val. Mỗi ô lưu checkpoint, đường cong, logits val,
   T và đo batch-1: 10 warmup + 100 lần, synchronize trước/sau.
4. Chỉ khi cả sáu val đã khóa mới chạy ô test. Mỗi model/seed chạy test một lượt duy nhất.
5. Chạy eval.py score/grade gốc, xuất bảng Final/FinalStats/PerClass và confusion matrix.
6. Tải step4_artifacts.zip, step4_best_checkpoints.zip và notebook có output.
7. Nhập về dự án để hoàn thiện Bước 5, report.md, Summary, ảnh lỗi và README tái lập; không cần train thêm.

Notebook mặc định dùng project/data Bước 3 sẵn trong deepweeds_step3_lightning.
Nếu dùng Studio mới, upload thêm step3_artifacts.zip đã tải ở Bước 3 vào cùng thư mục.
Không cần upload checkpoint T05. Bước 4 tự tải lại trọng số ImageNet khi cần.

## Ngân sách và dùng lại kết quả

Sáu lượt × khoảng 13 phút/lượt ở Colab trước đây ≈ 78 phút cho train/val.
Dự trù **1.5–2 giờ** gồm setup, dữ liệu, val 288 và test; Lightning thực tế có thể khác.
Cùng GPU không bảo đảm cùng thời gian. Không tự giảm epoch hay số seed riêng một nhóm.

Train xong dùng lại summary. Train dở có last.pt thì resume đúng RNG/optimizer/scheduler.
T đã khớp được giữ nguyên. Test có ledger test_started.json tạo độc quyền trước lượt forward.
Chạy lại ô chỉ tính từ cache test hoàn chỉnh, không forward model lại. Nếu ngắt giữa test
và không có cache hoàn chỉnh, dừng và giữ chẩn đoán; không âm thầm chạy test lần hai.
Không sửa frozen_plan.json hay công thức sau khi mở điểm test.
