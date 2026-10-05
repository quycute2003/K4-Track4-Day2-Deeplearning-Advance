# Bước 2 — Kế hoạch trên ConvNeXt-Tiny

Chọn B03 theo Bước 1: macro-F1 val 0,963028, tốt nhất trong năm backbone.
Dùng một backbone và một seed cho ablation để tiết kiệm GPU. Dành ba seed cho Bước 4.
T00 là alias của B03, không huấn luyện thêm; đường cong T00 được vẽ từ log B03.

| ID | Trục | Thay đổi duy nhất so với T00 |
|---|---|---|
| T00 | Mốc | B03: finetune, crop/flip, CE |
| T01 | A. Khởi tạo | Scratch |
| T02 | A. Khởi tạo | Frozen backbone; chỉ train head |
| T03 | B. Augmentation | Thêm ColorJitter |
| T04 | B. Augmentation | CutMix, alpha 1; trộn cả nhãn |
| T05 | C. Loss | Label smoothing epsilon 0,1 |
| T06 | C. Loss | Focal gamma 2, không class weight |
| T07 | Kết hợp | Augmentation tốt nhất giữa T03/T04 + loss tốt nhất giữa T05/T06 |

Mỗi trục có ba giá trị tính cả T00. T01–T06 giữ nền cố định, không chọn tham lam.
T07 là ngoại lệ hai yếu tố để kiểm tra tương tác; chọn bằng macro-F1 val, hòa chọn ID trước.
Thử kết hợp cả khi các biến thể mới thua T00. Lưu `combination_plan.json` trước khi train;
không đổi kế hoạch khi resume. Cấu hình cuối chọn macro-F1 val cao nhất trong T00–T07,
hòa điểm ưu tiên T00 rồi ID trước. Không đọc điểm test để đưa ra quyết định.

Giữ ConvNeXt-Tiny `fb_in1k`, seed 0, fold 0, 10 epoch, batch 32, input 224;
AdamW LR backbone/head 1e-4/1e-3, WD 0,05 trừ norm/bias, warmup 1 epoch + cosine;
AMP train, FP32 val, checkpoint epoch macro-F1 val cao nhất. Dùng CSV fold 0 gốc,
val resize 256/center crop 224, ImageNet mean/std, nội suy bicubic.
Mỗi run bắt đầu mới, từ ImageNet hoặc scratch; không tiếp tục từ checkpoint B03.
T02 không cập nhật backbone, theo đúng khác biệt khởi tạo được thử.

## GPU và thời gian

Thực đo B03 trên Tesla T4: trung bình train + val **78,162643 giây/epoch**.
Ước lượng 7 run mới × 10 epoch = **1,519829 giờ** theo tốc độ B03.
Dự trù **2–2,5 giờ** gồm tải dữ liệu, augmentation và dự phòng. Frozen có thể nhanh hơn.
Chưa đo thời gian thực tế T01–T07, nên đây không phải cam kết thời lượng.
Chưa biết tổng hạn mức tài khoản Colab; kế hoạch chỉ ước lượng nhu cầu của Bước 2.
Giảm từ 12–15 xuống 10 epoch và batch 64 xuống 32 giống Bước 1; không tự giảm riêng một run.
Mọi thay đổi do hạn mức phải được ghi vào báo cáo và áp dụng công bằng.

## Kết quả và độ không chắc chắn

Sheet `Training`: F1/top-1/ECE/NLL chung, Δ F1 (điểm phần trăm) với T00,
F1 Chinee Apple/Snake Weed, cấu hình, nguồn và trạng thái; ô chưa chạy để trống.
`per_class_f1.csv` lưu cả chín lớp. Raw CE/LS/Focal loss không dùng để so trực tiếp.
Mỗi run một PNG với train/val loss, val F1/top-1 và LR. T00 dùng log B03.

Ablation chỉ một seed: chưa có std giữa seed. Paired stratified bootstrap 200 lần
trên cùng val, seed 0, dùng mô tả Δ do mẫu val; không thay thế std huấn luyện và
không xác nhận ưu thế sau khi đã tuning trên val. Chênh lệch nhỏ chưa chứng minh tốt hơn.
Mean ± std qua seed dành cho cấu hình chung kết ở Bước 4.

## Thực hiện

Nếu bản một file làm Colab lag, dùng `notebooks/step2_colab_light.ipynb`:
upload notebook vào Colab, chạy setup, rồi upload `notebooks/step2_bundle.zip`
khi ô bootstrap yêu cầu. Bản nhẹ giữ nguyên T00–T07 và đầy đủ bằng chứng B01–B05.
SHA-256 kiểm tra notebook và ZIP cùng lần build; không dùng nhầm ZIP cũ.

Upload `notebooks/step2_colab.ipynb`, chọn T4, chạy từ đầu. Notebook tự giải nén code,
kết quả thật B01–B05 và workbook; không cần upload checkpoint B03.
Cuối phiên tải `step2_artifacts.zip` và `step2_best_checkpoints.zip`, thêm notebook có output.
ZIP checkpoint giữ tên thư mục T01–T07/seed0/best.pt, tránh các file tải đều mang tên best.pt.
Nếu bị ngắt trước khi xong, giữ last.pt/best.pt của run đó cùng config/log để resume.

Để build lại notebook trên Windows: `py -X utf8 tools/build_step2_notebook.py`.
Kiểm tra bundle trong thư mục sạch: `py -X utf8 tools/validate_step1_bundle.py step2_colab.ipynb`.
Tập test chưa được đánh giá trong Bước 2.
