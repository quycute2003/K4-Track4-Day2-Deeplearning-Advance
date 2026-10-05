# Bước 3 — Suy luận và đo độ trễ trên T05

Dùng nguyên checkpoint T05 (ConvNeXt-Tiny fb_in1k, label smoothing 0,1, seed 0).
Không train lại; không đánh giá test, không yêu cầu logits test ở giai đoạn này.
Số liệu I00 và temperature dùng logits val đã lưu. Các quyết định đều trên val.

| ID | Thử nghiệm | K | GPU cho dự đoán val |
|---|---|---:|---|
| I00 | 1-view FP32 224, resize 256/crop 224 | 1 | Không, dùng cache T05 |
| I01 | Identity + hflip, trung bình xác suất | 2 | Chỉ view hflip mới |
| I02 | Bốn góc + giữa, crop 224 từ ảnh 256 | 5 | Có, năm view tuần tự |
| I03 | Cùng hai view I01, trung bình logits rồi softmax | 2 | Không, dùng cache I01 |
| I04 | Input 256, resize 293/crop 256 | 1 | Có |
| I09 | Input 288, resize 329/crop 288 | 1 | Có |
| I10 | Input 320, resize 366/crop 320 | 1 | Có |
| I07 | 1-view + temperature tối ưu NLL val | 1 | Không, dùng cache I00 |
| I08 | 1-view AMP FP16, parameters vẫn FP32 | 1 | Có |

Có năm nhóm ngoài mốc: hflip, five-crop, resolution, calibration, AMP, cộng so sánh
gộp probability/logit. Không đưa các độ phân giải thành những nhóm phương pháp giả.
Mean/std/interpolation giữ từ tag T05. Resize/crop giữ tỷ lệ 256/224 giống mốc.

## Đo độ trễ

Chọn T4. Giữ timm 1.0.30; bộ torch/torchvision CUDA của Lightning có thể khác Colab.
Nguồn train giữ nguyên trong source.json; runtime.json ghi riêng nền tảng, GPU,
torch/torchvision/timm, CUDA/cuDNN, memory và các cờ deterministic/TF32 thực tế.
Đối chiếu toàn bộ 3.501 logits val FP32 và thứ tự/tên/nhãn với cache T05, sai số
atol/rtol 1e-4 và dự đoán lớp phải trùng. Nếu không khớp, dừng trước các thí nghiệm.
Không trộn số đo giữa Colab và Lightning hoặc giữa hai môi trường khác nhau.
Mỗi phương pháp đo cả
batch 1 và batch 32: 10 warmup bỏ đi + 100 lần đo. Đồng bộ GPU trước và sau từng lần.
Lưu toàn bộ samples_ms; tính p50/p95/p99 bằng numpy.percentile và mean cho thông lượng.
Chi phí tương đối = p50 batch 1 / p50 I00; thông lượng = batch / mean thời gian batch.

Input đã được resize/crop/normalize và có trên GPU. Tính tensor flip/crop, forward,
softmax, aggregation hoặc chia T; không tính đọc file/PIL preprocessing, H2D/D2H,
tải model hay DataLoader. K view chạy tuần tự, không tự thay batch chính thành K×batch.
Five-crop benchmark nhận batch ảnh 256, các phương pháp khác nhận crop cuối cùng.
Ghi input_size và img_size riêng để mô tả đúng điều kiện đo.

Ngưỡng p95≤100 ms chỉ kiểm tra pipeline GPU này, chưa bao gồm camera/tiền xử lý.
Chỉ đánh dấu đạt khi có số đo thật. Không lấy latency Bước 1 hay GPU máy cá nhân
thay cho phân vị T4. AMP không được mặc định là nhanh hơn hoặc giữ nguyên F1.

## Chọn và hiệu chuẩn

I07 khớp một T duy nhất bằng NLL trên val, tìm trong exp([-6,6]), ghi có chạm biên không.
Chia logits cho T>0 giữ thứ tự lớp; kiểm tra dự đoán không đổi. ECE dùng eval.py gốc,
15 bin đều nhau. Khớp và đo trên cùng val có thể lạc quan; không khớp T trên test.
T này chỉ thuộc phương pháp I07 1-view, không tự áp sang các TTA khác.

Chỉ chọn khi đủ dự đoán và cả hai batch của chín cấu hình. Ngoại tuyến: F1 val cao nhất,
hòa điểm chính xác thì ECE thấp hơn rồi p95 thấp hơn. Cấu hình triển khai dùng cùng quy tắc
trong các phương pháp p95≤100 ms. Nếu không có, ghi chưa đạt ngưỡng; không tạo số đo thay thế.
Mỗi cấu hình có một seed/checkpoint nên chênh lệch nhỏ chưa chứng minh ổn định.

ConvNeXt có LayerNorm, không có BatchNorm2d: fusion BN không áp dụng. Helper fuse_conv_bn
dùng FX để xác định cạnh Conv→BN an toàn, kiểm tra sai số ≤1e-5 trên fixture CPU.
T05 không có EMA; không tạo mục EMA. Không cần thêm checkpoint để làm ensemble/soup,
vì ngân sách này đã có đủ nhóm phương pháp yêu cầu.

## Chạy trên Colab

Upload `notebooks/step3_colab_light.ipynb`, bật T4, chạy từ đầu. Khi bootstrap yêu cầu,
chọn `notebooks/step3_bundle.zip`. ZIP chứa code, bằng chứng B/T và **chỉ checkpoint T05**;
notebook không nhúng chuỗi base64 dài. Các kết quả hiện có phải trùng nội dung; không ghi
đè cấu hình/dự đoán khác cho cùng exp_id.

Suy luận khoảng 10 lượt val mới; không có backward/optimizer/epoch train. Dự trù
**15–30 phút gồm setup/tải dữ liệu**, ước lượng từ tốc độ val T05 và khoản dự phòng,
chưa phải thời gian đo thật của cả loạt. Dữ liệu có sẵn thì bỏ qua tải lại.
Mỗi ô dùng lại kết quả/latency đã hoàn thành; một lượt suy luận dở chạy lại riêng lượt đó.

Khi xong tải `step3_artifacts.zip` và notebook có output. Không cần tải checkpoint mới.
ZIP có sheet Inference/Latency, scatter accuracy_latency.png, reliability plot, temperature,
selection, logits/probs/dự đoán val và raw timings. Bước 3 chỉ hoàn thành sau số đo GPU.

## Chạy trên Lightning AI

Upload `notebooks/step3_lightning.ipynb` và ZIP `notebooks/step3_bundle.zip` **bản mới**
vào cùng thư mục của Studio, chọn một GPU T4, mở notebook bằng JupyterLab và chạy từ đầu.
Notebook không dùng google.colab, không giả định đường dẫn /content, không train lại.
Không cần chuyển sang LightningModule hay cài PyTorch Lightning Trainer.

Giữ bộ torch/torchvision CUDA sẵn có. Nếu kernel thiếu bộ này, terminal Studio có thể
cài `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126`
rồi restart kernel. Đây là một cặp phiên bản/CUDA được PyTorch công bố; cần driver
và GPU của Studio thực sự hoạt động. Timm cố định 1.0.30. Dùng Python >=3.10.
Link: https://pytorch.org/get-started/previous-versions/ .

Mặc định dùng thư mục làm việc của kernel. Nếu ZIP ở nơi khác, đặt biến môi trường
DEEPWEEDS_BUNDLE bằng đường dẫn tuyệt đối trước ô bootstrap. Mọi dữ liệu/kết quả
được lưu tại `deepweeds_step3_lightning/` bên dưới thư mục làm việc; không phụ thuộc
thư mục tạm Colab. Giữ thư mục này để chạy lại các ô đã xong trong cùng môi trường.

Cuối notebook có liên kết đến `deepweeds_step3_lightning/step3_artifacts.zip`.
Nếu link không tải được, chọn Download của file trong Files JupyterLab. ZIP bao gồm
runtime.json và môi trường trong từng timing JSON. Lưu/download thêm notebook có output.
Cùng T4 không bảo đảm cùng latency giữa hai nhà cung cấp; mọi tỷ lệ chi phí ở Bước 3
lấy I00 và các phương pháp đo ngay trong phiên Lightning này, không dùng số Colab.
