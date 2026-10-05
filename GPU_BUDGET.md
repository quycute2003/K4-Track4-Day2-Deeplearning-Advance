# Đo thời gian và lập ngân sách trên Colab/Kaggle

**Cách nhanh trên Colab:** upload duy nhất `notebooks/gpu_budget_colab.ipynb`.
Chọn Runtime → Change runtime type → GPU (T4 nếu được cấp), rồi Runtime → Run all.
Notebook này nhúng sẵn script, tự tải dữ liệu DeepWeeds và kiểm tra MD5; không cần upload repo.
Tải về `gpu_budget/profile.json` và `gpu_budget/plan.md` trong thanh Files sau khi chạy.
`GPU_HOURS=None` nghĩa là chưa chốt hạn mức; nhập số giờ thật rồi chạy lại ô tính ngân sách.

**Cách dùng repo đầy đủ:**
Mở `notebooks/gpu_budget.ipynb` trong Colab hoặc Kaggle, bật GPU và Internet.
Notebook yêu cầu repo có `tools/gpu_budget.py`; upload bản repo hiện tại hoặc các file mới trước khi chạy.
Tải và kiểm tra MD5 dữ liệu bằng các ô có sẵn trong `starter/lab_day2.ipynb`.
Đặt đúng đường dẫn thư mục ảnh và các CSV fold 0.

Script này đo **một epoch train và một lượt val đầy đủ** cho 5 backbone:
ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small và EfficientNet-B0.
Thời gian có đọc dữ liệu, augmentation, truyền dữ liệu, forward/backward và cập nhật trọng số;
GPU được đồng bộ ở ranh giới đo. Tải trọng số/khởi tạo được ghi riêng.
Chỉ kiểm tra metadata và sự tồn tại file test; không đọc ảnh hay chạy inference test.

Đây là phép đo sơ bộ trước thí nghiệm, không thay cho việc hoàn thiện và kiểm tra pipeline ở Bước 0.
Các model sau profiling bị bỏ đi. Không dùng kết quả một epoch làm kết quả backbone của bài nộp.
Nếu pipeline cuối có tốc độ khác profiler, cập nhật số đo bằng pipeline cuối.

## Cấu hình đo

Mọi backbone dùng ImageNet pretrained, head 9 lớp, CE, AdamW, warmup 1 epoch rồi cosine,
LR backbone `1e-4`, LR head `1e-3`, weight decay `0.05` (norm/bias bằng 0), AMP,
crop/flip khi train và resize/center crop khi val. Mean/std và interpolation theo cấu hình trọng số.
Mặc định batch **32**, ảnh 224, kế hoạch 10 epoch, seed 0; ghi lại cấu hình và version trong JSON.
Batch 32 là lựa chọn dự kiến cho cả 5 backbone, cần xác nhận trên GPU phiên chạy.
Nếu OOM, giảm cùng batch cho **tất cả** backbone và đo lại, không đổi riêng một model.
Nếu đổi độ phân giải hoặc GPU, cũng đo lại toàn bộ.

Trên Windows dùng launcher `py`:

```powershell
py tools/gpu_budget.py measure --images-dir data/images --labels-dir data/labels
py tools/gpu_budget.py plan --selected resnet50 --epochs 10 --gpu-hours 20
```

`20` giờ chỉ là ví dụ cú pháp. Nhập hạn mức thật trong tài khoản, cùng chu kỳ tính với profiling.
Trong notebook gọi qua `sys.executable` để dùng đúng Python của phiên Colab/Kaggle.
Trên Windows profiler dùng `num_workers=0`; số worker thực tế được lưu vào JSON.
Các lệnh kiểm tra toàn repo trên Windows nên bật UTF-8 để `eval.py` ghi được tiếng Việt:

```powershell
py -X utf8 -m unittest discover -s tests -v
```

## Số lần chạy dự kiến

| Giai đoạn | Ablation 1 backbone | Ablation 2 backbone |
|---|---:|---:|
| B: 5 backbone, mỗi model 1 seed | 5 | 5 |
| T: 6 ablations và 1 kết hợp / backbone | 7 | 14 |
| F: cấu hình cuối 3 seed + baseline 3 seed | 6 | 6 |
| Tổng huấn luyện mới | **18** | **25** |

Ba trục T dự kiến: khởi tạo (finetune / scratch / frozen), augmentation
(basic / color / CutMix), loss (CE / label smoothing / focal).
Giá trị nền có từ B-stage (`T00`); mỗi trục có hai lần chạy mới, mỗi lần chỉ đổi một yếu tố.
Thêm một lần kết hợp các yếu tố có bằng chứng tốt trên val.
Vòng cuối vẫn dự trù chạy baseline mới cho đủ 3 seed, không trừ đi lần chạy sàng lọc.
Phương pháp suy luận dự kiến: hflip TTA, multi-crop TTA, ensemble, temperature scaling;
có mốc I00 riêng. Đây là kế hoạch, chưa có kết quả chứng minh kỹ thuật nào tốt.

Đổi `--selected resnet50` thành backbone dự kiến hoặc hai tên đã đo để so ngân sách.
Nếu có hai backbone, dùng `--final-backbone` để chỉ tên dự kiến cho vòng cuối.
Chọn tên trong kế hoạch chỉ là giả định tính giờ, **không** phải chốt model tốt nhất.

Tổng giờ = giờ profiling + (giờ B + giờ T + giờ F + khoản thêm) × (1 + dự phòng).
Mặc định khoản thêm 1 giờ và dự phòng 25% là **giả định**, có thể điều chỉnh bằng
`--extra-hours` và `--reserve`; chưa được đo cho EDA, TTA, test, ghi checkpoint hay lần chạy hỏng.
Thời gian epoch đầu có khởi động DataLoader/kernel nên phép ngoại suy chỉ là ước lượng.

Khi chưa có hạn mức, bỏ `--gpu-hours`: báo cáo ghi rõ **chưa chốt kế hoạch**.
Nếu vượt ngân sách, giảm 15 → 10 epoch, chỉ ablation một backbone, giữ 3 seed cho final
và baseline. Sau đó mới hạ độ phân giải, đo lại và ghi thay đổi vào báo cáo.

Kết quả lưu ở `runs/gpu_budget/profile.json` và `runs/gpu_budget/plan.md`.
Tải các file này về sau khi chạy để giữ bằng chứng; `runs/` được gitignore.
Chỉ hoàn thành mục ngân sách khi đã đo đủ model **trên GPU cloud thực tế** và nhập hạn mức thật.

API tham khảo: [timm models](https://huggingface.co/docs/timm/reference/models),
[PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html).

## Cập nhật sau Bước 1

Đã có đủ B01–B05. Chọn **B03 ConvNeXt-Tiny fb_in1k** cho Bước 2 theo macro-F1 val
0,963028. Giả định resnet50 ở phần ví dụ profiler không còn là backbone ablation.
Giữ một backbone, seed 0, 10 epoch, batch 32. T00 dùng lại B03 nên có **7 lượt train mới**.
Trung bình B03 train + val 78,162643 s/epoch trên T4 → 70 epoch khoảng **1,52 giờ**;
dự trù **2–2,5 giờ**. Xem [kế hoạch Bước 2](step2/plan.md).
Số này dựa trên B03, chưa đo thực tế từng công thức; hạn mức tài khoản vẫn cần người dùng kiểm tra.
Giữ các file profiling gốc để đối chiếu, không thay số đo cũ bằng ước lượng mới.
