# Bước 0 — code và cách chạy

Upload duy nhất `notebooks/step0_colab.ipynb` lên Colab, bật T4 GPU rồi Run all.
Notebook có code và bản `eval.py` nguyên gốc nhúng sẵn, tự tải hoặc dùng lại dữ liệu từ lần đo ngân sách.
Nó thực hiện EDA, unit tests, kiểm tra initial CE/overfit một batch, sau đó chạy ResNet-50
`B01` 10 epoch, batch 32, seed 0. **Không đánh giá test.**

File `code/lab_day2.ipynb` là bản notebook để dùng trong repo đã có các file Python.
Sau khi notebook tự chứa chạy, tải `step0_artifacts.zip`; bundle gồm code, notebook dùng trong repo,
tests, báo cáo Bước 0, các ảnh EDA/augmentation/overfit, history và dự đoán val.
Checkpoint không nằm trong ZIP: tải `runs/B01/seed0/best.pt` và `last.pt` riêng nếu cần giữ/tiếp tục.
Không commit dataset hay checkpoint vào git.

## Những gì đã cài đặt

- `dataset.py`: giữ CSV fold 0, kiểm tra metadata và file, transform, Dataset, loader, sampler, worker seed.
- `model.py`: backbone timm, scratch/frozen/finetune, bảo vệ BN/dropout khi frozen, nhóm LR/weight decay, params/GMAC.
- `losses.py`: CE, label smoothing, focal, trọng số tính từ train, Mixup/CutMix với hệ số theo diện tích thật.
- `train.py`: một `run(Config(...))`, AdamW, warmup/cosine theo bước, AMP, EMA, val macro-F1 chọn checkpoint, log/curve/predictions.
- `step0.py`: tải/checksum dữ liệu, EDA và các kiểm tra trước thí nghiệm.
- `inference.py` và `benchmark.py`: suy luận eval, TTA, temperature scaling trên val và đo p50/p95/p99 có warmup/đồng bộ GPU.
- `step4.py`: chung kết F00/F01 với ba seed, khóa quyết định trên val và lưu cache để không chạy lại model trên test.

Hướng dẫn hiện tại cho Lightning AI và Bước 4 ở [REPRODUCE.md](../REPRODUCE.md).

Ba nhóm tham số theo ý nghĩa là backbone có decay, norm/bias không decay và head có LR riêng.
Code tách head thêm thành hai nhóm optimizer để bias/norm của head cũng không bị weight decay.
GMAC dùng fvcore, tính 1 FMA là 1 operation; `config.json` ghi ops chưa được đếm, không xem số này là latency.
EMA trung bình tham số và sao chép buffer BN từ model đang train; nếu dùng EMA về sau,
cần phân tích thêm sự phù hợp của thống kê BN.

## Chạy trên Windows

```powershell
py -m pip install timm==1.0.30 fvcore==0.1.5.post20221221 matplotlib pandas pillow scikit-learn
py -X utf8 -m unittest discover -s tests -v
py -X utf8 code/train.py --set exp_id=B01 backbone=resnet50 epochs=10 batch_size=32 num_workers=0 seed=0 images_dir=data/images labels_dir=data/labels
```

Giữ `save_test_predictions=false` cho Bước 0–3. Seed vòng cuối dự kiến 0, 1, 2; chỉ đổi seed,
không thay fold hay cách chia. Fold 1–4 chỉ để thử nhiều fold ở phần thưởng.
Config/version/tag trọng số/mean/std/interpolation được ghi theo lần chạy thật.
Khởi tạo và checkpoint tốt nhất chỉ dùng macro-F1 val; hòa thì lấy epoch sớm hơn.

## File đầu ra và tiếp tục phiên chạy

Mỗi lần chạy lưu `runs/<exp_id>/seed<k>/config.json`, `history.csv`, `best.pt`, `last.pt`,
`summary.json`, `val_logits.npy`, `val_labels.npy`, `val_filenames.json` và kiểm tra split.
Mỗi epoch cập nhật checkpoint/log/curve để phiên ngắt vẫn có bằng chứng.
Đường cong: `curves/<exp_id>_<backbone>_seed<k>.png`.
Dự đoán: `predictions/<exp_id>_seed<k>_val.csv`, ghi bằng `eval.save_predictions`.

Nếu phiên bị ngắt mà `last.pt` còn tồn tại, đặt `RESUME=True` trong notebook rồi chạy lại ô huấn luyện;
hoặc thêm `resume=true` vào cùng lệnh CLI. Các tham số khác phải giữ nguyên.
Resume khôi phục optimizer, scheduler, EMA/scaler và trạng thái RNG/loader; môi trường phần cứng/thư viện
khác vẫn có thể tạo khác biệt. Notebook lưu dữ liệu/output trên đĩa phiên chạy, không tự mount Drive.
Tải checkpoint hoặc tự lưu ra ổ bền trước khi phiên Colab kết thúc.

CE ban đầu được đo trên 27 ảnh train cân bằng (3 ảnh/lớp), so với ln(9), có dung sai 0.75 cho head ngẫu nhiên.
Đây là kiểm tra sơ bộ, không phải bảo đảm mọi model bắt buộc đúng 2.197.
Overfit dùng 9 ảnh train cố định, LR chẩn đoán backbone 1e-3/head 1e-2, tối đa 200 bước,
kiểm tra CE ở eval ≤ 0.05 và accuracy = 1; trọng số chẩn đoán bị bỏ đi trước B01.
Nếu kiểm tra thất bại, notebook dừng trước các thí nghiệm thật.

Phần thống kê test chỉ gồm số ảnh/nhãn/tên file để kiểm tra chia dữ liệu; ảnh mẫu và ảnh giải mã
cho EDA lấy từ train/val. Test DataLoader chỉ được tạo khi cờ chung kết bật.
Số đo trong `step0_report.md` được sinh từ lần chạy thật; phần nhận xét ảnh cần bạn bổ sung bằng mắt.

CSV fold 0 gốc hiện có một nhãn khác `labels.csv`: `20170714-110407-3.jpg` được ghi Label 0
trong fold 0, Label 1 trong `labels.csv`. Vì vậy tổng theo fold 0 là Chinee Apple 1.126,
Lantana 1.063, còn bảng tham khảo/labels.csv là 1.125 và 1.064.
Code giữ nguyên nhãn của các CSV fold 0 theo S1, ghi `label_discrepancies.csv` và chênh lệch trong EDA/report.
Các giao vẫn rỗng, hợp tên ảnh vẫn đúng 17.509; không sửa dữ liệu để ép khớp bảng tham khảo.

API dùng: [timm](https://huggingface.co/docs/timm/reference/models),
[PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html),
[fvcore](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html).

## Bước 1 trên Colab

Upload `notebooks/step1_colab.ipynb`, bật T4 rồi chạy từng ô từ trên xuống.
Notebook kèm kết quả B01 thật; nếu phiên mới chưa có checkpoint, upload riêng
`best.pt` của B01 khi ô B01 yêu cầu. Không cần train lại B01 hay chạy lại EDA/overfit.

- B01: ResNet50 (đã train); chỉ đo latency sơ bộ.
- B02: ResNeXt50-32x4d.
- B03: ConvNeXt-Tiny, **tag `fb_in1k` rõ ràng**. Tag mặc định timm 1.0.30 là
  `in12k_ft_in1k`; không dùng tag này để giữ bộ so sánh pretrain ImageNet-1k.
- B04: DeiT-Small, transformer.
- B05: EfficientNet-B0, mạng nhẹ.

Tất cả kế thừa T00 của B01: seed 0, fold 0, 10 epoch, batch 32, AdamW/CE/AMP,
cùng LR và lịch LR. Mean/std/nội suy theo pretrained_cfg; resize val 256 và crop 224.
Notebook kiểm tra GPU và phiên bản torch/torchvision/timm khớp B01 trước khi train.
Nếu khác môi trường, dừng để xử lý trước khi so sánh; không âm thầm trộn các số đo.

Mỗi ô gọi cùng `train.run(Config(...))`. Lượt hoàn thành được giữ lại; lượt bị ngắt
resume từ `last.pt` với cùng cấu hình và đường dẫn. Không tự giảm batch hoặc LR riêng
cho một model khi OOM: ghi nhận lỗi, rồi thống nhất công thức nếu cần thay đổi.

Sau mỗi model, cập nhật `step1/backbones.csv`, `backbones.json` và sheet `Backbones`
trong `results.xlsx`. Ô tổng hợp chỉ chọn backbone khi đủ năm kết quả/latency.
Chọn một backbone macro-F1 val cao nhất để ablation theo kế hoạch GPU; đồng thời báo
model nhanh nhất, khoảng cách F1, mốc hội tụ 95% đỉnh riêng và dấu hiệu overfit ở cuối.
Đây là screening một seed; chênh lệch nhỏ chưa chứng minh ưu thế thống kê.

Latency sơ bộ dùng best checkpoint, FP32/batch 1, 10 warmup và một forward có
synchronize CUDA; input đã ở GPU, không gồm decode/preprocess/truyền dữ liệu.
Bước 3 mới đo phân vị p50/p95/p99. Bảng ImageNet lấy từ
[timm results](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv),
chỉ ghép đúng tag và input 224, giữ crop/interpolation của nguồn để nhận diện khác biệt.

Tải `step1_artifacts.zip`, best.pt/last.pt từng model và notebook có output.
ZIP chứa Excel, báo cáo, cấu hình/log/ảnh biểu đồ và dự đoán val, không chứa checkpoint.
`results.xlsx` hiện có đủ năm backbone thật. B03 ConvNeXt-Tiny được chọn cho ablation.

## Bước 2 trên Colab

Khuyên dùng bản nhẹ: upload `notebooks/step2_colab_light.ipynb` vào Colab,
sau đó chọn `notebooks/step2_bundle.zip` khi ô bootstrap yêu cầu upload.
Notebook và ZIP phải là cặp cùng lần build; SHA-256 được kiểm tra tự động.
Bản này giữ đủ code/kết quả và các thí nghiệm, tách payload khỏi ô code dài để giảm lag editor.

Upload `notebooks/step2_colab.ipynb`, chọn T4 và chạy từng ô từ trên xuống.
Notebook kèm kết quả thật B01–B05; T00 dùng lại B03, không cần upload checkpoint.
Giữ 10 epoch, batch 32, seed 0. Có sáu ablation riêng theo ba trục khởi tạo,
augmentation và loss, rồi T07 kiểm tra kết hợp chọn bằng val.
Xem chi tiết và ngân sách trong `step2/plan.md`: 1,52 giờ theo tốc độ B03,
dự trù 2–2,5 giờ. Kết quả T01–T07 chưa chạy sẽ được để trống.

Sau mỗi lượt, cập nhật sheet `Training`, `step2/ablations.json/csv`, F1 từng lớp
và đường cong. Chọn công thức chỉ khi đủ các lượt; T00 có thể vẫn thắng.
Các run mới không tiếp tục từ B03. Một seed chưa đo được std giữa seed;
bootstrap val chỉ mô tả nhiễu theo mẫu, không thay thế việc chạy nhiều seed.
Cuối phiên tải `step2_artifacts.zip`, `step2_best_checkpoints.zip` và notebook có output.
ZIP checkpoint giữ đường dẫn T01–T07/seed0/best.pt, không bị trùng tên khi tải.

## Bước 3 trên Colab

Upload `notebooks/step3_colab_light.ipynb`, bật T4, chọn `notebooks/step3_bundle.zip`
khi bootstrap yêu cầu. ZIP kèm chỉ best.pt T05 và bằng chứng/logit B/T; không train lại.
Xem `step3/plan.md`: I00, hflip prob/logit, five-crop, 256/288/320, temperature và AMP.
Đo mỗi cấu hình batch 1/32, 10 warmup + 100 lần có synchronize trước/sau.
Pipeline timing gồm tensor views/model/softmax/T, không gồm PIL/decode/H2D/D2H.
Không lấy latency GPU khác để lấp số T4. T và lựa chọn cuối chỉ khớp/chốt trên val.

Có bản Lightning AI: `notebooks/step3_lightning.ipynb` + `notebooks/step3_bundle.zip`.
Upload cùng thư mục trong Studio, chọn một GPU T4 rồi mở bằng JupyterLab. Notebook
giữ torch/torchvision CUDA của Studio và timm 1.0.30; không dùng API/đường dẫn Colab.
Đối chiếu toàn bộ logits val với T05 trước khi dùng cache. Nguồn train giữ nguyên;
môi trường suy luận/đo riêng trong step3/runtime.json và từng timing JSON.
Không trộn latency giữa các nền tảng/phiên bản thư viện. Download ZIP kết quả trong
Files tại deepweeds_step3_lightning/step3_artifacts.zip. Xem setup trong step3/plan.md.

Temperature fitting dùng cache T05 val, giữ nguyên T chính xác khi chạy lại.
Đối chiếu metric từ CSV đọc bởi eval.py gốc (CSV dùng 8 chữ số có nghĩa).
Code `inference.py` hỗ trợ BN fusion an toàn theo cạnh FX; ConvNeXt không có BN
nên ghi không áp dụng. Không tạo EMA khi T05 không train EMA.
Scatter chỉ tạo khi có latency thật cho đủ phương pháp. Khi xong tải `step3_artifacts.zip`
và notebook có output; không có checkpoint mới. Bước 3 chưa hoàn thành nếu thiếu số đo.

API: [CUDA synchronization](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.synchronize.html),
[AMP](https://docs.pytorch.org/docs/2.11/amp.html),
[Conv/BN fusion](https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.fuse_conv_bn_eval.html).
