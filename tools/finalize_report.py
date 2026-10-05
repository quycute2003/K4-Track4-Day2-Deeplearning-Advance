"""Finish the lab from audited cached predictions; no training or model inference."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
import train


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def write(path, value):
    train.write_json(ROOT / path, value)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def fmt(mean, std):
    return f'{100 * mean:.2f} ± {100 * std:.2f}%'


def main():
    audit = read('step5/verification.json')
    assert audit['official_evaluator_recomputed'] and audit['additional_model_forwards'] == 0
    rows = read('step4/final.json')
    summary = read('step4/summary.json')
    uncal = read('step4/eval_outputs/F01_uncal_summary.json')
    classes = summary['F01']['classes']
    aggregate = {}
    columns = ['exp_id', 'seed', 'backbone', 'recipe', 'inference', 'temperature', 'macro_f1_val',
               'macro_f1_val_std', 'top1_val', 'top1_val_std', 'macro_f1_test', 'macro_f1_test_std',
               'top1_test', 'top1_test_std', 'ece_test', 'ece_test_std', 'ece_test_uncal', 'ece_test_uncal_std',
               'p95_ms', 'best_epoch', 'status']
    workbook = [{c: dict(r, backbone='convnext_tiny.fb_in1k').get(c) for c in columns} for r in rows]
    for group in ('F00', 'F01'):
        source = [r for r in rows if r['exp_id'] == group]
        a = dict(exp_id=group, seed='0/1/2', backbone='convnext_tiny.fb_in1k', recipe=source[0]['recipe'],
                 inference=source[0]['inference'], temperature=None, p95_ms=max(r['p95_ms'] for r in source),
                 best_epoch=None, status='Mean ± std; 3 seed')
        for key in ('macro_f1_val', 'top1_val', 'macro_f1_test', 'top1_test', 'ece_test', 'ece_test_uncal'):
            a[key], a[key + '_std'] = train.ev.mean_std([r[key] for r in source])
        aggregate[group] = a
        workbook.append({c: a.get(c) for c in columns})
        assert abs(a['macro_f1_test'] - summary[group]['macro_f1']['mean']) < 1e-12
    write('step5/final_workbook.json', workbook)
    write('step5/final_aggregate.json', aggregate)
    # Keep all measured inference records and append the six preregistered finals.
    latency = read('step3/latency.json')
    for r in latency:
        r.update(seed=0, temperature=read('step3/temperature.json')['T'] if r['exp_id'] == 'I07' else 1.)
    for r in rows:
        source_path = f"step4/{r['exp_id']}/seed{r['seed']}/latency_batch1.json"
        timing = read(source_path)
        latency.append(dict(exp_id=r['exp_id'], method=r['recipe'] + '/' + r['inference'], gpu=timing['gpu'],
            dtype=timing['dtype'], batch=1, img_size=timing['img_size'], input_size=timing['input_size'], k=1,
            fused_bn=False, p50_ms=timing['p50'], p95_ms=timing['p95'], p99_ms=timing['p99'],
            mean_ms=timing['mean'], images_per_s=timing['images_per_s'], iterations=100, warmup=10,
            preprocessing_included=False, transfer_included=False, timing_scope=timing['timing_scope'],
            torch=timing['torch'], source_path=source_path, seed=r['seed'], temperature=r['temperature']))
    write('step5/latency_workbook.json', latency)
    # Top 10 remains a val ranking; final rows use mean val across the three seeds.
    ranking = []
    for stage, source in [('B', read('step1/backbones.json')), ('T', read('step2/ablations.json')), ('I', read('step3/inference.json'))]:
        for r in source:
            if r['exp_id'] in ('T00', 'T05'):
                continue
            ranking.append(dict(exp_id=r['exp_id'], stage=stage,
                description=r.get('method', r.get('variant', r.get('backbone'))), macro_f1_val=r['macro_f1_val'],
                top1_val=r['top1_val'], ece_val=r.get('ece_val'), p95_ms=r.get('p95_ms'), seed_count=1,
                timing_scope='Lightning T4 GPU pipeline' if stage == 'I' else 'Screening; no comparable p95',
                notes='Một seed; T00=B03 và T05=I00 bỏ trùng'))
    for group, a in aggregate.items():
        source = [read(f"step4/{group}/seed{s}/val_done.json") for s in (0, 1, 2)]
        ranking.append(dict(exp_id=group, stage='F', description='ConvNeXt / ' + a['recipe'] + ' / ' + a['inference'],
            macro_f1_val=a['macro_f1_val'], top1_val=a['top1_val'], ece_val=float(np.mean([r['val_metrics']['ece'] for r in source])),
            p95_ms=a['p95_ms'], seed_count=3, timing_scope='Lightning T4; maximum seed p95',
            notes='Mean val 3 seed; công thức khóa trước test'))
    ranking.sort(key=lambda r: (-r['macro_f1_val'], r['ece_val'] if r['ece_val'] is not None else 1))
    write('step5/summary.json', [dict(rank=n + 1, **r) for n, r in enumerate(ranking[:10])])
    cm = pd.read_csv(ROOT / 'step4/eval_outputs/F01_confusion_sum.csv', index_col=0).to_numpy()
    errors = [(int(cm[i, j]), classes[i], classes[j]) for i in range(9) for j in range(9) if i != j and cm[i, j]]
    errors.sort(reverse=True)
    pred = pd.read_csv(ROOT / 'predictions/F01_seed0_test.csv')
    pred['confidence'] = pred[[f'p{i}' for i in range(9)]].max(axis=1)
    wrong = pred[pred.y_true != pred.y_pred].sort_values(['confidence', 'Filename'], ascending=[False, True])
    pair = wrong[((wrong.y_true == 0) & (wrong.y_pred == 7)) | ((wrong.y_true == 7) & (wrong.y_pred == 0))]
    selected = pd.concat([pair.head(6), wrong[~wrong.Filename.isin(pair.head(6).Filename)].head(6)]).head(12)
    image_archive = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('C:/Users/QUY/Downloads/images.zip')
    md5 = hashlib.md5()
    with image_archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            md5.update(chunk)
    assert md5.hexdigest() == 'b7b30f96d466fba86016aa5a26606e0f', 'Original images checksum differs'
    gallery = []
    with zipfile.ZipFile(image_archive) as archive:
        fig, axes = plt.subplots(3, 4, figsize=(13, 10))
        for ax, (_, r) in zip(axes.flat, selected.iterrows()):
            with Image.open(io.BytesIO(archive.read(r.Filename))) as picture:
                ax.imshow(picture.convert('RGB'))
            ax.axis('off')
            ax.set_title(f"True: {classes[int(r.y_true)]}\nPred: {classes[int(r.y_pred)]} ({r.confidence:.1%})\n{r.Filename}", fontsize=9)
            gallery.append(dict(Filename=r.Filename, y_true=int(r.y_true), y_pred=int(r.y_pred), confidence=float(r.confidence),
                true_class=classes[int(r.y_true)], predicted_class=classes[int(r.y_pred)], seed=0))
        fig.suptitle('F01 seed 0: Chinee apple / Snake weed errors and other confident mistakes', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(ROOT / 'step5/error_examples.png', dpi=150); plt.close(fig)
    write('step5/error_examples.json', gallery)
    # Plot the selected model's calibration from cached probabilities only.
    calibrated = train.ev.load_group(str(ROOT / 'predictions/F01_seed*_test.csv'), str(ROOT / 'runs/step0_validation/labels/test_subset0.csv'))
    raw = train.ev.load_group(str(ROOT / 'predictions/F01_uncal_seed*_test.csv'), str(ROOT / 'runs/step0_validation/labels/test_subset0.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, group, title in zip(axes, (raw, calibrated), ('Before temperature', 'After val-fit temperature')):
        confidence = np.concatenate([p.probs.max(axis=1) for p in group.preds])
        correct = np.concatenate([p.y_pred == p.y_true for p in group.preds])
        bins = np.minimum((confidence * 15).astype(int), 14)
        x, y, count = [], [], []
        for bin_id in range(15):
            chosen = bins == bin_id
            if chosen.any():
                x.append(confidence[chosen].mean()); y.append(correct[chosen].mean()); count.append(chosen.sum())
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
        ax.scatter(x, y, s=np.maximum(np.asarray(count) / 30, 12), color='#32759B')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean confidence', ylabel='Accuracy', title=title)
        ax.grid(alpha=.15)
    fig.suptitle('F01 test reliability: pooled visualization of 3 seeds, 15 bins')
    fig.tight_layout(); fig.savefig(ROOT / 'step5/final_calibration.png', dpi=160); plt.close(fig)
    # Required backbone scatter uses screening latency, explicitly distinct from p95.
    b = read('step1/backbones.json')
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in b:
        ax.scatter(r['params_m'], 100 * r['macro_f1_val'], s=55)
        ax.annotate(r['exp_id'], (r['params_m'], 100 * r['macro_f1_val']), xytext=(5, 4), textcoords='offset points')
    ax.set(xlabel='Parameters (M, 9-class head)', ylabel='Macro-F1 val (%)', title='Backbones: same T00, one seed')
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(ROOT / 'step5/backbone_tradeoff.png', dpi=160); plt.close(fig)
    a, base = aggregate['F01'], aggregate['F00']
    delta = a['macro_f1_test'] - base['macro_f1_test']
    noise = max(a['macro_f1_test_std'], base['macro_f1_test_std'])
    pc = read('step4/per_class.json')
    final_table = table(['Nhóm', 'F1 val', 'F1 test', 'Top-1 test', 'ECE test', 'p95 cao nhất'], [
        [g, fmt(x['macro_f1_val'], x['macro_f1_val_std']), fmt(x['macro_f1_test'], x['macro_f1_test_std']),
         fmt(x['top1_test'], x['top1_test_std']), f"{x['ece_test']:.5f} ± {x['ece_test_std']:.5f}", f"{x['p95_ms']:.3f} ms"] for g, x in aggregate.items()])
    seeds_table = table(['Nhóm', 'Seed', 'Best epoch', 'T trên val', 'F1 val', 'F1 test', 'Top-1 test', 'p95 ms'], [
        [r['exp_id'], r['seed'], r['best_epoch'], f"{r['temperature']:.6f}", f"{r['macro_f1_val']:.2%}",
         f"{r['macro_f1_test']:.2%}", f"{r['top1_test']:.2%}", f"{r['p95_ms']:.3f}"] for r in rows])
    class_table = table(['Lớp', 'Số ảnh/seed', 'Precision F01', 'Recall F01', 'F1 F01', 'F1 mốc'], [
        [r['class'], r['support'], fmt(r['precision_mean'], r['precision_std']), fmt(r['recall_mean'], r['recall_std']),
         fmt(r['f1_mean'], r['f1_std']), fmt(next(x for x in pc if x['exp_id'] == 'F00' and x['class'] == r['class'])['f1_mean'],
                                          next(x for x in pc if x['exp_id'] == 'F00' and x['class'] == r['class'])['f1_std'])]
        for r in pc if r['exp_id'] == 'F01'])
    train_minutes = sum(pd.read_csv(ROOT / f"runs/{r['exp_id']}/seed{r['seed']}/history.csv").epoch_seconds.sum() for r in rows) / 60
    tail = f'''## 7. Chung kết và phân tích lỗi

F00=T00/I00 là mốc; F01=T05/I09+T khớp trên val là cấu hình chính, đã chốt trước test.
Cả hai nhóm train lại từ ImageNet trên Lightning T4, seed 0/1/2, train 224, 10 epoch,
batch 32, AdamW LR backbone/head 1e-4/1e-3, decay 0.05 trừ norm/bias, warmup 1 epoch,
cosine và AMP train. F00 dùng CE, suy luận FP32 224, T=1. F01 dùng label smoothing 0.1,
suy luận FP32 288 một view. Checkpoint chọn bằng macro-F1 val 224 (hòa lấy epoch sớm).
Mỗi seed F01 khớp một T trên NLL val 288; khóa cả sáu val trước khi mở test.
Mỗi model/seed chỉ forward test một lượt; F01_uncal lấy từ cùng logits. Tổng train+val
ghi trong history sáu lượt là {train_minutes:.1f} phút, chưa gồm setup/val 288/test.

{final_table}

Mean ± std mẫu (ddof=1), ba seed, mỗi seed đủ 3.507 ảnh test. Không gộp 10.521 lượt
dự đoán thành 10.521 ảnh độc lập, không chọn seed tốt nhất.
Δ macro-F1 test F01−F00 = **{100 * delta:.4f} điểm phần trăm**, so với std lớn hơn
**{100 * noise:.4f} điểm phần trăm**; Δ vượt std và vượt 1 điểm phần trăm theo rubric I2.
Đây là so sánh cấu hình kết hợp T05+I09+calibration với T00+I00; không quy toàn bộ Δ
cho label smoothing. Calibration không đổi nhãn/F1; screening riêng của T05 và I09
mới tách được đóng góp của recipe và kích thước. Không phải kiểm định ý nghĩa thống kê.
Chênh tuyệt đối mean F1 val/test F01 là {100 * abs(a['macro_f1_val'] - a['macro_f1_test']):.4f} pp, dưới 2 pp.

{seeds_table}

ECE test F01 trước/sau TS: **{uncal['ece']['mean']:.5f} ± {uncal['ece']['std']:.5f} →
{a['ece_test']:.5f} ± {a['ece_test_std']:.5f}**; NLL giảm
{uncal['nll']['mean']:.5f} → {summary['F01']['nll']['mean']:.5f}. T chỉ khớp trên val;
không dùng test để tối ưu T. Top-1/F1 calibrated và uncal giống nhau ở cả ba seed.

![Hiệu chuẩn chung kết](step5/final_calibration.png)

{class_table}

Recall Chinee apple **{fmt(summary['F01']['recall']['mean'][0], summary['F01']['recall']['std'][0])}**,
Snake weed **{fmt(summary['F01']['recall']['mean'][7], summary['F01']['recall']['std'][7])}**,
cao hơn các mốc tham khảo 88,5%/88,8%. F1 thấp nhất còn ở Snake weed và Chinee apple.
Điều kiện bài báo khác nên không coi so sánh này là đánh giá cùng protocol.

![Ma trận mốc, tổng ba seed](step4/F00_confusion.png)
![Ma trận cuối, tổng ba seed](step4/F01_confusion.png)

Ma trận là tổng counts của ba seed, support hàng bằng ba lần số ảnh lớp.
F01 Chinee→Snake có **{int(cm[0, 7])}** lượt, Snake→Chinee **{int(cm[7, 0])}** lượt;
không diễn giải các counts này thành số ảnh khác nhau. Những hướng nhầm nhiều nhất:
{'; '.join(f'{true}→{predicted}: {count}' for count, true, predicted in errors[:5])}.
Nhiều lỗi là cỏ dại bị đưa về Negative; accuracy tổng bị chi phối bởi lớp Negative
chiếm khoảng 52%, nên vẫn dùng macro-F1 và bảng từng lớp.

![Ảnh lỗi F01 seed 0](step5/error_examples.png)

Ảnh lấy từ đúng tên file trong CSV test seed 0: ưu tiên cặp Chinee↔Snake, rồi các lỗi
khác có confidence cao. Không chọn seed theo điểm; seed 0 cố định chỉ để minh họa.
Tên/nhãn/confidence và quy tắc chọn lưu ở step5/error_examples.json.
Các giả thuyết từ ảnh cần đối chiếu trực quan: lá hẹp/lá nhỏ trong nền cỏ dày có thể
giống nhau; thiếu bộ phận phân biệt như hoa/quả và ánh sáng mạnh có thể làm đặc trưng
loài yếu đi; tâm crop có thể không chứa đầy đủ cây mục tiêu. Ảnh lỗi minh họa không
chứng minh quan hệ nhân quả. Không sửa nhãn, augmentation hay crop sau khi xem test.

Trong 20170803-132744-1.jpg, cây được gán Chinee apple chiếm vùng nhỏ giữa cành/cỏ khô;
20170405-160251-0.jpg có vùng sáng mạnh sát vùng tối; 20171009-085448-2.jpg có thân cây
lớn và ít lá trong khung. Các quan sát này gợi ý ảnh hưởng của nền và độ nhìn rõ mục tiêu,
chưa chứng minh nguyên nhân nhầm lớp.

eval.py grade tính lại từ CSV cho mục I: **{audit['grade_I']['total']}/20**, không có warning.
Đây chỉ là điểm tự kiểm phần chất lượng model I, không phải điểm toàn bài của giảng viên.

## 8. Kết luận và hạn chế

Cấu hình cuối đã chọn trên val là ConvNeXt-Tiny fb_in1k, finetune T05, một view 288
FP32 với T riêng khớp trên val. Test macro-F1 **{fmt(a['macro_f1_test'], a['macro_f1_test_std'])}**,
top-1 **{fmt(a['top1_test'], a['top1_test_std'])}**. So với mốc tăng {100 * delta:.2f} pp F1,
vượt std lớn hơn {100 * noise:.2f} pp trong ba seed này. Không chọn lại cấu hình từ điểm test.

Backbone/pretrain tạo thay đổi lớn nhất ở screening: ConvNeXt hơn ResNet khoảng
10,57 pp F1 val. Recipe T05 chỉ hơn T00 0,1075 pp trong một seed, bootstrap chứa 0;
I09 hơn I00 0,5735 pp trên cùng checkpoint. Ba seed cuối xác nhận cấu hình kết hợp
cải thiện mốc, chưa xác nhận độc lập hiệu quả từng kỹ thuật. Các tag pretrained có
lịch sử khác nhau; không quy toàn bộ khác biệt backbone cho riêng kiến trúc.

Robot có ngân sách 30–100 ms: dùng F01 một view 288 FP32 như đã chốt; p95 đo riêng
các seed {', '.join(f"{r['p95_ms']:.3f}" for r in rows if r['exp_id'] == 'F01')} ms,
giá trị cao nhất **{a['p95_ms']:.3f} ms**. Mọi seed đạt ≤100 ms ở batch 1.
Đây chỉ là pipeline với tensor đã normalize trên GPU, chưa gồm camera, decode/PIL,
transfer hoặc hàng đợi. TTA không cần cho cấu hình này; AMP batch lớn có throughput
tốt ở screening nhưng batch 1 p95 không tốt hơn, nên không đổi dtype cuối sau test.
Ba seed dùng ba T riêng, không có một T chung để tùy ý ghép với checkpoint khác.

Hạn chế: một fold, screening một seed, vòng cuối ba seed, 10 epoch/batch 32,
ablation một backbone và chưa có ensemble/EMA. Split ngẫu nhiên không theo địa điểm
có thể làm điểm test lạc quan khi sang cánh đồng/mùa/ánh sáng mới. Latency có biến động
theo chuỗi đo và môi trường; không đo end-to-end. Bài báo train khoảng 100 epoch và
augmentation khác; mốc 95,7% chỉ tham khảo. Bước tiếp theo có thể kiểm tra nhiều fold,
dữ liệu miền mới và latency camera thực, dùng dữ liệu mới để chọn cấu hình.

## 9. Tái lập và bằng chứng

Xem REPRODUCE.md: notebook, môi trường, thứ tự chạy và lệnh eval. exp_id/cấu hình
ở step1/backbones.json, step2/ablations.json, step3/inference.json, step4/frozen_plan.json.
Code huấn luyện/evaluator đã khóa trước test; evaluator không sửa.
Sáu config/history/summary/split_checks ở step4/training_records; từng seed có
val_done, test_started, test_done, logits/cache và raw latency trong step4/Fxx/seedN.
Curves đầy đủ B/T/F; 18 CSV chung kết val/test/calibrated/uncal trong predictions.
step5/verification.json đối chiếu SHA-256 hai ZIP, sáu checkpoint, metadata/CSV gốc,
logits→xác suất, T khớp trên val và output eval.py tính lại. Không chạy model test lần hai.
Dataset, ZIP và checkpoint lớn giữ ngoài Git. Notebook sạch được cung cấp trong repo;
notebook có output không có trong hai ZIP đã nhận, nên logs/cache là bằng chứng phiên.
'''
    current = (ROOT / 'report.md').read_text(encoding='utf-8')
    before = current.split('## 7.')[0]
    start = before.index('**Trạng thái:')
    end = before.index('## 2.')
    introduction = f'''**Trạng thái: đã hoàn thành Bước 0–5. Kết quả chung kết được tính lại bằng eval.py gốc.**

## 1. Tóm tắt

Phân loại chín lớp DeepWeeds, fold 0; năm backbone, ba trục công thức huấn luyện,
một kết hợp và chín cấu hình suy luận được sàng lọc trên val. Cấu hình chốt trước test:
ConvNeXt-Tiny fb_in1k, label smoothing 0.1, một view 288 FP32, T khớp riêng trên val.
Ba seed đạt macro-F1 test **{fmt(a['macro_f1_test'], a['macro_f1_test_std'])}** và top-1
**{fmt(a['top1_test'], a['top1_test_std'])}**. F1 tăng **{100 * delta:.2f} pp** so mốc
T00/I00, vượt std lớn hơn {100 * noise:.2f} pp. ECE test giảm {uncal['ece']['mean']:.5f}→{a['ece_test']:.5f};
p95 batch 1 cao nhất **{a['p95_ms']:.3f} ms** trên T4, chưa gồm camera/tiền xử lý/transfer.
Mọi lựa chọn khóa trên val; mỗi model/seed test một lượt. Giới hạn chính là một fold
và chia ngẫu nhiên, nên chưa bảo đảm hiệu quả trên miền hoặc hệ thống robot mới.

'''
    before = before[:start] + introduction + before[end:]
    before = before.replace('Test ở đây chỉ được kiểm tra metadata/sự tồn tại file, chưa chạy model hay xem ảnh test.',
                            'Trong Bước 0–3, test chỉ dùng metadata/sự tồn tại file. Bước 4 mới chạy model test theo kế hoạch đã khóa.')
    if '![F1 và số tham số backbone]' not in before:
        before = before.replace('![B03](curves/B03_convnext_tiny_seed0.png)',
                                '![B03](curves/B03_convnext_tiny_seed0.png)\n\n![F1 và số tham số backbone](step5/backbone_tradeoff.png)')
    (ROOT / 'report.md').write_text(before + tail, encoding='utf-8')
    write('step5/analysis.json', dict(delta_f1_test=delta, larger_std=noise, delta_exceeds_std=delta > noise,
        val_test_f1_gap=abs(a['macro_f1_val'] - a['macro_f1_test']), train_val_minutes_six_runs=float(train_minutes),
        worst_final_p95_ms=a['p95_ms'], off_diagonal_counts=errors, gallery_seed=0, std_ddof=1))
    for name in ('pending_final.json', 'pending_per_class.json'):
        (ROOT / 'step5' / name).unlink(missing_ok=True)
    write('step5/status.json', dict(completed_steps=[0, 1, 2, 3, 4, 5], pending_steps=[],
        final_seeds_completed=3, baseline_seeds_completed=3, test_prediction_files=9, val_prediction_files=9,
        remaining_gpu_training_runs=0, ready_for_submission=True, std_ddof=1, notebook_outputs_received=False))
    print('Final report, aggregate tables, 24 latency rows and cached-prediction error analysis completed.')


if __name__ == '__main__':
    main()
