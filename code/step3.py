"""Val-only inference screening with a frozen T05 checkpoint and measured timings."""
from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__:
    from . import dataset, model, train, step1, step2, inference, benchmark
else:
    import dataset, model, train, step1, step2, inference, benchmark

METHODS = {
    'I00': dict(method='1 view FP32', kind='single', size=224, input_size=224, k=1, space='prob', dtype='fp32'),
    'I01': dict(method='Hflip TTA / probability', kind='hflip', size=224, input_size=224, k=2, space='prob', dtype='fp32'),
    'I02': dict(method='5 crops / probability', kind='fivecrop', size=224, input_size=256, k=5, space='prob', dtype='fp32'),
    'I03': dict(method='Hflip TTA / logits', kind='hflip', size=224, input_size=224, k=2, space='logit', dtype='fp32'),
    'I04': dict(method='Resolution 256', kind='single', size=256, input_size=256, k=1, space='prob', dtype='fp32'),
    'I09': dict(method='Resolution 288', kind='single', size=288, input_size=288, k=1, space='prob', dtype='fp32'),
    'I10': dict(method='Resolution 320', kind='single', size=320, input_size=320, k=1, space='prob', dtype='fp32'),
    'I07': dict(method='1 view + temperature', kind='single', size=224, input_size=224, k=1, space='prob', dtype='fp32'),
    'I08': dict(method='1 view AMP FP16', kind='single', size=224, input_size=224, k=1, space='prob', dtype='amp'),
}
COLUMNS = ['exp_id','method','checkpoint','k','img_size','input_size','dtype','aggregation','temperature',
           'macro_f1_val','top1_val','ece_val','nll_val','p50_ms','p95_ms','p99_ms','relative_cost',
           'images_per_s_batch32','p95_le_100ms','status','prediction_path']
LATENCY_COLUMNS = ['exp_id','method','gpu','dtype','batch','img_size','input_size','k','fused_bn',
                   'p50_ms','p95_ms','p99_ms','mean_ms','images_per_s','warmup','iterations',
                   'torch','preprocessing_included','transfer_included','timing_scope','source_path']


def runtime_details():
    """Actual inference environment; training provenance stays in source.json."""
    import timm, torchvision
    if not torch.cuda.is_available():
        raise RuntimeError('Choose a T4 GPU machine before running Step 3.')
    gpu=torch.cuda.get_device_name(0)
    if gpu.strip().lower() not in ('tesla t4','nvidia t4','t4'):
        raise RuntimeError(f'This Step 3 protocol requires T4; actual GPU: {gpu}')
    if timm.__version__!='1.0.30':
        raise RuntimeError('Use timm==1.0.30 to preserve the T05 architecture.')
    # Keep the deterministic FP32 evaluation policy used when saving T05 logits.
    train.set_seed(0)
    return dict(benchmark.device_details('cuda'),python=sys.version,
                torchvision=str(torchvision.__version__),timm=str(timm.__version__),
                gpu_memory_bytes=int(torch.cuda.get_device_properties(0).total_memory))


def assert_environment(project,*,platform=None):
    root=Path(project); actual=runtime_details(); path=root/'step3/runtime.json'
    previous=step1.read_json(path) if path.exists() else None
    platform=platform or (previous['platform'] if previous else 'Unspecified')
    actual.update(platform=platform)
    if previous is not None and previous!=actual:
        raise RuntimeError('Inference environment changed; preserve this run and use a fresh Step 3 folder. Do not mix latency environments.')
    train.write_json(path,actual)
    return actual


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def reference(project):
    root=Path(project)
    selected=step1.read_json(root/'step2/selection.json')
    if selected['source_exp_id']!='T05' or selected['test_used'] or selected['seed']!=0:
        raise ValueError('This preregistered Step 3 protocol requires the selected T05, seed 0')
    record=step1.read_json(root/'runs/T05/seed0/config.json')
    step2.assert_run_record(step2.baseline_record(root),record,dict(loss='ls',label_smoothing=.1),'T05')
    return record


def _source(root):
    ref=reference(root)
    checkpoint=root/'runs/T05/seed0/best.pt'
    # The bundle carries the same source record when working from cached logits.
    if checkpoint.exists():
        weight_hash=sha256(checkpoint)
    else:
        weight_hash=step1.read_json(root/'step3/source.json')['checkpoint_sha256']
    code_hash=hashlib.sha256(b''.join((Path(__file__).parent/f'{name}.py').read_bytes()
                                    for name in ('inference','benchmark','step3'))).hexdigest()
    return dict(source_exp_id='T05',seed=0,checkpoint='runs/T05/seed0/best.pt',checkpoint_sha256=weight_hash,
                val_logits_sha256=sha256(root/'runs/T05/seed0/val_logits.npy'),
                val_labels_sha256=sha256(root/'runs/T05/seed0/val_labels.npy'),
                val_filenames_sha256=sha256(root/'runs/T05/seed0/val_filenames.json'),
                code_sha256=code_hash,versions=ref['versions'],gpu=ref['gpu'],protocol=2,test_used=False,
                selection_rule='macro-F1 val highest; exact ties: ECE then p95; realtime requires p95<=100ms',
                warmup=10,iterations=100,batches=[1,32],preprocessing_included=False)


def _save_result(root,exp_id,probs,*,logits=None,views=None,seconds=None):
    source=step1.read_json(root/'step3/source.json')
    labels=np.load(root/'step3/val_labels.npy')
    names=step1.read_json(root/'step3/val_filenames.json')
    if probs.shape!=(len(labels),9) or not np.isfinite(probs).all() or (probs<0).any() or not np.allclose(probs.sum(1),1):
        raise ValueError('Invalid aligned probabilities')
    folder=root/'step3'/exp_id
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'probs.npy',probs)
    if logits is not None:
        np.save(folder/'logits.npy',logits)
    if views is not None:
        np.save(folder/'view_logits.npy',views)
    pred_path=root/'predictions'/f'{exp_id}_seed0_val.csv'
    pred_path.parent.mkdir(parents=True,exist_ok=True)
    train.ev.save_predictions(str(pred_path),names,labels,probs)
    # Official CSV serialization uses 8 significant digits. Read it back so the
    # workbook agrees exactly with the unchanged evaluator, even for tiny probs.
    exported=train.ev.read_pred(str(pred_path))
    metrics=train.ev.compute_metrics(exported.y_true,exported.y_pred,exported.probs)
    temperature=step1.read_json(root/'step3/temperature.json')['T'] if exp_id=='I07' else 1.
    result=dict(exp_id=exp_id,source=source,method=METHODS[exp_id],temperature=temperature,
                metrics={key:metrics[key] for key in train.ev.SCALARS},n=len(labels),
                val_prediction_wall_seconds=seconds,test_used=False)
    train.write_json(folder/'result.json',result)
    return result


def prepare_offline(project):
    """Only cached val logits; no model, GPU timing or test evaluation."""
    root=Path(project)
    source=_source(root)
    path=root/'step3/source.json'
    if path.exists() and step1.read_json(path)!=source:
        raise ValueError('Cached inference protocol/checkpoint/code changed; preserve old evidence before rerunning')
    train.write_json(path,source)
    pred,base_metrics=step2.val_metrics(root,'T05')
    logits=np.load(root/'runs/T05/seed0/val_logits.npy')
    labels=pred.y_true
    names=list(pred.filenames)
    np.save(root/'step3/val_labels.npy',labels)
    train.write_json(root/'step3/val_filenames.json',names)
    if not (root/'step3/I00/result.json').exists():
        _save_result(root,'I00',inference.apply_temperature(logits,1),logits=logits)
    temperature_path=root/'step3/temperature.json'
    # Fit once, then preserve the exact T used by existing GPU measurements.
    # Different NumPy versions can move a scalar optimum by a few ULPs.
    if temperature_path.exists():
        previous=step1.read_json(temperature_path)
        if previous['fit_split']!='val' or previous['n']!=len(labels) or previous['test_used']:
            raise ValueError('Invalid cached temperature fit')
        t=previous['T']
    else:
        t=inference.fit_temperature(logits,labels,split='val')
    probs=inference.apply_temperature(logits,t)
    if not np.array_equal(probs.argmax(1),pred.y_pred):
        raise ValueError('Temperature changed class predictions unexpectedly')
    after=train.ev.compute_metrics(labels,probs.argmax(1),probs)
    before_full=train.ev.compute_metrics(labels,logits.argmax(1),inference.apply_temperature(logits,1))
    if after['nll']>before_full['nll']+1e-10:
        raise ValueError('Temperature optimization increased val NLL')
    train.write_json(root/'step3/temperature.json',dict(T=t,fit_split='val',n=len(labels),
        before={key:base_metrics[key] for key in train.ev.SCALARS},after={key:after[key] for key in train.ev.SCALARS},
        bounds=[math.exp(-6),math.exp(6)],at_search_boundary=abs(math.log(t))>5.99,
        accuracy_unchanged=True,scope='Fit and evaluate on the same val; ECE estimate may be optimistic',test_used=False))
    calibrated=(step1.read_json(root/'step3/I07/result.json') if (root/'step3/I07/result.json').exists()
                else _save_result(root,'I07',probs,logits=logits/t))
    diagnostics=step1.read_json(root/'step3/temperature.json')
    diagnostics['after']=calibrated['metrics']
    train.write_json(root/'step3/temperature.json',diagnostics)
    plot_calibration(root,pred.probs,probs,labels)
    return collect_results(root)


def make_loader(project,images_dir,labels_dir,exp_id,*,batch_size=32,num_workers=2):
    root=Path(project); ref=reference(root); spec=METHODS[exp_id]
    # Only the val CSV is loaded; no test DataLoader is created.
    frame=pd.read_csv(Path(labels_dir)/'val_subset0.csv')
    dataset._validate_frame(frame,'val')
    if list(frame.Filename)!=step1.read_json(root/'step3/val_filenames.json') or not np.array_equal(frame.Label.to_numpy(),np.load(root/'step3/val_labels.npy')):
        raise ValueError('Val CSV order/labels differ from saved logits')
    cfg=ref['pretrained_cfg']
    if spec['kind']=='fivecrop':
        from torchvision import transforms as T
        interp=getattr(T.InterpolationMode,cfg['interpolation'].upper())
        transform=T.Compose([T.Resize(256,interpolation=interp),T.CenterCrop(256),T.ToTensor(),T.Normalize(cfg['mean'],cfg['std'])])
    else:
        transform=dataset.build_transforms(False,spec['size'],mean=cfg['mean'],std=cfg['std'],interpolation=cfg['interpolation'])
    return dataset.make_loader(frame,images_dir,transform,batch_size,False,num_workers=num_workers,seed=0)


def load_selected_model(project,device='cuda'):
    root=Path(project); ref=reference(root)
    assert_environment(root)
    expected=step1.read_json(root/'step3/source.json')
    path=root/expected['checkpoint']
    if sha256(path)!=expected['checkpoint_sha256']:
        raise ValueError('T05 checkpoint SHA256 mismatch')
    train.set_seed(0)
    network=model.build_model(ref['config']['backbone'],pretrained=False).to(device).eval()
    state=torch.load(path,map_location=device,weights_only=True)
    summary=step1.read_json(root/'runs/T05/seed0/summary.json')
    if state['epoch']!=summary['best_epoch'] or not math.isclose(state['macro_f1'],summary['macro_f1_val'],abs_tol=1e-10):
        raise ValueError('Checkpoint metadata differs from T05')
    network.load_state_dict(state['model'],strict=True)
    network.requires_grad_(False)
    network._step3_source_sha=expected['checkpoint_sha256']
    return network


@torch.inference_mode()
def verify_model(project,images_dir,labels_dir,network):
    root=Path(project)
    runtime=assert_environment(root)
    loader=make_loader(root,images_dir,labels_dir,'I00')
    names,labels,actual=inference.predict_logits(network,loader,'cuda')
    expected=np.load(root/'runs/T05/seed0/val_logits.npy')
    if names!=step1.read_json(root/'step3/val_filenames.json') or not np.array_equal(labels,np.load(root/'step3/val_labels.npy')):
        raise ValueError('Reload verification changed val order/labels')
    error=float(np.max(np.abs(actual-expected)))
    if not np.allclose(actual,expected,atol=1e-4,rtol=1e-4) or not np.array_equal(actual.argmax(1),expected.argmax(1)):
        raise ValueError(f'Reloaded model/transform differs from cached T05 logits: max error {error}. Keep this diagnostic and align the environment before reusing cached logits.')
    bn=sum(isinstance(m,torch.nn.BatchNorm2d) for m in network.modules())
    train.write_json(root/'step3/model_check.json',dict(n=len(actual),scope='entire val; FP32; compared with T05 cache',max_abs_logit_diff=error,
        bn_count=bn,bn_fusion_applicable=bn>0,checkpoint_sha256=network._step3_source_sha,
        runtime=runtime,numerically_aligned=True,
        note='ConvNeXt uses LayerNorm; BN fusion not applicable' if not bn else 'BN present',test_used=False))
    return error


def run_method(project,images_dir,labels_dir,exp_id,network):
    root=Path(project); ref=reference(root)
    runtime=assert_environment(root)
    if exp_id not in METHODS:
        raise ValueError('Unknown inference experiment')
    source=step1.read_json(root/'step3/source.json')
    if network._step3_source_sha!=source['checkpoint_sha256']:
        raise ValueError('Wrong network for inference screening')
    verified=step1.read_json(root/'step3/model_check.json')
    if (verified.get('runtime')!=runtime or verified.get('checkpoint_sha256')!=source['checkpoint_sha256']
            or not verified.get('numerically_aligned') or verified.get('n')!=len(np.load(root/'step3/val_labels.npy'))):
        raise ValueError('Verify all val logits in the current inference environment before running methods')
    folder=root/'step3'/exp_id
    if not (folder/'result.json').exists():
        if exp_id=='I03':
            if not (root/'step3/I01/view_logits.npy').exists():
                raise FileNotFoundError('Run I01 before comparing logit aggregation')
            views=np.load(root/'step3/I01/view_logits.npy')
            _save_result(root,exp_id,inference.aggregate_views(views,'logit'),views=views)
        else:
            loader=make_loader(root,images_dir,labels_dir,exp_id)
            spec=METHODS[exp_id]
            view_fn=(lambda x:[inference.view_hflip(x)]) if spec['kind']=='hflip' else (lambda x:inference.views_multicrop(x,224)) if spec['kind']=='fivecrop' else None
            torch.cuda.synchronize(); start=time.perf_counter()
            names,y,views=inference.predict_views(network,loader,'cuda',view_fn,dtype=spec['dtype'])
            torch.cuda.synchronize(); seconds=time.perf_counter()-start
            if names!=step1.read_json(root/'step3/val_filenames.json') or not np.array_equal(y,np.load(root/'step3/val_labels.npy')):
                raise ValueError('Prediction order changed')
            if spec['kind']=='hflip':
                views=np.concatenate([np.load(root/'runs/T05/seed0/val_logits.npy')[None],views],axis=0)
            probs=inference.aggregate_views(views,spec['space'])
            _save_result(root,exp_id,probs,logits=views[0] if len(views)==1 else None,views=views,seconds=seconds)
    # Validate completed prediction evidence before reusing it or benchmarking.
    collect_results(root)
    measure_method(root,exp_id,network)
    return next(row for row in collect_results(root) if row['exp_id']==exp_id)


def measure_method(project,exp_id,network):
    root=Path(project); spec=METHODS[exp_id]
    result=step1.read_json(root/'step3'/exp_id/'result.json')
    for batch in (1,32):
        path=root/'step3'/exp_id/f'latency_batch{batch}.json'
        if path.exists():
            continue
        measured=benchmark.pipeline_report(network,batch_size=batch,img_size=spec['size'],input_size=spec['input_size'],
            kind=spec['kind'],space=spec['space'],temperature=result['temperature'],dtype=spec['dtype'],
            device='cuda',warmup=10,iters=100)
        measured.update(exp_id=exp_id,source=result['source'],method=spec,temperature=result['temperature'],
                        runtime=step1.read_json(root/'step3/runtime.json'))
        train.write_json(path,measured)
        print(f"{exp_id} batch {batch}: p50={measured['p50']:.3f}, p95={measured['p95']:.3f}, p99={measured['p99']:.3f} ms",flush=True)


def checked_latency(root,exp_id,batch,source,result):
    path=root/'step3'/exp_id/f'latency_batch{batch}.json'
    if not path.exists():
        return None
    value=step1.read_json(path); spec=METHODS[exp_id]
    runtime=step1.read_json(root/'step3/runtime.json')
    if (value['source']!=source or value['method']!=spec or value['exp_id']!=exp_id or value['batch']!=batch
            or value.get('runtime')!=runtime or value['gpu']!=runtime['gpu'] or value['torch']!=runtime['torch']
            or value['dtype']!=spec['dtype'] or value['temperature']!=result['temperature']
            or value['img_size']!=spec['size'] or value['input_size']!=spec['input_size'] or value['views']!=spec['k'] or value['fused_bn']
            or value['warmup']<10 or value['n']<100 or value['preprocessing_included'] or value['transfer_included']):
        raise ValueError('Latency protocol or environment mismatch')
    samples=np.asarray(value['samples_ms'])
    if len(samples)!=value['n'] or not np.isfinite(samples).all() or (samples<=0).any():
        raise ValueError('Invalid raw timing samples')
    for key,expected in zip(('p50','p95','p99'),np.percentile(samples,[50,95,99])):
        if not math.isclose(value[key],expected,abs_tol=1e-9):
            raise ValueError('Reported latency differs from raw samples')
    if not math.isclose(value['mean'],samples.mean(),abs_tol=1e-9) or not math.isclose(value['images_per_s'],batch*1000/samples.mean(),rel_tol=1e-10):
        raise ValueError('Mean or throughput differs from measured times')
    return value


def collect_results(project):
    root=Path(project); source=step1.read_json(root/'step3/source.json')
    labels=np.load(root/'step3/val_labels.npy'); names=step1.read_json(root/'step3/val_filenames.json')
    rows,latency=[],[]
    for exp_id,spec in METHODS.items():
        row=dict.fromkeys(COLUMNS)
        row.update(exp_id=exp_id,method=spec['method'],checkpoint=source['checkpoint'],k=spec['k'],
                   img_size=spec['size'],input_size=spec['input_size'],dtype=spec['dtype'],aggregation=spec['space'],
                   temperature=1.,status='Chưa chạy',prediction_path=f'predictions/{exp_id}_seed0_val.csv')
        folder=root/'step3'/exp_id
        if not (folder/'result.json').exists():
            rows.append(row); continue
        result=step1.read_json(folder/'result.json')
        if result['source']!=source or result['method']!=spec or result['test_used'] or result['exp_id']!=exp_id:
            raise ValueError('Saved inference method/source differs from protocol')
        pred=train.ev.read_pred(str(root/row['prediction_path']))
        probs=np.load(folder/'probs.npy')
        if exp_id in ('I00','I07'):
            expected_t=step1.read_json(root/'step3/temperature.json')['T'] if exp_id=='I07' else 1.
            expected_probs=inference.apply_temperature(np.load(root/'runs/T05/seed0/val_logits.npy'),expected_t)
            if result['temperature']!=expected_t or not np.allclose(probs,expected_probs,atol=1e-10):
                raise ValueError('Cached temperature/probabilities differ from source logits')
        if list(pred.filenames)!=names or not np.array_equal(pred.y_true,labels) or not np.allclose(pred.probs,probs,atol=1e-8):
            raise ValueError('Inference prediction alignment mismatch')
        metrics=train.ev.compute_metrics(labels,pred.y_pred,pred.probs)
        if any(not math.isclose(metrics[key],result['metrics'][key],abs_tol=1e-9) for key in train.ev.SCALARS):
            raise ValueError('Saved inference metrics differ from predictions')
        row.update(macro_f1_val=metrics['macro_f1'],top1_val=metrics['top1'],ece_val=metrics['ece'],
                   nll_val=metrics['nll'],temperature=result['temperature'],status='Chờ latency T4')
        for batch in (1,32):
            measured=checked_latency(root,exp_id,batch,source,result)
            if measured is None:
                continue
            latency.append(dict(exp_id=exp_id,method=spec['method'],gpu=measured['gpu'],dtype=spec['dtype'],
                batch=batch,img_size=spec['size'],input_size=spec['input_size'],k=spec['k'],fused_bn=False,
                p50_ms=measured['p50'],p95_ms=measured['p95'],p99_ms=measured['p99'],mean_ms=measured['mean'],
                images_per_s=measured['images_per_s'],warmup=measured['warmup'],iterations=measured['n'],
                torch=measured['torch'],preprocessing_included=False,transfer_included=False,
                timing_scope=measured['timing_scope'],source_path=f'step3/{exp_id}/latency_batch{batch}.json'))
            if batch==1:
                row.update(p50_ms=measured['p50'],p95_ms=measured['p95'],p99_ms=measured['p99'],
                           p95_le_100ms='Đạt' if measured['p95']<=100 else 'Không đạt')
            else:
                row['images_per_s_batch32']=measured['images_per_s']
        if row['p95_ms'] is not None and row['images_per_s_batch32'] is not None:
            row['status']='Đủ kết quả'
        rows.append(row)
    base=rows[0]['p50_ms']
    for row in rows:
        if base is not None and row['p50_ms'] is not None:
            row['relative_cost']=row['p50_ms']/base
    train.write_json(root/'step3/inference.json',rows)
    train.write_json(root/'step3/latency.json',latency)
    train.write_json(root/'step3/inference_schema.json',COLUMNS)
    train.write_json(root/'step3/latency_schema.json',LATENCY_COLUMNS)
    pd.DataFrame(rows,columns=COLUMNS).to_csv(root/'step3/inference.csv',index=False)
    pd.DataFrame(latency,columns=LATENCY_COLUMNS).to_csv(root/'step3/latency.csv',index=False)
    return rows


def choose(rows):
    if any(row['status']!='Đủ kết quả' for row in rows):
        raise ValueError('Finish all predictions and batch-1/batch-32 timings before selection')
    def key(row):
        return (-row['macro_f1_val'],row['ece_val'],row['p95_ms'])
    offline=min(rows,key=key)
    eligible=[row for row in rows if row['p95_ms']<=100]
    return offline,min(eligible,key=key) if eligible else None


def write_report(project):
    root=Path(project); rows=collect_results(root); offline,realtime=choose(rows)
    runtime=step1.read_json(root/'step3/runtime.json')
    selection=dict(source_exp_id='T05',offline_exp_id=offline['exp_id'],
        realtime_exp_id=realtime['exp_id'] if realtime else None,
        deployment_exp_id=realtime['exp_id'] if realtime else None,
        rule='highest val macro-F1; exact ties ECE then p95; realtime p95<=100ms',
        offline_method=METHODS[offline['exp_id']],offline_temperature=offline['temperature'],
        realtime_method=METHODS[realtime['exp_id']] if realtime else None,
        realtime_temperature=realtime['temperature'] if realtime else None,
        single_seed=True,test_used=False,runtime=runtime,scope='GPU-input pipeline; decode/PIL/H2D excluded')
    train.write_json(root/'step3/selection.json',selection)
    plot_tradeoff(root,rows)
    t=step1.read_json(root/'step3/temperature.json')
    lines=['# Bước 3 — Suy luận và độ trễ', '',
        'Checkpoint T05 ConvNeXt-Tiny fb_in1k, seed 0. Không train lại, không đánh giá test. '
        'Val có 3.501 ảnh theo CSV fold 0 gốc. I00 dùng lại logit T05; I07 và I03 dùng cache, '
        'I01/I02/độ phân giải/AMP chạy lại model trên val.', '',
        '## Phương pháp và số đo', '',
        '| ID | Phương pháp | K | F1 val | Top-1 | ECE | p50/p95/p99 batch 1 (ms) | × I00 (p50) | ảnh/s batch 32 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['exp_id']} | {r['method']} | {r['k']} | {r['macro_f1_val']:.2%} | {r['top1_val']:.2%} | {r['ece_val']:.5f} | "
                     f"{r['p50_ms']:.3f}/{r['p95_ms']:.3f}/{r['p99_ms']:.3f} | {r['relative_cost']:.2f} | {r['images_per_s_batch32']:.1f} |")
    lines+=['', '## Điều kiện đo và giới hạn triển khai', '',
        f"Nền tảng {runtime['platform']}, GPU {runtime['gpu']}; torch {runtime['torch']}, "
        f"torchvision {runtime['torchvision']}, timm {runtime['timm']}, CUDA {runtime['cuda']}, cuDNN {runtime['cudnn']}. "
        'Đã đối chiếu FP32 logits/dự đoán toàn bộ val với T05 trước khi dùng cache. Mỗi phương pháp đo batch 1 và 32, '
        '10 warmup bỏ đi và 100 lần đo; synchronize ngay trước/sau mỗi lần, perf_counter. '
        'Giữ raw samples_ms để tính lại phân vị. Thông lượng = batch / mean thời gian batch.', '',
        'Input đã resize/crop/normalize và có trên GPU. Đo gồm tensor flip/crop, forward(s), '
        'softmax và gộp view/chia T; không gồm đọc ảnh, PIL preprocessing, H2D/D2H hay tải model. '
        'Do đó p95<=100 ms là tiêu chí pipeline GPU, chưa chứng minh toàn hệ thống camera/robot <=100 ms. '
        'Five-crop chạy lần lượt năm view, không gộp thành một batch 5 lần lớn hơn. '
        'Các độ phân giải dùng resize round(size*256/224) và crop size; giữ tỷ lệ crop giống I00.', '',
        '## Hiệu chuẩn', '',
        f"I07 khớp T={t['T']:.6f} trên val bằng NLL; miền tìm [exp(-6),exp(6)], "
        f"biên tìm kiếm: {t['at_search_boundary']}. ECE {t['before']['ece']:.5f} → {t['after']['ece']:.5f}; "
        f"NLL {t['before']['nll']:.5f} → {t['after']['nll']:.5f}. Top-1 và macro-F1 giữ nguyên.",
        'Khớp và đo ECE trên cùng val nên kết quả hiệu chuẩn có thể lạc quan; không khớp lại T trên test. '
        'I07 chỉ hiệu chuẩn 1-view. Không giả định T này tối ưu cho TTA hay độ phân giải khác.', '',
        '## Gộp view và độ phân giải', '',
        f"Hflip: gộp prob I01 F1 {rows[1]['macro_f1_val']:.2%}, gộp logit I03 F1 {rows[3]['macro_f1_val']:.2%}. "
        'Cùng hai view; chọn bằng val, không giả định một cách luôn tốt hơn.',
        'I04/I09/I10 là dò 256/288/320 từ cùng checkpoint, không đổi tham số. '
        'Không suy ra latency chỉ bằng FLOPs hay nhân K; tất cả được đo thật.', '',
        '## BN, EMA và ensemble', '',
        'ConvNeXt-Tiny có LayerNorm, không có BatchNorm2d: gộp BN không áp dụng. '
        'T05 không train EMA nên không tạo một mục EMA giả. Không thử soup/ensemble trong ngân sách '
        'này; đã có ít nhất bốn nhóm phương pháp ngoài 1-view (hflip, 5 crop, resolution, calibration, AMP).', '',
        '## Chọn bằng val', '',
        f"Ngoại tuyến: {offline['exp_id']} ({offline['method']}), F1 {offline['macro_f1_val']:.2%}, p95 {offline['p95_ms']:.3f} ms."]
    if realtime:
        lines.append(f"Có ràng buộc 100 ms: {realtime['exp_id']} ({realtime['method']}), F1 {realtime['macro_f1_val']:.2%}, p95 {realtime['p95_ms']:.3f} ms.")
    else:
        lines.append('Chưa có cấu hình p95<=100 ms; tiêu chí I5 chưa đạt. Không suy diễn số đo khác GPU.')
    fastest=min(rows,key=lambda r:r['p95_ms'])
    lines+= [f"Nhanh nhất theo p95: {fastest['exp_id']} ({fastest['p95_ms']:.3f} ms). "
             'Phương án K view cần đối chiếu mức tăng F1 với chi phí tương đối trong bảng. '
             'Nếu không tăng rõ, 1-view/calibration/AMP là lựa chọn cần cân nhắc cho xử lý trực tiếp.', '',
             'Mọi metric là một checkpoint/seed. Chênh lệch nhỏ chưa chứng minh cải thiện ổn định. '
             'Selection.json giữ phương pháp và T cụ thể; Bước 4 mới huấn luyện nhiều seed và chạy test.', '',
             '![Đánh đổi](accuracy_latency.png)', '![Hiệu chuẩn](calibration_reliability.png)', '',
             'Dự đoán: predictions/Ixx_seed0_val.csv. Logits/probs và raw timing: step3/Ixx/. '
             'Nguồn checkpoint/môi trường train: source.json. Môi trường suy luận thực tế: runtime.json. '
             'Không gộp latency từ Colab và Lightning. Conditions chi tiết: latency_batch1/32.json.']
    (root/'step3/step3_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return selection


def plot_calibration(root,before,after,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
    for axis,p,title in zip(axes,[before,after],['Before T (I00)','After val-fit T (I07)']):
        confidence=p.max(1); correct=p.argmax(1)==labels
        bins=np.clip(np.digitize(confidence,np.linspace(0,1,16),right=True)-1,0,14)
        points=[(confidence[bins==i].mean(),correct[bins==i].mean()) for i in range(15) if np.any(bins==i)]
        axis.plot([0,1],[0,1],'--',color='gray')
        axis.plot([v[0] for v in points],[v[1] for v in points],'o-')
        axis.set(xlabel='Mean confidence',ylabel='Accuracy',title=title,xlim=(0,1),ylim=(0,1)); axis.grid(alpha=.2)
    fig.suptitle('Val reliability: 15 equal-width bins (same val used to fit T)')
    fig.tight_layout(); fig.savefig(root/'step3/calibration_reliability.png',dpi=160); plt.close(fig)


def plot_tradeoff(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for row in rows:
        ax.scatter(row['p95_ms'],100*row['macro_f1_val'])
        ax.annotate(row['exp_id'],(row['p95_ms'],100*row['macro_f1_val']),xytext=(5,5),textcoords='offset points')
    ax.axvline(100,color='gray',linestyle='--',label='100 ms requirement')
    runtime=step1.read_json(root/'step3/runtime.json')
    ax.set(xlabel='GPU pipeline p95, batch 1 (ms)',ylabel='Macro-F1 val (%)',
           title=f"T05 / {runtime['platform']} / {runtime['gpu']}: accuracy vs measured latency")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(root/'step3/accuracy_latency.png',dpi=160); plt.close(fig)


def export_colab_xlsx(project):
    root=Path(project); rows=collect_results(root)
    step1.write_colab_sheet(root,rows,COLUMNS,'Inference',[
        'T05, seed 0, val only. Relative cost = batch-1 p50 / I00 p50.',
        'Timing: GPU-ready input; tensor views + forward + softmax/T. PIL/decode/transfers excluded.',
        'p95<=100ms applies to this pipeline, not total camera/robot latency.'])
    latency=step1.read_json(root/'step3/latency.json')
    return step1.write_colab_sheet(root,latency,LATENCY_COLUMNS,'Latency',[
        'Warmup 10 + 100 measured iterations, synchronized before/after each measurement.',
        'Throughput = batch / mean seconds. Raw samples are saved in step3/Ixx/latency_batch1/32.json.',
        'Actual inference environment and platform: step3/runtime.json (also embedded in each timing JSON).'])


def export_artifacts(project,out_path):
    import zipfile
    root=Path(project)
    with zipfile.ZipFile(out_path,'w',zipfile.ZIP_DEFLATED) as bundle:
        for folder in ('code','tests','starter','step1','step2','step3','curves','predictions'):
            for path in sorted((root/folder).rglob('*')):
                if path.is_file() and path.suffix in ('.py','.json','.csv','.md','.npy','.png','.ipynb'):
                    bundle.write(path,path.relative_to(root))
        for folder in sorted((root/'runs').glob('[BT]*/seed0')):
            for path in sorted(folder.glob('*')):
                if path.is_file() and path.suffix in ('.json','.csv','.npy'):
                    bundle.write(path,path.relative_to(root))
        for name in ('eval.py','results.xlsx','tools/gpu_budget.py'):
            bundle.write(root/name,name)
    return Path(out_path)
