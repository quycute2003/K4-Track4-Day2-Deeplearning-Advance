"""Synchronized repeated inference timings; preprocessing boundary explicit."""
from __future__ import annotations
import copy
import time
import numpy as np
import torch
if __package__:
    from . import inference
else:
    import inference

def bench(fn,warmup=10,iters=100,sync=None):
    if warmup<10 or iters<50:
        raise ValueError('Require warmup>=10 and at least 50 measured iterations')
    for _ in range(warmup):
        fn()
    samples=[]
    for _ in range(iters):
        if sync is not None:
            sync()
        begin=time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter()-begin)*1000)
    values=np.asarray(samples)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Invalid measured duration')
    q=np.percentile(values,[50,95,99])
    return dict(p50=float(q[0]),p95=float(q[1]),p99=float(q[2]),mean=float(values.mean()),
                n=iters,warmup=warmup,samples_ms=samples,
                timer='perf_counter; synchronization before/after each measurement')

def device_details(device):
    d=torch.device(device)
    return dict(gpu=torch.cuda.get_device_name(d) if d.type=='cuda' else None,device=str(d),
                torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)

@torch.inference_mode()
def pipeline_report(model,*,batch_size=1,img_size=224,input_size=None,kind='single',space='prob',
                    temperature=1.,dtype='fp32',device='cuda',warmup=10,iters=100):
    if batch_size<=0 or img_size<=0:
        raise ValueError('Positive batch/image sizes required')
    model.eval()
    size=input_size or img_size
    x=torch.randn(batch_size,3,size,size,device=device)
    sync=(lambda:torch.cuda.synchronize(device)) if torch.device(device).type=='cuda' else None
    fn=lambda:inference.forward_probs(model,x,kind=kind,space=space,temperature=temperature,dtype=dtype)
    result=bench(fn,warmup,iters,sync)
    result.update(device_details(device),dtype=dtype,batch=batch_size,img_size=img_size,input_size=size,
                  views=5 if kind=='fivecrop' else 2 if kind=='hflip' else 1,
                  images_per_s=batch_size*1000/result['mean'],throughput_basis='batch / mean measured seconds',
                  preprocessing_included=False,transfer_included=False,
                  timing_scope='normalized GPU input; tensor TTA + model + softmax/aggregation/T',
                  fused_bn=False)
    return result

@torch.inference_mode()
def latency_report(model,batch_size,img_size,dtype='fp32',device='cuda',warmup=10,iters=100):
    """Compatibility API: model forward only; FP16 does not mutate original."""
    if dtype not in ('fp32','amp','fp16') or batch_size<=0 or img_size<=0:
        raise ValueError('Invalid benchmark configuration')
    network=copy.deepcopy(model).to(device).eval()
    if dtype=='fp16':
        network.half()
    x=torch.randn(batch_size,3,img_size,img_size,device=device,
                  dtype=torch.float16 if dtype=='fp16' else torch.float32)
    def fn():
        with inference.autocast_context(device,'amp' if dtype=='amp' else 'fp32'):
            return network(x)
    sync=(lambda:torch.cuda.synchronize(device)) if torch.device(device).type=='cuda' else None
    result=bench(fn,warmup,iters,sync)
    result.update(device_details(device),dtype=dtype,batch=batch_size,img_size=img_size,
                  images_per_s=batch_size*1000/result['mean'],preprocessing_included=False,
                  fused_bn=bool(getattr(network,'fused_bn_count',0)),timing_scope='model forward only')
    return result

def tta_latency(model,k_views,**kw):
    """Measure actual K forwards, not K times a single-view timing."""
    if k_views<1:
        raise ValueError('Positive number of views required')
    batch_size=kw.pop('batch_size',1); size=kw.pop('img_size',224)
    device=kw.pop('device','cuda'); dtype=kw.pop('dtype','fp32')
    warmup=kw.pop('warmup',10); iters=kw.pop('iters',100)
    if kw:
        raise TypeError(f'Unexpected timing options: {kw}')
    model.eval()
    x=torch.randn(batch_size,3,size,size,device=device)
    @torch.inference_mode()
    def fn():
        with inference.autocast_context(device,dtype):
            return torch.stack([model(x).float().softmax(-1) for _ in range(k_views)]).mean(0)
    sync=(lambda:torch.cuda.synchronize(device)) if torch.device(device).type=='cuda' else None
    result=bench(fn,warmup,iters,sync)
    result.update(device_details(device),views=k_views,batch=batch_size,img_size=size,dtype=dtype,
                  images_per_s=batch_size*1000/result['mean'],timing_scope='K forwards + prob averaging',
                  preprocessing_included=False)
    return result
