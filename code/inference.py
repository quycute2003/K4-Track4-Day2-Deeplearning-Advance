"""Evaluation-only prediction, TTA, calibration and dataflow-safe Conv/BN fusion."""
from __future__ import annotations
import copy
import math
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn

def _logits(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=2 or not a.shape[0] or a.shape[1]<2 or not np.isfinite(a).all():
        raise ValueError('Expected finite nonempty logits [N,C]')
    return a

def apply_temperature(logits,T:float):
    if not math.isfinite(T) or T<=0:
        raise ValueError('Temperature must be finite and positive')
    z=_logits(logits)/T
    z-=z.max(axis=1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(axis=1,keepdims=True)

def aggregate_views(logits_per_view,space='prob'):
    views=[_logits(v) for v in logits_per_view]
    if not views or any(v.shape!=views[0].shape for v in views):
        raise ValueError('Views must have matching shapes and aligned filenames')
    if space=='prob':
        return ensemble_probs([apply_temperature(v,1) for v in views])
    if space=='logit':
        return apply_temperature(np.mean(views,axis=0),1)
    raise ValueError('Aggregation space must be prob or logit')

def ensemble_probs(list_of_probs):
    values=[_logits(v) for v in list_of_probs]
    if not values or any(v.shape!=values[0].shape for v in values):
        raise ValueError('Ensemble arrays must have matching shapes and aligned filenames')
    if any((v<0).any() or not np.allclose(v.sum(axis=1),1,atol=1e-6) for v in values):
        raise ValueError('Expected normalized probabilities')
    p=np.mean(values,axis=0)
    return p/p.sum(axis=1,keepdims=True)

def fit_temperature(val_logits,val_labels,*,split='val')->float:
    """Optimize val NLL with coarse search then golden-section in log T.
    Search T in exp([-6,6]); NLL is convex in inverse T, hence unimodal in log T.
    """
    if split!='val':
        raise ValueError('Fit temperature on val only')
    z=_logits(val_logits)
    y=np.asarray(val_labels)
    if y.shape!=(len(z),) or not np.issubdtype(y.dtype,np.integer) or (y<0).any() or (y>=z.shape[1]).any():
        raise ValueError('Invalid val labels')
    def nll(log_t):
        a=z/math.exp(log_t); m=a.max(axis=1)
        return float((m+np.log(np.exp(a-m[:,None]).sum(axis=1))-a[np.arange(len(y)),y]).mean())
    grid=np.linspace(-6,6,49); scores=[nll(v) for v in grid]
    best=int(np.argmin(scores))
    lo,hi=grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]
    ratio=(math.sqrt(5)-1)/2
    c,d=hi-ratio*(hi-lo),lo+ratio*(hi-lo); fc,fd=nll(c),nll(d)
    for _ in range(70):
        if fc<fd:
            hi,d,fd=d,c,fc; c=hi-ratio*(hi-lo); fc=nll(c)
        else:
            lo,c,fc=c,d,fd; d=lo+ratio*(hi-lo); fd=nll(d)
    candidates=[0.,float(grid[best]),(lo+hi)/2]
    return math.exp(min(candidates,key=nll))

def view_identity(x):
    return x

def view_hflip(x):
    if x.ndim!=4:
        raise ValueError('Expected images [N,C,H,W]')
    return torch.flip(x,dims=(-1,))

def views_multicrop(x,crop):
    if x.ndim!=4 or crop<=0 or crop>min(x.shape[-2:]):
        raise ValueError('Crop must fit image batch')
    h,w=x.shape[-2:]
    offsets=[(0,0),(0,w-crop),(h-crop,0),(h-crop,w-crop),
             (round((h-crop)/2),round((w-crop)/2))]
    return [x[...,top:top+crop,left:left+crop] for top,left in offsets]

def views_multiscale(x,sizes):
    if x.ndim!=4 or not sizes or any(s<=0 for s in sizes):
        raise ValueError('Positive sizes required for [N,C,H,W]')
    return [torch.nn.functional.interpolate(x,size=(s,s),mode='bilinear',align_corners=False,
                                            antialias=True) for s in sizes]

def autocast_context(device,dtype):
    if dtype=='fp32':
        return nullcontext()
    if dtype!='amp' or torch.device(device).type!='cuda':
        raise ValueError('AMP requires CUDA; choose dtype fp32 or amp')
    return torch.autocast('cuda',dtype=torch.float16)

@torch.inference_mode()
def predict_views(model,loader,device,views=None,*,dtype='fp32'):
    """Return ordered filenames, labels and [K,N,C] FP32 logits; eval only."""
    model.eval()
    names,labels,chunks=[],[],[]; k=None
    for x,y,filenames in loader:
        x=x.to(device,non_blocking=True)
        batches=[x] if views is None else views(x)
        if not batches or (k is not None and len(batches)!=k):
            raise ValueError('Consistent positive view count required')
        k=len(batches)
        with autocast_context(device,dtype):
            logits=[model(v).float() for v in batches]
        if any(v.ndim!=2 or len(v)!=len(y) or not torch.isfinite(v).all() for v in logits):
            raise FloatingPointError('Invalid/nonfinite model logits')
        names.extend(str(v) for v in filenames)
        labels.append(np.asarray(y.cpu(),dtype=np.int64))
        chunks.append(torch.stack(logits).cpu().numpy())
    if not chunks or len(names)!=len(set(names)):
        raise ValueError('Empty loader or duplicate filenames')
    return names,np.concatenate(labels),np.concatenate(chunks,axis=1)

def predict_logits(model,loader,device,view=None,*,dtype='fp32'):
    views=None if view is None else lambda x:[view(x)]
    names,y,logits=predict_views(model,loader,device,views,dtype=dtype)
    return names,y,logits[0]

@torch.inference_mode()
def forward_probs(model,x,*,kind='single',space='prob',temperature=1.,dtype='fp32'):
    """GPU pipeline: tensor views + forward(s) + FP32 softmax/aggregation.
    Input normalized and already on device; five-crop input 256x256.
    Excludes PIL/decode/H2D, includes tensor TTA and temperature division.
    """
    if kind not in ('single','hflip','fivecrop') or space not in ('prob','logit'):
        raise ValueError('Unsupported method')
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid temperature')
    if model.training:
        model.eval()
    batches=([x,view_hflip(x)] if kind=='hflip' else views_multicrop(x,224) if kind=='fivecrop' else [x])
    with autocast_context(x.device,dtype):
        values=[model(v).float() for v in batches]
    if len(values)==1:
        return (values[0]/temperature).softmax(-1)
    if temperature!=1:
        raise ValueError('This protocol calibrates single-view logits only')
    if space=='logit':
        return torch.stack(values).mean(0).softmax(-1)
    return torch.stack([v.softmax(-1) for v in values]).mean(0)

def fuse_conv_bn(model):
    """Fuse Conv2d→BN2d edges with one consumer; original model is preserved.
    Registration order is not dataflow, so use FX edges. No BN: unchanged copy.
    """
    result=copy.deepcopy(model).eval()
    if not any(isinstance(m,nn.BatchNorm2d) for m in result.modules()):
        result.fused_bn_count=0
        return result
    from torch.fx import symbolic_trace
    try:
        traced=symbolic_trace(result)
    except Exception as exc:
        raise ValueError('Cannot establish Conv/BN dataflow for safe fusion') from exc
    count=0
    for node in traced.graph.nodes:
        if node.op!='call_module' or not isinstance(traced.get_submodule(node.target),nn.BatchNorm2d):
            continue
        prior=node.args[0]
        if not hasattr(prior,'op') or prior.op!='call_module' or len(prior.users)!=1:
            continue
        if any(sum(n.op=='call_module' and n.target==target for n in traced.graph.nodes)!=1
               for target in (prior.target,node.target)):
            continue
        conv=traced.get_submodule(prior.target); bn=traced.get_submodule(node.target)
        if not isinstance(conv,nn.Conv2d):
            continue
        fused=torch.nn.utils.fuse_conv_bn_eval(conv,bn)
        for target,replacement in [(prior.target,fused),(node.target,nn.Identity())]:
            parent,_,leaf=target.rpartition('.')
            setattr(traced.get_submodule(parent) if parent else traced,leaf,replacement)
        count+=1
    traced.fused_bn_count=count
    return traced
