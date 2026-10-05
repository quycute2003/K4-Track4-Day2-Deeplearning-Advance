"""Preregistered paired-seed finals; one test forward pass per model/seed."""
from __future__ import annotations
import gc
import hashlib
import io
import json
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
if __package__:
    from . import train,dataset,model,inference,benchmark,step1,step3
else:
    import train,dataset,model,inference,benchmark,step1,step3

SEEDS=(0,1,2)
GROUPS=('F00','F01')
FINAL_COLUMNS=['exp_id','seed','recipe','inference','temperature','macro_f1_val','top1_val',
    'macro_f1_test','top1_test','ece_test','ece_test_uncal','p95_ms','best_epoch','status']


def code_hash():
    return hashlib.sha256(b''.join((Path(__file__).parent/f'{n}.py').read_bytes()
        for n in ('step4','train','dataset','model','losses','inference','benchmark'))).hexdigest()


def freeze_plan(project):
    root=Path(project)
    rows=step3.collect_results(root)
    selected=step1.read_json(root/'step3/selection.json')
    if any(r['status']!='Đủ kết quả' for r in rows) or selected['deployment_exp_id']!='I09':
        raise ValueError('Finish Step 3; this final protocol requires the val-selected I09')
    plan=dict(protocol=1,seeds=list(SEEDS),groups=list(GROUPS),backbone='convnext_tiny.fb_in1k',
        training_source={g:step1.read_json(root/f'runs/{s}/seed0/config.json')['config']
                         for g,s in [('F00','B03'),('F01','T05')]},
        inference=dict(F00=dict(size=224,dtype='fp32',temperature='1'),
                       F01=dict(size=288,dtype='fp32',temperature='fit once on own val NLL')),
        selection=selected,runtime=step1.read_json(root/'step3/runtime.json'),code_sha256=code_hash(),
        test_policy='Train all six runs and freeze all val temperatures before any test; one forward pass per model/seed',
        calibration='F01 calibrated is primary; F01_uncal derives from the same logits; no additional test pass',
        checkpoint_policy='Highest macro-F1 on 224 val; earliest epoch on ties',test_used=False)
    path=root/'step4/frozen_plan.json'
    if path.exists() and step1.read_json(path)!=plan:
        raise ValueError('Final protocol changed. Preserve the existing plan; do not retune after test.')
    train.write_json(path,plan)
    return plan


def plan_at(root):
    plan=step1.read_json(Path(root)/'step4/frozen_plan.json')
    if plan['code_sha256']!=code_hash():
        raise ValueError('Final code differs from frozen protocol')
    return plan


def runtime_check(root):
    plan=plan_at(root)
    actual=dict(step3.runtime_details(),platform='Lightning AI')
    if actual!=plan['runtime']:
        raise RuntimeError('Use the same Lightning T4/PyTorch environment as Step 3 for both final groups')
    return actual


def make_config(root,images_dir,labels_dir,group,seed):
    if group not in GROUPS or seed not in SEEDS:
        raise ValueError('Unknown final group/seed')
    root=Path(root); plan=plan_at(root)
    values=dict(plan['training_source'][group])
    values.update(exp_id=group,seed=seed,images_dir=str(images_dir),labels_dir=str(labels_dir),
        out_dir=str(root/'runs'),pred_dir=str(root/'predictions'),curves_dir=str(root/'curves'),
        save_test_predictions=False,device='cuda',resume=False)
    return train.Config(**values)


def run_training(project,images_dir,labels_dir,group,seed):
    root=Path(project); runtime_check(root)
    cfg=make_config(root,images_dir,labels_dir,group,seed)
    folder=train.run_dir(cfg)
    if (folder/'config.json').exists():
        saved=step1.read_json(folder/'config.json')['config']
        if any(saved[k]!=v for k,v in asdict(cfg).items() if k!='resume'):
            raise ValueError('Existing final training configuration differs')
    if (folder/'summary.json').exists():
        print(f'{group} seed {seed}: completed training reused; no retraining.')
        return step1.read_json(folder/'summary.json')
    if (folder/'last.pt').exists():
        cfg.resume=True
    result=train.run(cfg)
    gc.collect(); torch.cuda.empty_cache()
    return result


def folder_at(root,group,seed):
    return Path(root)/'step4'/group/f'seed{seed}'


def load_model(root,group,seed):
    root=Path(root); plan=plan_at(root)
    path=root/f'runs/{group}/seed{seed}/best.pt'
    state=torch.load(path,map_location='cuda',weights_only=True)
    network=model.build_model(plan['backbone'],pretrained=False).cuda().eval()
    network.load_state_dict(state['model'],strict=True); network.requires_grad_(False)
    return network


def loader_at(root,images_dir,labels_dir,group,seed,split):
    if split not in ('val','test'):
        raise ValueError('Only val/test evaluation')
    root=Path(root); plan=plan_at(root)
    frame=pd.read_csv(Path(labels_dir)/f'{split}_subset0.csv'); dataset._validate_frame(frame,split)
    record=step1.read_json(root/f'runs/{group}/seed{seed}/config.json')
    cfg=record['pretrained_cfg']; size=plan['inference'][group]['size']
    transform=dataset.build_transforms(False,size,mean=cfg['mean'],std=cfg['std'],interpolation=cfg['interpolation'])
    return dataset.make_loader(frame,images_dir,transform,32,False,num_workers=2,seed=seed)


def write_prediction(root,exp,seed,split,names,labels,logits,t):
    path=Path(root)/'predictions'/f'{exp}_seed{seed}_{split}.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    train.ev.save_predictions(str(path),names,labels,inference.apply_temperature(logits,t))
    pred=train.ev.read_pred(str(path))
    return {k:v for k,v in train.ev.compute_metrics(pred.y_true,pred.y_pred,pred.probs).items() if k in train.ev.SCALARS}


def val_metadata(root,group,seed):
    root=Path(root)
    return dict(group=group,seed=seed,checkpoint_sha256=step3.sha256(root/f'runs/{group}/seed{seed}/best.pt'),
        plan_sha256=step3.sha256(root/'step4/frozen_plan.json'),runtime=runtime_check(root))


def prepare_val(project,images_dir,labels_dir,group,seed):
    root=Path(project); folder=folder_at(root,group,seed); signature=val_metadata(root,group,seed)
    path=folder/'val_done.json'
    if path.exists():
        saved=step1.read_json(path)
        if saved['signature']!=signature:
            raise ValueError('Val checkpoint/protocol/runtime changed')
        return saved
    if (folder/'test_started.json').exists():
        raise RuntimeError('Cannot refit val choices after test started')
    network=load_model(root,group,seed)
    names,labels,logits=inference.predict_logits(network,loader_at(root,images_dir,labels_dir,group,seed,'val'),'cuda')
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'val_logits.npy',logits); np.save(folder/'val_labels.npy',labels)
    train.write_json(folder/'val_filenames.json',names)
    t=inference.fit_temperature(logits,labels,split='val') if group=='F01' else 1.
    before=write_prediction(root,group+'_uncal',seed,'val',names,labels,logits,1.) if group=='F01' else None
    metrics=write_prediction(root,group,seed,'val',names,labels,logits,t)
    timing=benchmark.pipeline_report(network,batch_size=1,img_size=plan_at(root)['inference'][group]['size'],
        temperature=t,dtype='fp32',device='cuda',warmup=10,iters=100)
    timing.update(runtime=signature['runtime'],checkpoint_sha256=signature['checkpoint_sha256'])
    train.write_json(folder/'latency_batch1.json',timing)
    result=dict(signature=signature,T=t,fit_split='val' if group=='F01' else None,
        val_metrics=metrics,val_uncal_metrics=before,p95_ms=timing['p95'],test_used=False)
    train.write_json(path,result)
    del network; gc.collect(); torch.cuda.empty_cache()
    return result


def claim_test(folder,signature):
    """Atomic once-only ledger; incomplete GPU passes cannot be silently rerun."""
    path=Path(folder)/'test_started.json'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(signature,stream,ensure_ascii=False,indent=2)


def test_once(project,images_dir,labels_dir,group,seed):
    root=Path(project); folder=folder_at(root,group,seed)
    # Freeze ALL choices before opening any test predictions.
    for g in GROUPS:
        for s in SEEDS:
            saved=step1.read_json(folder_at(root,g,s)/'val_done.json')
            if saved['signature']!=val_metadata(root,g,s):
                raise ValueError('Train/prepare all six frozen val runs before test')
    frozen=step1.read_json(folder/'val_done.json')
    marker=folder/'test_started.json'; logits_path=folder/'test_logits.npy'
    cache_names=folder/'test_filenames.json'; cache_labels=folder/'test_labels.npy'
    if marker.exists():
        if step1.read_json(marker)!=frozen or not all(p.exists() for p in (logits_path,cache_names,cache_labels)):
            raise RuntimeError('Test was already started but no complete matching cache exists. Keep diagnostics; do not rerun test.')
        names=step1.read_json(cache_names); labels=np.load(cache_labels); logits=np.load(logits_path)
        print(f'{group} seed {seed}: reuse saved test logits; no model rerun.')
    else:
        network=load_model(root,group,seed)
        loader=loader_at(root,images_dir,labels_dir,group,seed,'test')
        claim_test(folder,frozen)
        names,labels,logits=inference.predict_logits(network,loader,'cuda')
        np.save(folder/'test_labels.npy',labels); train.write_json(cache_names,names)
        np.save(logits_path,logits)
        del network; gc.collect(); torch.cuda.empty_cache()
    if len(names)!=3507 or logits.shape!=(3507,9):
        raise ValueError('Expected full test fold 0, 3507 images')
    metrics=write_prediction(root,group,seed,'test',names,labels,logits,frozen['T'])
    pred=train.ev.read_pred(str(root/f'predictions/{group}_seed{seed}_test.csv'))
    train.ev.check_against_csv(pred,str(Path(labels_dir)/'test_subset0.csv'),'test')
    uncal=write_prediction(root,group+'_uncal',seed,'test',names,labels,logits,1.) if group=='F01' else None
    result=dict(group=group,seed=seed,val=frozen,metrics=metrics,uncal_metrics=uncal,n=3507,test_forward_passes=1)
    train.write_json(folder/'test_done.json',result)
    return result


def summarize(project,labels_dir):
    root=Path(project); plan=plan_at(root); rows=[]
    for group in GROUPS:
        for seed in SEEDS:
            done=step1.read_json(folder_at(root,group,seed)/'test_done.json')
            val=done['val']; metrics=done['metrics']
            rows.append(dict(exp_id=group,seed=seed,recipe='T00' if group=='F00' else 'T05',
                inference='I00' if group=='F00' else 'I09 + val-fit T',temperature=val['T'],
                macro_f1_val=val['val_metrics']['macro_f1'],top1_val=val['val_metrics']['top1'],
                macro_f1_test=metrics['macro_f1'],top1_test=metrics['top1'],ece_test=metrics['ece'],
                ece_test_uncal=done['uncal_metrics']['ece'] if done['uncal_metrics'] else metrics['ece'],
                p95_ms=val['p95_ms'],best_epoch=step1.read_json(root/f'runs/{group}/seed{seed}/summary.json')['best_epoch'],status='Đủ kết quả'))
    out=root/'step4/eval_outputs'; logs=[]
    common=['--test-csv',str(Path(labels_dir)/'test_subset0.csv'),'--labels',str(Path(labels_dir)/'labels.csv'),'--out',str(out)]
    for tag in ('F00','F01','F01_uncal'):
        buffer=io.StringIO()
        with redirect_stdout(buffer),redirect_stderr(buffer):
            result=train.ev.main(['score','--pred',str(root/f'predictions/{tag}_seed*_test.csv'),'--tag',tag,*common])
        if result!=0:
            raise ValueError(buffer.getvalue())
        logs.append(buffer.getvalue())
    worst=max(r['p95_ms'] for r in rows if r['exp_id']=='F01'); buffer=io.StringIO()
    with redirect_stdout(buffer),redirect_stderr(buffer):
        result=train.ev.main(['grade','--final',str(root/'predictions/F01_seed*_test.csv'),
            '--baseline',str(root/'predictions/F00_seed*_test.csv'),'--uncal',str(root/'predictions/F01_uncal_seed*_test.csv'),
            '--final-val',str(root/'predictions/F01_seed*_val.csv'),'--val-csv',str(Path(labels_dir)/'val_subset0.csv'),
            '--latency-p95-ms',str(worst),*common])
    if result!=0:
        raise ValueError(buffer.getvalue())
    logs.append(buffer.getvalue()); (out/'eval_output.md').write_text('\n\n'.join(logs),encoding='utf-8')
    train.write_json(root/'step4/final.json',rows)
    summary={g:step1.read_json(out/f'{g}_summary.json') for g in GROUPS}
    train.write_json(root/'step4/summary.json',summary)
    per_class=[]
    for group in GROUPS:
        frame=pd.read_csv(out/f'{group}_per_class.csv'); frame.insert(0,'exp_id',group)
        per_class.extend(frame.to_dict('records'))
    train.write_json(root/'step4/per_class.json',per_class)
    step1.write_colab_sheet(root,rows,FINAL_COLUMNS,'Final',['F00=T00/I00; F01=T05/I09 + own val-fit T. Three paired seeds; test once per model/seed.'])
    step1.write_colab_sheet(root,per_class,list(per_class[0]),'PerClass',['Mean and sample std across three seeds; support is per seed.'])
    # Required explicit aggregate mean/std rows, kept separate from per-seed values.
    aggregate=[dict(exp_id=g,**{f'{k}_{stat}':summary[g][k][stat] for k in train.ev.SCALARS for stat in ('mean','std')}) for g in GROUPS]
    for entry in aggregate:
        for key in ('macro_f1_val','top1_val'):
            mean,std=train.ev.mean_std([r[key] for r in rows if r['exp_id']==entry['exp_id']])
            entry[key+'_mean']=mean; entry[key+'_std']=std
    train.write_json(root/'step4/final_stats.json',aggregate)
    step1.write_colab_sheet(root,aggregate,list(aggregate[0]),'FinalStats',['Three seeds per group; sample std ddof=1.'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group in GROUPS:
        cm=pd.read_csv(out/f'{group}_confusion_sum.csv',index_col=0)
        fig,ax=plt.subplots(figsize=(10,8)); graphic=ax.imshow(cm.values,cmap='Blues'); fig.colorbar(graphic,ax=ax)
        labels=summary[group]['classes']; ax.set_xticks(range(9),labels,rotation=45,ha='right'); ax.set_yticks(range(9),labels)
        for i in range(9):
            for j in range(9):
                ax.text(j,i,int(cm.iloc[i,j]),ha='center',va='center',color='white' if cm.iloc[i,j]>cm.values.max()/2 else 'black',fontsize=8)
        ax.set(xlabel='Predicted',ylabel='True',title=f'{group}: sum of test confusion counts, 3 seeds')
        fig.tight_layout(); fig.savefig(root/f'step4/{group}_confusion.png',dpi=150); plt.close(fig)
    return summary


def export(project,out_path,*,checkpoints=False):
    import zipfile
    root=Path(project)
    with zipfile.ZipFile(out_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for group in GROUPS:
            for seed in SEEDS:
                folder=root/f'runs/{group}/seed{seed}'
                for path in sorted(folder.glob('*')):
                    if path.is_file() and (path.name=='best.pt' if checkpoints else path.suffix in ('.json','.csv','.npy')):
                        archive.write(path,path.relative_to(root))
        if not checkpoints:
            for folder in ('step4','code','tests'):
                for path in sorted((root/folder).rglob('*')):
                    if path.is_file() and path.suffix in ('.py','.json','.csv','.npy','.md','.png'):
                        archive.write(path,path.relative_to(root))
            for folder in ('predictions','curves'):
                for path in sorted((root/folder).glob('F*')):
                    if path.is_file():
                        archive.write(path,path.relative_to(root))
            for name in ('eval.py','results.xlsx'):
                archive.write(root/name,name)
    return Path(out_path)
