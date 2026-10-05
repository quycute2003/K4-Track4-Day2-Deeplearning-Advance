"""Cache alignment and protocol guards using temporary synthetic fixtures only."""
import importlib
import tempfile
import unittest
from unittest import mock
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from test_implementation import train

step3=importlib.import_module('deepweeds_lab.step3')


class TestStep3(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        cfg=train.Config(exp_id='B03',backbone='convnext_tiny',epochs=10,batch_size=32)
        record=dict(config=asdict(cfg),gpu='Tesla T4',versions=dict(torch='fixture',torchvision='fixture',timm='fixture'),
                    pretrained_cfg=dict(tag='fb_in1k',mean=[.485,.456,.406],std=[.229,.224,.225],interpolation='bicubic'))
        train.write_json(self.root/'runs/B03/seed0/config.json',record)
        record['config'].update(exp_id='T05',loss='ls',label_smoothing=.1)
        train.write_json(self.root/'runs/T05/seed0/config.json',record)
        train.write_json(self.root/'step2/selection.json',dict(source_exp_id='T05',test_used=False,seed=0))
        folder=self.root/'runs/T05/seed0'
        (folder/'best.pt').write_bytes(b'fixture hash only; never loaded as a model')
        self.labels=np.tile(np.arange(9),3); self.names=[f'{i}.jpg' for i in range(len(self.labels))]
        predicted=self.labels.copy(); predicted[::4]=(predicted[::4]+1)%9
        logits=np.zeros((len(self.labels),9)); logits[np.arange(len(self.labels)),predicted]=.5
        np.save(folder/'val_logits.npy',logits); np.save(folder/'val_labels.npy',self.labels)
        train.write_json(folder/'val_filenames.json',self.names)
        probs=train.probabilities(logits)
        metrics=train.ev.compute_metrics(self.labels,predicted,probs)
        train.write_json(folder/'summary.json',dict(exp_id='T05',seed=0,test_evaluated=False,
            macro_f1_val=metrics['macro_f1'],top1_val=metrics['top1'],ece_val=metrics['ece']))
        (self.root/'predictions').mkdir()
        train.ev.save_predictions(str(self.root/'predictions/T05_seed0_val.csv'),self.names,self.labels,probs)

    def tearDown(self):
        self.temp.cleanup()

    def test_offline_preparation_calibrates_real_arrays_but_never_invents_latency(self):
        rows=step3.prepare_offline(self.root)
        self.assertIsNotNone(rows[0]['macro_f1_val'])
        self.assertTrue(all(row['p95_ms'] is None for row in rows))
        self.assertTrue(all(row['status']!='Đủ kết quả' for row in rows))
        ts=next(row for row in rows if row['exp_id']=='I07')
        self.assertEqual(ts['macro_f1_val'],rows[0]['macro_f1_val'])
        self.assertLess(ts['nll_val'],rows[0]['nll_val'])
        with mock.patch.object(step3.inference,'fit_temperature',side_effect=AssertionError('must reuse exact T')):
            self.assertEqual(step3.prepare_offline(self.root)[0]['macro_f1_val'],rows[0]['macro_f1_val'])
        with self.assertRaisesRegex(ValueError,'Finish all'):
            step3.write_report(self.root)
        self.assertFalse((self.root/'step3/selection.json').exists())

    def test_changed_checkpoint_or_logit_cache_cannot_reuse_protocol(self):
        step3.prepare_offline(self.root)
        (self.root/'runs/T05/seed0/best.pt').write_bytes(b'another checkpoint')
        with self.assertRaisesRegex(ValueError,'protocol/checkpoint/code changed'):
            step3.prepare_offline(self.root)

    def test_csv_alignment_checked_before_any_image_is_read(self):
        step3.prepare_offline(self.root)
        labels_dir=self.root/'labels'; labels_dir.mkdir()
        frame=pd.DataFrame(dict(Filename=list(reversed(self.names)),Label=self.labels))
        frame.to_csv(labels_dir/'val_subset0.csv',index=False)
        with self.assertRaisesRegex(ValueError,'CSV order/labels'):
            step3.make_loader(self.root,'nonexistent-images',labels_dir,'I00',num_workers=0)

    def test_modified_prediction_metrics_are_rejected(self):
        step3.prepare_offline(self.root)
        path=self.root/'step3/I00/result.json'
        result=step3.step1.read_json(path); result['metrics']['macro_f1']=1.
        train.write_json(path,result)
        with self.assertRaisesRegex(ValueError,'metrics differ'):
            step3.collect_results(self.root)

    def test_raw_timing_percentiles_and_environment_are_verified(self):
        step3.prepare_offline(self.root)
        source=step3.step1.read_json(self.root/'step3/source.json')
        result=step3.step1.read_json(self.root/'step3/I00/result.json')
        times=np.linspace(1.,2.,100)
        q=np.percentile(times,[50,95,99])
        runtime=dict(platform='Lightning AI',gpu='Tesla T4',torch='lightning-fixture')
        train.write_json(self.root/'step3/runtime.json',runtime)
        data=dict(source=source,runtime=runtime,method=step3.METHODS['I00'],exp_id='I00',batch=1,gpu='Tesla T4',torch='lightning-fixture',
            dtype='fp32',temperature=1.,img_size=224,input_size=224,views=1,fused_bn=False,warmup=10,n=100,
            preprocessing_included=False,transfer_included=False,samples_ms=times.tolist(),
            p50=float(q[0]),p95=float(q[1]),p99=float(q[2]),mean=float(times.mean()),images_per_s=1000/times.mean())
        path=self.root/'step3/I00/latency_batch1.json'; train.write_json(path,data)
        self.assertIsNotNone(step3.checked_latency(self.root,'I00',1,source,result))
        data['p95']=.01; train.write_json(path,data)
        with self.assertRaisesRegex(ValueError,'differs from raw'):
            step3.checked_latency(self.root,'I00',1,source,result)
        data['gpu']='another GPU'; train.write_json(path,data)
        with self.assertRaisesRegex(ValueError,'environment mismatch'):
            step3.checked_latency(self.root,'I00',1,source,result)

    def test_lightning_environment_keeps_training_provenance_and_blocks_mixed_runtime(self):
        training_before=(self.root/'runs/T05/seed0/config.json').read_bytes()
        actual=dict(gpu='Tesla T4',torch='2.11.0+cu126',torchvision='0.26.0+cu126',timm='1.0.30')
        with mock.patch.object(step3,'runtime_details',return_value=actual.copy()):
            first=step3.assert_environment(self.root,platform='Lightning AI')
            self.assertEqual(first['torch'],'2.11.0+cu126')
            self.assertEqual(step3.assert_environment(self.root),first)
            with self.assertRaisesRegex(RuntimeError,'environment changed'):
                step3.assert_environment(self.root,platform='Colab')
        with mock.patch.object(step3,'runtime_details',return_value=dict(actual,torch='2.12.0+cu126')):
            with self.assertRaisesRegex(RuntimeError,'environment changed'):
                step3.assert_environment(self.root)
        self.assertEqual((self.root/'runs/T05/seed0/config.json').read_bytes(),training_before)

    def test_full_val_verification_detects_drift_at_last_example(self):
        step3.prepare_offline(self.root)
        runtime=dict(platform='Lightning AI',gpu='Tesla T4',torch='fixture')
        logits=np.load(self.root/'runs/T05/seed0/val_logits.npy')
        network=torch.nn.Linear(3,9).eval()
        network._step3_source_sha=step3.sha256(self.root/'runs/T05/seed0/best.pt')
        with mock.patch.object(step3,'assert_environment',return_value=runtime), \
             mock.patch.object(step3,'make_loader',return_value='val-only'), \
             mock.patch.object(step3.inference,'predict_logits',return_value=(self.names,self.labels,logits)):
            self.assertEqual(step3.verify_model(self.root,'images','labels',network),0.)
        result=step3.step1.read_json(self.root/'step3/model_check.json')
        self.assertEqual(result['n'],len(self.labels)); self.assertTrue(result['numerically_aligned'])
        drift=logits.copy(); drift[-1,0]+=1.
        with mock.patch.object(step3,'assert_environment',return_value=runtime), \
             mock.patch.object(step3,'make_loader',return_value='val-only'), \
             mock.patch.object(step3.inference,'predict_logits',return_value=(self.names,self.labels,drift)):
            with self.assertRaisesRegex(ValueError,'differs from cached'):
                step3.verify_model(self.root,'images','labels',network)


if __name__=='__main__':
    unittest.main()
