"""Scientific correctness checks for TTA, calibration and benchmark boundaries."""
import importlib
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn
from test_implementation import train

inf=importlib.import_module('deepweeds_lab.inference')
bench=importlib.import_module('deepweeds_lab.benchmark')
step3=importlib.import_module('deepweeds_lab.step3')


class TestInference(unittest.TestCase):
    def test_temperature_preserves_classes_and_improves_underconfident_nll(self):
        labels=np.arange(90)%9
        predictions=labels.copy(); predictions[::4]=(predictions[::4]+1)%9
        logits=np.zeros((90,9)); logits[np.arange(90),predictions]=.5
        t=inf.fit_temperature(logits,labels)
        a=inf.apply_temperature(logits,1); b=inf.apply_temperature(logits,t)
        self.assertGreater(t,0); self.assertLess(t,1)
        np.testing.assert_array_equal(a.argmax(1),b.argmax(1))
        self.assertLess(train.ev.compute_metrics(labels,b.argmax(1),b)['nll'],
                        train.ev.compute_metrics(labels,a.argmax(1),a)['nll'])
        with self.assertRaisesRegex(ValueError,'val only'):
            inf.fit_temperature(logits,labels,split='test')
        for bad in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                inf.apply_temperature(logits,bad)

    def test_view_order_crop_geometry_flip_and_no_mutation(self):
        x=torch.arange(36).reshape(1,1,6,6).float(); before=x.clone()
        crops=inf.views_multicrop(x,4)
        self.assertEqual([v[0,0,0,0].item() for v in crops],[0,2,12,14,7])
        torch.testing.assert_close(inf.view_hflip(inf.view_hflip(x)),before)
        torch.testing.assert_close(x,before)
        self.assertEqual([v.shape[-1] for v in inf.views_multiscale(x,[4,8])],[4,8])
        with self.assertRaises(ValueError):
            inf.views_multicrop(x,7)

    def test_probability_and_logit_aggregation_are_distinct_and_normalized(self):
        views=[np.array([[4.,0.,0.]]),np.array([[0.,1.,0.]])]
        prob=inf.aggregate_views(views,'prob'); logits=inf.aggregate_views(views,'logit')
        self.assertFalse(np.allclose(prob,logits))
        np.testing.assert_allclose(prob.sum(1),1)
        np.testing.assert_allclose(logits.sum(1),1)
        with self.assertRaises(ValueError):
            inf.ensemble_probs([np.array([[.2,.2]])])
        with self.assertRaises(ValueError):
            inf.aggregate_views([np.zeros((1,9)),np.zeros((2,9))])

    def test_prediction_disables_dropout_gradients_and_preserves_file_order(self):
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__(); self.drop=nn.Dropout(.9); self.seen=[]
            def forward(self,x):
                self.seen.append((self.training,torch.is_grad_enabled()))
                return self.drop(x.mean((2,3)))
        network=Recorder().train()
        x=torch.arange(48).reshape(2,3,2,4).float()
        loader=[(x,torch.tensor([1,0]),['b.jpg','a.jpg'])]
        names,y,z=inf.predict_logits(network,loader,'cpu',inf.view_hflip)
        self.assertEqual(names,['b.jpg','a.jpg']); np.testing.assert_array_equal(y,[1,0])
        np.testing.assert_allclose(z,x.mean((2,3)).numpy())
        self.assertEqual(network.seen,[(False,False)])
        p=inf.forward_probs(network.train(),x,kind='hflip',space='logit')
        self.assertFalse(network.training)
        self.assertFalse(p.requires_grad)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            inf.predict_logits(network,[(x,torch.tensor([1,0]),['a','a'])],'cpu')

    def test_fusion_equivalence_and_original_model_preserved(self):
        torch.manual_seed(0)
        network=nn.Sequential(nn.Conv2d(3,4,3,padding=1,bias=False),nn.BatchNorm2d(4),nn.ReLU()).eval()
        network[1].running_mean.copy_(torch.randn(4)); network[1].running_var.copy_(torch.rand(4)+.5)
        x=torch.randn(3,3,12,12)
        with torch.no_grad():
            expected=network(x)
        fused=inf.fuse_conv_bn(network)
        actual=fused(x)
        self.assertLess(float((actual-expected).abs().max().detach()),1e-5)
        self.assertEqual(fused.fused_bn_count,1)
        self.assertIsInstance(network[1],nn.BatchNorm2d)
        self.assertIsInstance(fused.get_submodule('1'),nn.Identity)

    def test_fusion_skips_conv_used_by_another_branch_or_call(self):
        class Branched(nn.Module):
            def __init__(self):
                super().__init__(); self.conv=nn.Conv2d(3,3,1); self.bn=nn.BatchNorm2d(3)
            def forward(self,x):
                y=self.conv(x)
                return self.bn(y)+y
        class Shared(Branched):
            def forward(self,x):
                return self.bn(self.conv(x))+self.conv(x+1)
        x=torch.randn(2,3,4,4)
        for kind in (Branched,Shared):
            network=kind().eval(); fused=inf.fuse_conv_bn(network)
            torch.testing.assert_close(fused(x),network(x))
            self.assertEqual(fused.fused_bn_count,0)
        no_bn=inf.fuse_conv_bn(nn.Sequential(nn.Conv2d(3,3,1),nn.GroupNorm(1,3)))
        self.assertEqual(no_bn.fused_bn_count,0)


class TestBenchmark(unittest.TestCase):
    def test_warmup_excluded_and_every_measurement_bracketed_by_sync(self):
        events=[]
        clock=iter(v for i in range(50) for v in (float(i),float(i)+.002))
        def tick():
            events.append('clock'); return next(clock)
        with mock.patch.object(bench.time,'perf_counter',side_effect=tick):
            r=bench.bench(lambda:events.append('fn'),10,50,lambda:events.append('sync'))
        self.assertEqual(events[:10],['fn']*10)
        self.assertEqual(events[10:],['sync','clock','fn','sync','clock']*50)
        self.assertEqual((r['n'],r['warmup']),(50,10))
        self.assertAlmostEqual(r['p50'],2,places=8)
        self.assertLessEqual(r['p50'],r['p95']); self.assertLessEqual(r['p95'],r['p99'])
        for warmup,iters in ((9,100),(10,49)):
            with self.assertRaises(ValueError):
                bench.bench(lambda:None,warmup,iters)

    def test_full_pipeline_measurement_reports_batch_throughput_and_scope(self):
        class Mean(nn.Module):
            def forward(self,x):
                self.grad_enabled=torch.is_grad_enabled(); return x.mean((2,3))
        network=Mean().train()
        measured=dict(p50=2.,p95=3.,p99=4.,mean=2.5,n=100,warmup=10,samples_ms=[2.5]*100)
        def fake(fn,*args):
            fn(); return dict(measured)
        with mock.patch.object(bench,'bench',side_effect=fake):
            r=bench.pipeline_report(network,batch_size=32,img_size=8,device='cpu')
        self.assertFalse(network.training); self.assertFalse(network.grad_enabled)
        self.assertEqual(r['images_per_s'],12800.)
        self.assertFalse(r['preprocessing_included']); self.assertFalse(r['transfer_included'])

    def test_selection_requires_timings_and_enforces_realtime_p95(self):
        rows=[dict(exp_id='I00',status='Đủ kết quả',macro_f1_val=.9,ece_val=.1,p95_ms=8),
              dict(exp_id='I02',status='Đủ kết quả',macro_f1_val=.95,ece_val=.1,p95_ms=101),
              dict(exp_id='I07',status='Đủ kết quả',macro_f1_val=.9,ece_val=.01,p95_ms=9)]
        offline,realtime=step3.choose(rows)
        self.assertEqual(offline['exp_id'],'I02'); self.assertEqual(realtime['exp_id'],'I07')
        rows[0]['status']='Chờ latency T4'
        with self.assertRaisesRegex(ValueError,'Finish all'):
            step3.choose(rows)


if __name__=='__main__':
    unittest.main()
