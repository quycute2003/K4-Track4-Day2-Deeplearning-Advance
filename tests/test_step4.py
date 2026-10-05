"""Final configuration and once-only test guards using temporary synthetic data."""
import importlib
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
from test_implementation import train

step4=importlib.import_module('deepweeds_lab.step4')


class TestFinals(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.root=Path(self.temporary.name)
        baseline=asdict(train.Config(backbone='convnext_tiny',epochs=10,batch_size=32))
        final=dict(baseline,loss='ls',label_smoothing=.1)
        plan=dict(code_sha256=step4.code_hash(),training_source={'F00':baseline,'F01':final},
                  inference={'F00':{'size':224},'F01':{'size':288}})
        train.write_json(self.root/'step4/frozen_plan.json',plan)

    def tearDown(self):
        self.temporary.cleanup()

    def ready(self):
        for g in step4.GROUPS:
            for s in step4.SEEDS:
                checkpoint=self.root/f'runs/{g}/seed{s}/best.pt'
                checkpoint.parent.mkdir(parents=True,exist_ok=True); checkpoint.write_bytes(b'fixture; not a torch model')
                with mock.patch.object(step4,'runtime_check',return_value={'fixture':True}):
                    signature=step4.val_metadata(self.root,g,s)
                train.write_json(step4.folder_at(self.root,g,s)/'val_done.json',dict(signature=signature,T=1.,test_used=False))

    def test_configs_keep_paired_seed_and_separate_training_from_test(self):
        a=step4.make_config(self.root,'images','labels','F00',2)
        b=step4.make_config(self.root,'images','labels','F01',2)
        self.assertEqual((a.seed,b.seed),(2,2)); self.assertEqual((a.epochs,b.epochs),(10,10))
        self.assertEqual((a.img_size,b.img_size),(224,224))
        self.assertEqual((a.loss,b.loss),('ce','ls')); self.assertEqual(b.label_smoothing,.1)
        self.assertFalse(a.save_test_predictions); self.assertFalse(b.save_test_predictions)
        with self.assertRaises(ValueError):
            step4.make_config(self.root,'images','labels','F01',3)

    def test_atomic_test_claim_rejects_second_start(self):
        folder=self.root/'once'; step4.claim_test(folder,{'seed':0})
        with self.assertRaises(FileExistsError):
            step4.claim_test(folder,{'seed':0})

    def test_interrupted_test_without_complete_logits_cannot_rerun(self):
        self.ready(); folder=step4.folder_at(self.root,'F00',0)
        step4.claim_test(folder,step4.step1.read_json(folder/'val_done.json'))
        with mock.patch.object(step4,'runtime_check',return_value={'fixture':True}), \
             mock.patch.object(step4,'load_model',side_effect=AssertionError('must not rerun GPU')):
            with self.assertRaisesRegex(RuntimeError,'already started'):
                step4.test_once(self.root,'images','labels','F00',0)

    def test_complete_test_cache_is_reused_without_model_forward(self):
        self.ready(); folder=step4.folder_at(self.root,'F00',0)
        frozen=step4.step1.read_json(folder/'val_done.json'); step4.claim_test(folder,frozen)
        y=np.arange(3507)%9; names=[f'fixture{i}.jpg' for i in range(len(y))]
        logits=np.zeros((len(y),9)); logits[np.arange(len(y)),y]=4.
        np.save(folder/'test_logits.npy',logits); np.save(folder/'test_labels.npy',y)
        train.write_json(folder/'test_filenames.json',names)
        labels_dir=self.root/'labels'; labels_dir.mkdir()
        pd.DataFrame({'Filename':names,'Label':y}).to_csv(labels_dir/'test_subset0.csv',index=False)
        with mock.patch.object(step4,'runtime_check',return_value={'fixture':True}), \
             mock.patch.object(step4,'load_model',side_effect=AssertionError('must use cache')):
            result=step4.test_once(self.root,'images',labels_dir,'F00',0)
        self.assertEqual(result['metrics']['macro_f1'],1.)
        self.assertEqual(result['n'],3507)


if __name__=='__main__':
    unittest.main()
