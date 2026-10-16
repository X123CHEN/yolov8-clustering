"""Guard metric identity and historical confidence compatibility."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import latent_dg_config as settings
from utils.latent_validation import ValidationDecodeBox, CocoValidationCallback
from utils.utils_bbox import DecodeBox


class ValidationProtocolTests(unittest.TestCase):
    def test_four_presets_share_historical_training_protocol(self):
        protocols=[settings.resolve(name)['validation'] for name in settings.EXPERIMENTS]
        self.assertTrue(all(p == protocols[0] for p in protocols))
        self.assertEqual(protocols[0]['metric'], 'source_val_coco_ap50')
        self.assertEqual(protocols[0]['confidence_mode'], 'legacy_uncertainty')
        self.assertFalse(protocols[0]['voc_fallback'])

    def test_legacy_nms_matches_original_decoder(self):
        root=Path(__file__).resolve().parents[1]
        source=subprocess.check_output(['git','show','HEAD:utils/utils_bbox.py'],cwd=root)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'legacy.py';path.write_bytes(source)
            spec=importlib.util.spec_from_file_location('test_original_decoder',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            old=module.DecodeBox(4,[640,640])
            new=ValidationDecodeBox(4,[640,640],'legacy_uncertainty')
            torch.manual_seed(11)
            prediction=torch.rand(2,100,8)
            prediction[:,:,2:4]*=.4
            for threshold in [.001,.05,.5]:
                a=old.non_max_suppression(prediction.clone(),4,[640,640],[720,1280],False,threshold,.5)
                b=new.non_max_suppression(prediction.clone(),4,[640,640],[720,1280],False,threshold,.5)
                for x,y in zip(a,b):
                    if x is None:self.assertIsNone(y)
                    else:np.testing.assert_array_equal(x,y)

    def test_standard_mode_is_unchanged(self):
        prediction=torch.rand(1,50,8)
        a=DecodeBox(4,[640,640]).non_max_suppression(prediction.clone(),4,[640,640],[720,1280],False,.05,.5)
        b=ValidationDecodeBox(4,[640,640],'max_class_sigmoid').non_max_suppression(prediction.clone(),4,[640,640],[720,1280],False,.05,.5)
        np.testing.assert_array_equal(a[0],b[0])

    def test_coco_failure_is_not_relabelled_as_voc(self):
        with tempfile.TemporaryDirectory() as temp:
            callback=CocoValidationCallback(None,[640,640],['D00'],1,[],temp,False,
                map_out_path=str(Path(temp)/'predictions'),eval_flag=True,period=1)
            with patch('utils.latent_validation.get_coco_map',side_effect=RuntimeError('COCO failed')):
                with self.assertRaisesRegex(RuntimeError,'COCO failed'):
                    callback.on_epoch_end(1,None)
            self.assertFalse((Path(temp)/'validation_coco_metrics.jsonl').exists())

    def test_skipped_epochs_are_missing_and_selected_replay_records_original_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            callback=CocoValidationCallback(None,[640,640],['D00'],1,[],temp,False,
                map_out_path=str(Path(temp)/'predictions'),eval_flag=True,period=5)
            callback.last_map_50=.61
            with patch('utils.latent_validation.get_coco_map',return_value=[.3,.49]) as evaluate:
                self.assertIsNone(callback.on_epoch_end(1,None))
                evaluate.assert_not_called()
                value=callback.on_epoch_end(3,None,force=True,purpose='selected_checkpoint_replay',weights_sha256='selected-hash')
                self.assertEqual(value,.49)
            evidence=json.loads((Path(temp)/'validation_coco_metrics.jsonl').read_text())
            self.assertEqual(evidence['epoch'],3)
            self.assertEqual(evidence['purpose'],'selected_checkpoint_replay')
            self.assertEqual(evidence['weights_sha256'],'selected-hash')


if __name__=='__main__':unittest.main()
