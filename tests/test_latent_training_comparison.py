"""A high curve peak must not pass the minimum-val-loss checkpoint criterion."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import latent_dg_config as settings
from utils.summarize_latent_training import summarize


class TrainingSelectionTests(unittest.TestCase):
    def test_peak_cannot_replace_minimum_val_loss_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            suite=Path(temp);folder=suite/'training/hard_kway';(folder/'metrics').mkdir(parents=True)
            baseline=dict(validation_protocol=settings.VALIDATION,selected_map_50=.5,selected_epoch=31,selected_val_loss=1.)
            (suite/'manifest.json').write_text(json.dumps(dict(experiments=['hard_kway'],baseline=baseline)))
            (folder/'validation_protocol.json').write_text(json.dumps(settings.VALIDATION))
            records=[dict(epoch=e,val_loss=1. if e==1 else 2.,map_50=.49 if e==1 else .60 if e==2 else .48) for e in range(1,61)]
            with (folder/'metrics/metrics_epoch.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=['epoch','val_loss','map_50']);w.writeheader();w.writerows(records)
            (folder/'epoch_audit.jsonl').write_text(''.join(json.dumps(dict(epoch=e,images_seen=12742,validation_images=1592))+'\n' for e in range(1,61)))
            (folder/'metrics/validation_coco_metrics.jsonl').write_text(''.join(json.dumps(dict(epoch=r['epoch'],value=r['map_50'],images=1592,metric='source_val_coco_ap50',confidence_mode='legacy_uncertainty'))+'\n' for r in records))
            (folder/'completion.json').write_text(json.dumps(dict(smoke_test=False,epochs_completed=60)))
            (folder/'best_map_checkpoint.json').write_text(json.dumps(dict(epoch=2,map_50=.60)))
            (folder/'best_epoch_weights.pth').write_bytes(b'test-only-weight-placeholder')
            (folder/'best_val_training_state.pt').write_bytes(b'test-only-state-placeholder')
            digest=hashlib.sha256((folder/'best_epoch_weights.pth').read_bytes()).hexdigest()
            (folder/'best_checkpoint.json').write_text(json.dumps(dict(epoch=1,val_loss=1.,map_50=.49,weights_sha256=digest)))
            result=summarize(suite)
            row=result['experiments'][0]
            self.assertEqual(row['selected_epoch'],1)
            self.assertEqual(row['selected_map_50'],.49)
            self.assertEqual(row['peak_map_50'],.60)
            self.assertFalse(row['exceeds_baseline'])
            self.assertTrue(result['all_training_complete'])
            self.assertFalse(result['all_three_exceed_baseline'])
            # Sparse AP must still select epoch 1 by loss, NOT the measured epoch 5.
            (folder/'run_config.json').write_text(json.dumps(dict(coco_period=5)))
            for r in records:r['map_50']=.60 if r['epoch']%5==0 else None
            with (folder/'metrics/metrics_epoch.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=['epoch','val_loss','map_50']);w.writeheader();w.writerows(records)
            scheduled=[dict(epoch=r['epoch'],value=r['map_50'],images=1592,metric='source_val_coco_ap50',
                            confidence_mode='legacy_uncertainty') for r in records if r['map_50'] is not None]
            proof=dict(epoch=1,value=.49,images=1592,metric='source_val_coco_ap50',confidence_mode='legacy_uncertainty',
                       purpose='selected_checkpoint_replay',weights_sha256=digest)
            (folder/'metrics/validation_coco_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scheduled+[proof]))
            (folder/'best_map_checkpoint.json').write_text(json.dumps(dict(epoch=5,map_50=.60)))
            replay=dict(epoch=1,map_50=.49,weights_sha256=digest,validation_images=1592,metric='source_val_coco_ap50',
                        confidence_mode='legacy_uncertainty')
            (folder/'selected_checkpoint_coco.json').write_text(json.dumps(replay))
            row=summarize(suite)['experiments'][0]
            self.assertEqual(row['epochs_completed'],60)
            self.assertEqual(row['selected_epoch'],1)
            self.assertEqual(row['selected_map_50'],.49)
            self.assertEqual(row['peak_epoch'],5)
            self.assertFalse(row['reaches_51'])
            # A replay attached to the wrong epoch must fail, even if its AP is high.
            replay['epoch']=5
            (folder/'selected_checkpoint_coco.json').write_text(json.dumps(replay))
            with self.assertRaisesRegex(ValueError,'Invalid minimum-loss'):
                summarize(suite)
            replay['epoch']=1
            (folder/'selected_checkpoint_coco.json').write_text(json.dumps(replay))
            (folder/'best_epoch_weights.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'weights do not match'):
                summarize(suite)


if __name__=='__main__': unittest.main()
