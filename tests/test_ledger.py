import copy,json,tempfile,unittest
from pathlib import Path
from ledger import build,persist,verify,rebuild,checksum,validate
from evaluate import matrix,train_mask,train_logistic,classification
import numpy as np

def source():
    events=[dict(event_id='usage-1',entity_id='a',feature='usage_risk',revision=1,event_time='2025-01-01T00:00:00+00:00',available_at='2025-01-01T01:00:00+00:00',value=.2),
            dict(event_id='usage-1',entity_id='a',feature='usage_risk',revision=2,event_time='2025-01-01T00:00:00+00:00',available_at='2025-01-04T00:00:00+00:00',value=9.0)]
    observations=[dict(observation_id='o1',entity_id='a',prediction_at='2025-01-02T00:00:00+00:00',label=1,label_available_at='2025-01-03T00:00:00+00:00')]
    return events,observations

class LedgerTests(unittest.TestCase):
    def test_late_revision_does_not_hide_usable_previous_value(self):
        e,o=source();safe=build(e,o,'2025-01-05T00:00:00+00:00')
        usage=next(r for r in safe['rows'] if r['feature']=='usage_risk')
        self.assertEqual(usage['value'],.2);self.assertEqual(usage['revision'],1)
        bad=build(e,o,'2025-01-05T00:00:00+00:00','event_only')
        self.assertEqual(bad['availability_violations'],1)
    def test_newer_unavailable_event_falls_back_to_older_event(self):
        e,o=source();e.append(dict(e[0],event_id='usage-2',event_time='2025-01-01T12:00:00+00:00',available_at='2025-01-04T00:00:00+00:00',value=3))
        b=build(e,o,'2025-01-05T00:00:00+00:00')
        self.assertEqual(next(r['event_id'] for r in b['rows'] if r['feature']=='usage_risk'),'usage-1')
    def test_future_event_and_other_entity_not_joined(self):
        e,o=source();e.extend([dict(e[0],event_id='future',event_time='2025-01-03T00:00:00+00:00',available_at='2025-01-03T00:00:00+00:00'),dict(e[0],event_id='other',entity_id='b',value=88)])
        b=build(e,o,'2025-01-05T00:00:00+00:00')
        self.assertEqual(next(r['value'] for r in b['rows'] if r['feature']=='usage_risk'),.2)
    def test_exact_available_boundary_and_missingness(self):
        e,o=source();e[0]['available_at']=o[0]['prediction_at']
        b=build(e,o,'2025-01-05T00:00:00+00:00')
        self.assertEqual(b['missing_feature_rows'],1)
        self.assertIsNone(next(r['value'] for r in b['rows'] if r['feature']=='support_risk'))
    def test_snapshot_cutoff_excludes_late_labels(self):
        e,o=source()
        with self.assertRaises(ValueError):build(e,o,'2025-01-02T00:00:00+00:00')
    def test_late_revision_is_usable_for_a_later_prediction(self):
        e,o=source()
        o.append(dict(o[0],observation_id='o2',prediction_at='2025-01-05T00:00:00+00:00',label_available_at='2025-01-06T00:00:00+00:00'))
        b=build(e,o,'2025-01-07T00:00:00+00:00')
        values={r['observation_id']:r['value'] for r in b['rows'] if r['feature']=='usage_risk'}
        self.assertEqual(values,{'o1':.2,'o2':9.0})
    def test_duplicate_and_invalid_revisions_rejected(self):
        e,o=source()
        for modified in [e+[e[0]], [dict(e[0],value=float('nan'))], [e[0],dict(e[1],entity_id='b')], [e[0],dict(e[1],available_at=e[0]['available_at'])]]:
            with self.assertRaises(ValueError):validate(modified,o)
    def test_append_after_cutoff_preserves_snapshot(self):
        e,o=source();a=build(e,o,'2025-01-05T00:00:00+00:00')
        b=build(e+[dict(e[0],revision=3,available_at='2025-02-01T00:00:00+00:00',value=50)],o,'2025-01-05T00:00:00+00:00')
        self.assertEqual(a['snapshot_id'],b['snapshot_id']);self.assertEqual(a['rows_sha256'],b['rows_sha256'])
    def test_input_order_does_not_change_snapshot(self):
        e,o=source()
        self.assertEqual(build(e,o,'2025-01-05T00:00:00+00:00')['snapshot_id'],build(e[::-1],o,'2025-01-05T00:00:00+00:00')['snapshot_id'])
    def test_parquet_reconstruction_and_immutable_tamper_rejection(self):
        e,o=source();b=build(e,o,'2025-01-05T00:00:00+00:00')
        with tempfile.TemporaryDirectory() as td:
            path=persist(b,Path(td));self.assertTrue(rebuild(path)['logical_rows_match'])
            self.assertEqual(path,persist(b,Path(td)))
            (path/'events.json').write_text('[]')
            with self.assertRaises(ValueError):verify(path)
            with self.assertRaises(ValueError):persist(b,Path(td))
    def test_label_availability_gates_training(self):
        obs=[dict(prediction_at='2025-01-01',label_available_at='2025-03-20'),dict(prediction_at='2025-01-01',label_available_at='2025-03-01')]
        self.assertEqual(train_mask(obs).tolist(),[False,True])
    def test_imputation_and_scaling_use_training_only(self):
        x=np.array([[0.,np.nan],[2.,3.],[9999.,9999.]])
        _,model=train_logistic(x,np.array([0,1,1]),np.array([True,True,False]))
        self.assertEqual(model['train_median'],[1.,3.]);self.assertEqual(model['train_mean'],[1.,3.])
    def test_logistic_metrics_and_real_parameter_updates(self):
        x=np.array([[-2.,0],[-1.,0],[1.,0],[2.,0]])
        p,m=train_logistic(x,np.array([0,0,1,1]),np.ones(4,dtype=bool))
        self.assertGreater(m['weights'][1],0)
        self.assertEqual(classification(np.array([0,0,1,1]),p)['accuracy'],1)

if __name__=='__main__':unittest.main()
