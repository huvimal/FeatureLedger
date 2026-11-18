"""Real NumPy logistic training on safe and intentionally leaky snapshot features."""
import json,os,platform,time
import numpy as np
from ledger import ROOT,PRIVATE,FEATURES,build,persist,verify,rebuild,checksum,file_sha

AS_OF='2025-05-15T00:00:00+00:00'
TRAIN_CUTOFF='2025-03-14T00:00:00+00:00'

def matrix(rows):
    groups={}
    for row in rows:
        groups.setdefault(row['observation_id'],{})[row['feature']]=row
    observations=[];values=[];labels=[]
    for oid,features in sorted(groups.items()):
        first=features[FEATURES[0]];observations.append(first)
        # Missingness is retained as a model feature; imputation is fitted on train only.
        raw=[features[f]['value'] for f in FEATURES]
        values.append([np.nan if v is None else v for v in raw])
        labels.append(first['label'])
    return observations,np.asarray(values,dtype=float),np.asarray(labels,dtype=float)

def train_mask(observations,cutoff=TRAIN_CUTOFF):
    return np.array([o['prediction_at']<cutoff and o['label_available_at']<=cutoff for o in observations])

def train_logistic(x,y,mask):
    if not mask.any() or len(np.unique(y[mask]))<2:raise ValueError('Insufficient training labels')
    median=np.nanmedian(x[mask],axis=0)
    median=np.where(np.isfinite(median),median,0)
    imputed=np.where(np.isnan(x),median,x)
    mean=imputed[mask].mean(axis=0);scale=imputed[mask].std(axis=0);scale=np.maximum(scale,1e-8)
    z=np.column_stack([np.ones(len(x)),(imputed-mean)/scale,np.isnan(x).astype(float)])
    w=np.zeros(z.shape[1]);losses=[]
    for step in range(600):
        p=1/(1+np.exp(-np.clip(z[mask]@w,-30,30)))
        penalty=.01*w;penalty[0]=0
        w-=.2*(z[mask].T@(p-y[mask])/mask.sum()+penalty)
        if step%100==0:losses.append(float(-np.mean(y[mask]*np.log(np.maximum(p,1e-12))+(1-y[mask])*np.log(np.maximum(1-p,1e-12)))))
    probability=1/(1+np.exp(-np.clip(z@w,-30,30)))
    return probability,dict(weights=w.tolist(),train_median=median.tolist(),train_mean=mean.tolist(),train_scale=scale.tolist(),epochs=600,learning_rate=.2,l2=.01,training_loss_samples=losses)

def classification(y,p):
    predicted=p>=.5
    tp=int(np.sum(predicted&(y==1)));fp=int(np.sum(predicted&(y==0)));fn=int(np.sum(~predicted&(y==1)))
    return dict(rows=len(y),accuracy=float(np.mean(predicted==y)),f1=float(2*tp/max(2*tp+fp+fn,1)),
                log_loss=float(-np.mean(y*np.log(np.clip(p,1e-12,1-1e-12))+(1-y)*np.log(np.clip(1-p,1e-12,1-1e-12)))))

def main():
    start=time.perf_counter()
    events=json.loads((PRIVATE/'events.json').read_text());observations=json.loads((PRIVATE/'observations.json').read_text())
    report=dict(data_kind='seeded synthetic leakage stress test; not customer or business accuracy',seed=42,
        as_of=AS_OF,train_cutoff=TRAIN_CUTOFF,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        protocol='Same fixed logistic setup in both arms. Train labels must be available by cutoff. Future prediction test. No hyperparameter tuning. Leaky arm intentionally ignores feature availability.',methods={})
    for mode in ('safe','event_only'):
        bundle=build(events,observations,AS_OF,mode);path=persist(bundle)
        manifest,rows=verify(path);rebuilt=rebuild(path)
        obs,x,y=matrix(rows);train=train_mask(obs)
        test=np.array([o['prediction_at']>=TRAIN_CUTOFF for o in obs])
        p,model=train_logistic(x,y,train)
        model['snapshot_id']=bundle['snapshot_id']
        (PRIVATE/f'model-{mode}.json').write_text(json.dumps(model,indent=2))
        # Existing entities recur across time; this is a temporal forecasting slice, not unseen-account generalization.
        majority=float(y[train].mean()>=.5)
        report['methods'][mode]=dict(snapshot_id=bundle['snapshot_id'],feature_rows=len(rows),
            missing_feature_rows=bundle['missing_feature_rows'],availability_violations=bundle['availability_violations'],
            train_rows=int(train.sum()),test_rows=int(test.sum()),label_gap_rows=int((~train&~test).sum()),
            test=classification(y[test],p[test]),majority_baseline=classification(y[test],np.full(test.sum(),majority)),
            reconstruction=rebuilt,model_sha256=file_sha(PRIVATE/f'model-{mode}.json'))
    # Append late revision: old safe snapshot ID/content must remain unchanged.
    original=build(events,observations,AS_OF)
    appended=dict(events[0],revision=3,available_at='2025-06-01T00:00:00+00:00',value=999.0)
    later=build(events+[appended],observations,AS_OF)
    if original['snapshot_id']!=later['snapshot_id'] or original['rows_sha256']!=later['rows_sha256']:raise RuntimeError('Late append changed old snapshot')
    report['late_append_old_snapshot_unchanged']=True
    report['seconds']=time.perf_counter()-start
    report['source_sha256']={n:file_sha(ROOT/n) for n in ('ledger.py','generate.py','evaluate.py')}
    report['code_and_dependencies_are_part_of_snapshot_identity']=True
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    lines=['# FeatureLedger measured results','','Synthetic, engineered leakage stress test. Metrics measure the demonstration, not production forecast quality.','',
           '| Join | Availability violations | Train rows | Test rows | Accuracy | F1 |','|---|---:|---:|---:|---:|---:|']
    for mode,result in report['methods'].items():
        lines.append(f"| {mode} | {result['availability_violations']} | {result['train_rows']} | {result['test_rows']} | {result['test']['accuracy']:.4f} | {result['test']['f1']:.4f} |")
    lines+=['','Both snapshots reconstructed with identical logical row fingerprints. A later-available append did not change the original safe snapshot.',
            'The leaky arm uses outcome-informed corrections intentionally created after prediction. Its apparent performance is invalid for historical deployment.',
            'Parquet byte hashes detect artifact changes; logical row hashes establish content equality. No claim of cross-version byte-identical Parquet, external data authenticity, real-world accuracy, or production security.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
