"""Immutable, content-addressed point-in-time datasets using DuckDB/Parquet."""
import hashlib,json,math,os,re,shutil,tempfile
from datetime import datetime,timezone
from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parent
PRIVATE=ROOT/'private'
FEATURES=('usage_risk','support_risk')
VERSION='point-in-time-v1'

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def checksum(value):return hashlib.sha256(canonical(value)).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def timestamp(value):
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None or dt.utcoffset().total_seconds()!=0:raise ValueError('UTC timestamp required')
    return dt.astimezone(timezone.utc).isoformat(timespec='seconds')

def validate(events,observations):
    revisions=set();event_keys={};event_versions={};observation_keys=set()
    for e in events:
        if e['feature'] not in FEATURES:raise ValueError('Unknown feature')
        if not isinstance(e['revision'],int) or isinstance(e['revision'],bool) or e['revision']<1:raise ValueError('Invalid revision')
        if not math.isfinite(float(e['value'])):raise ValueError('Non-finite feature')
        if e['event_time']!=timestamp(e['event_time']) or e['available_at']!=timestamp(e['available_at']):raise ValueError('Use canonical UTC timestamps')
        if e['available_at']<e['event_time']:raise ValueError('Availability precedes event')
        key=(e['event_id'],e['revision'])
        if key in revisions:raise ValueError('Duplicate event revision')
        revisions.add(key)
        identity=(e['entity_id'],e['feature'],e['event_time'])
        if e['event_id'] in event_keys and event_keys[e['event_id']]!=identity:raise ValueError('Revision changed event identity')
        event_keys[e['event_id']]=identity
        event_versions.setdefault(e['event_id'],[]).append(e)
    for event_id in event_keys:
        versions=sorted(event_versions[event_id],key=lambda e:e['revision'])
        if any(a['available_at']>=b['available_at'] for a,b in zip(versions,versions[1:])):raise ValueError('Revisions must become available in order')
    for o in observations:
        if o['observation_id'] in observation_keys:raise ValueError('Duplicate observation')
        observation_keys.add(o['observation_id'])
        if o['prediction_at']!=timestamp(o['prediction_at']) or o['label_available_at']!=timestamp(o['label_available_at']):raise ValueError('Use canonical UTC timestamps')
        if o['label'] not in (0,1):raise ValueError('Binary label required')
        if o['label_available_at']<=o['prediction_at']:raise ValueError('Future label required')

def sql(mode):
    if mode not in ('safe','event_only'):raise ValueError('Unsupported join mode')
    availability='AND e.available_at <= o.prediction_at' if mode=='safe' else ''
    # Filter before ORDER BY so a late revision cannot hide the previous usable value.
    return f'''SELECT o.observation_id,o.entity_id,o.prediction_at,o.label,o.label_available_at,
        f.feature,e.event_id,e.revision,e.event_time,e.available_at,e.value
        FROM observations o CROSS JOIN features f
        LEFT JOIN LATERAL (
          SELECT * FROM events e WHERE e.entity_id=o.entity_id AND e.feature=f.feature
          AND e.event_time<=o.prediction_at AND e.available_at<=? {availability}
          ORDER BY e.event_time DESC,e.available_at DESC,e.revision DESC,e.event_id DESC LIMIT 1
        ) e ON TRUE ORDER BY o.observation_id,f.feature'''

def build(events,observations,as_of,mode='safe'):
    as_of=timestamp(as_of)
    validate(events,observations)
    # The same snapshot cannot see subsequently appended, later-available records.
    visible=sorted((e for e in events if e['available_at']<=as_of),key=lambda e:(e['event_id'],e['revision']))
    obs=sorted((o for o in observations if o['label_available_at']<=as_of and o['prediction_at']<=as_of),key=lambda o:o['observation_id'])
    if not visible or not obs:raise ValueError('No visible training dataset')
    con=duckdb.connect(':memory:')
    try:
        con.execute('CREATE TABLE events(event_id VARCHAR,entity_id VARCHAR,feature VARCHAR,revision INTEGER,event_time VARCHAR,available_at VARCHAR,value DOUBLE)')
        con.executemany('INSERT INTO events VALUES (?,?,?,?,?,?,?)',[[e[k] for k in ('event_id','entity_id','feature','revision','event_time','available_at','value')] for e in visible])
        con.execute('CREATE TABLE observations(observation_id VARCHAR,entity_id VARCHAR,prediction_at VARCHAR,label INTEGER,label_available_at VARCHAR)')
        con.executemany('INSERT INTO observations VALUES (?,?,?,?,?)',[[o[k] for k in ('observation_id','entity_id','prediction_at','label','label_available_at')] for o in obs])
        con.execute('CREATE TABLE features(feature VARCHAR)')
        con.executemany('INSERT INTO features VALUES (?)',[(f,) for f in FEATURES])
        result=con.execute(sql(mode),[as_of]);names=[c[0] for c in result.description]
        rows=[dict(zip(names,row)) for row in result.fetchall()]
    finally:con.close()
    violations=sum(r['available_at'] is not None and r['available_at']>r['prediction_at'] for r in rows)
    if mode=='safe' and violations:raise RuntimeError('Point-in-time invariant failed')
    spec=dict(version=VERSION,engine=duckdb.__version__,builder_sha256=file_sha(__file__),mode=mode,as_of=as_of,
              features=list(FEATURES),query_sha256=hashlib.sha256(sql(mode).encode()).hexdigest(),
              events_sha256=checksum(visible),observations_sha256=checksum(obs))
    return dict(snapshot_id=checksum(spec),spec=spec,events=visible,observations=obs,rows=rows,
                rows_sha256=checksum(rows),availability_violations=violations,
                missing_feature_rows=sum(r['value'] is None for r in rows))

def verify(path):
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text())
    if path.name!=manifest['snapshot_id'] or checksum(manifest['spec'])!=manifest['snapshot_id']:raise ValueError('Snapshot identity mismatch')
    if set(manifest['files'])!={'events.json','observations.json','dataset.parquet'}:raise ValueError('Incomplete snapshot manifest')
    for name,sha in manifest['files'].items():
        if name not in ('events.json','observations.json','dataset.parquet'):raise ValueError('Unexpected manifest file')
        if file_sha(path/name)!=sha:raise ValueError('Snapshot artifact changed: '+name)
    con=duckdb.connect(':memory:')
    try:
        cursor=con.execute('SELECT * FROM read_parquet(?) ORDER BY observation_id,feature',[str(path/'dataset.parquet')])
        names=[c[0] for c in cursor.description]
        rows=[dict(zip(names,r)) for r in cursor.fetchall()]
    finally:con.close()
    if checksum(rows)!=manifest['rows_sha256']:raise ValueError('Dataset content mismatch')
    return manifest,rows

def persist(bundle,root=None):
    root=Path(root or PRIVATE/'snapshots');root.mkdir(parents=True,exist_ok=True)
    dest=root/bundle['snapshot_id']
    if dest.exists():
        manifest,_=verify(dest)
        if manifest['rows_sha256']!=bundle['rows_sha256']:raise ValueError('Immutable snapshot conflict')
        return dest
    stage=Path(tempfile.mkdtemp(prefix='.building-',dir=root))
    try:
        for name in ('events','observations'):(stage/(name+'.json')).write_bytes(canonical(bundle[name]))
        con=duckdb.connect(':memory:')
        try:
            schema='observation_id VARCHAR,entity_id VARCHAR,prediction_at VARCHAR,label INTEGER,label_available_at VARCHAR,feature VARCHAR,event_id VARCHAR,revision INTEGER,event_time VARCHAR,available_at VARCHAR,value DOUBLE'
            con.execute('CREATE TABLE dataset('+schema+')')
            keys=list(bundle['rows'][0])
            con.executemany('INSERT INTO dataset VALUES ('+','.join('?' for _ in keys)+')',[[r[k] for k in keys] for r in bundle['rows']])
            target=str(stage/'dataset.parquet').replace("'","''")
            con.execute("COPY (SELECT * FROM dataset ORDER BY observation_id,feature) TO '"+target+"' (FORMAT PARQUET, COMPRESSION ZSTD)")
        finally:con.close()
        manifest={k:bundle[k] for k in ('snapshot_id','spec','rows_sha256','availability_violations','missing_feature_rows')}
        manifest['feature_rows']=len(bundle['rows'])
        manifest['files']={n:file_sha(stage/n) for n in ('events.json','observations.json','dataset.parquet')}
        (stage/'manifest.json').write_bytes(canonical(manifest))
        os.rename(stage,dest)
        verify(dest)
        return dest
    finally:
        if stage.exists():
            if stage.resolve().parent!=root.resolve() or not stage.name.startswith('.building-'):raise RuntimeError('Unsafe staging cleanup path')
            shutil.rmtree(stage)

def rebuild(path):
    path=Path(path);manifest,rows=verify(path)
    rebuilt=build(json.loads((path/'events.json').read_text()),json.loads((path/'observations.json').read_text()),
                  manifest['spec']['as_of'],manifest['spec']['mode'])
    if rebuilt['snapshot_id']!=manifest['snapshot_id'] or rebuilt['rows_sha256']!=manifest['rows_sha256']:raise ValueError('Snapshot cannot be reconstructed with this builder')
    return {'snapshot_id':rebuilt['snapshot_id'],'logical_rows_match':True,'feature_rows':len(rows)}
