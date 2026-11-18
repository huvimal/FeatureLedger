import json,re
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import HTMLResponse
from ledger import ROOT,PRIVATE,verify,rebuild,file_sha

@asynccontextmanager
async def lifespan(app):
    report=json.loads((ROOT/'reports/evaluation.json').read_text())
    for name,sha in report['source_sha256'].items():
        if file_sha(ROOT/name)!=sha:raise RuntimeError('Code changed after evaluation. Run pipeline.py again.')
    app.state.report=report
    app.state.snapshots={}
    for result in report['methods'].values():
        sid=result['snapshot_id'];manifest,rows=verify(PRIVATE/'snapshots'/sid)
        app.state.snapshots[sid]=(manifest,rows)
    yield

app=FastAPI(title='FeatureLedger',version='1.0.0',lifespan=lifespan)

@app.get('/report')
def report():return app.state.report

@app.get('/snapshots/{snapshot_id}')
def snapshot(snapshot_id:str,limit:int=Query(20,ge=1,le=100)):
    if snapshot_id not in app.state.snapshots:raise HTTPException(404,'Unknown snapshot')
    manifest,rows=app.state.snapshots[snapshot_id]
    return {'manifest':manifest,'rows':rows[:limit],'total_rows':len(rows),'limit':limit}

@app.get('/snapshots/{snapshot_id}/verify')
def verify_snapshot(snapshot_id:str):
    if snapshot_id not in app.state.snapshots:raise HTTPException(404,'Unknown snapshot')
    return rebuild(PRIVATE/'snapshots'/snapshot_id)

@app.get('/',response_class=HTMLResponse)
def index():return PAGE

PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FeatureLedger</title>
<style>body{margin:0;background:#091525;color:#ebf4ff;font:16px system-ui}main{max-width:1150px;margin:40px auto;padding:24px}h1{font-size:48px;margin:8px 0}section{background:#12263b;border:1px solid #294760;border-radius:16px;padding:22px;margin-top:22px}.accent{color:#5ce1c1}.muted,small{color:#abc0d4}button,select{font:inherit;padding:10px;background:#0d685f;color:white;border:1px solid #44a599;border-radius:8px;margin:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #294760}.scroll{overflow:auto}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warning{color:#ffd48b}</style>
<main><p class="accent">POINT-IN-TIME ML DATA PIPELINE</p><h1>FeatureLedger</h1><p>Track when it happened. Check when it became available.</p><p class="warning">Seeded synthetic stress test. Late corrections deliberately encode outcomes. Leaky metrics are invalid for historical prediction.</p>
<section><h2>Safe vs. intentionally leaky join</h2><div class="scroll"><table><thead><tr><th>Join</th><th>Availability violations</th><th>Train rows</th><th>Test rows</th><th>Accuracy</th><th>F1</th></tr></thead><tbody id="metrics"></tbody></table></div><p class="muted">Same logistic model and chronological cutoff. Training labels must already be available. Recurring fictional accounts appear across time.</p></section>
<section><h2>Snapshot and feature lineage</h2><label>Join mode <select id="mode"><option value="safe">Safe point-in-time</option><option value="event_only">Intentionally leaky</option></select></label><button id="check">Reconstruct snapshot</button><p id="status" role="status"></p><p>Snapshot ID: <code id="sid"></code></p><div class="scroll"><table><thead><tr><th>Observation</th><th>Feature</th><th>Prediction at</th><th>Event time</th><th>Available at</th><th>Revision</th><th>Value</th></tr></thead><tbody id="lineage"></tbody></table></div><p class="muted">First 20 feature rows. Missing values remain missing; imputation is fitted only on training data. Reconstruct checks stored source files and canonical row fingerprints.</p><details><summary>Snapshot manifest</summary><pre id="manifest"></pre></details></section>
<section><h2>Late-arrival stability</h2><p id="late"></p><p class="muted">A record whose availability is after the snapshot cutoff cannot change that snapshot. Stored inputs and Parquet are checked before reconstruction. Local hashes detect changes, but are not signed proof of source authenticity.</p></section></main>
<script>
let evidence;
const el=id=>document.getElementById(id);
function cell(row,v){let c=document.createElement('td');c.textContent=v??'missing';row.appendChild(c)}
async function get(url){let r=await fetch(url);let d=await r.json();if(!r.ok)throw Error(d.detail||'Request failed');return d}
async function show(){let mode=el('mode').value;let sid=evidence.methods[mode].snapshot_id;el('sid').textContent=sid;let d=await get('/snapshots/'+sid);el('manifest').textContent=JSON.stringify(d.manifest,null,2);el('lineage').replaceChildren();for(let x of d.rows){let tr=document.createElement('tr');for(let key of ['observation_id','feature','prediction_at','event_time','available_at','revision','value'])cell(tr,key==='value'&&x[key]!==null?x[key].toFixed(3):x[key]);el('lineage').appendChild(tr)}el('status').textContent=mode==='safe'?'Only features available by prediction time.':'WARNING: availability filter intentionally omitted.'}
el('mode').onchange=()=>show().catch(e=>el('status').textContent=e.message);
el('check').onclick=async()=>{el('check').disabled=true;try{let sid=evidence.methods[el('mode').value].snapshot_id;let d=await get('/snapshots/'+sid+'/verify');el('status').textContent=d.logical_rows_match?'Reconstruction passed: logical rows match.':'Reconstruction failed.'}catch(e){el('status').textContent=e.message}finally{el('check').disabled=false}};
async function init(){try{evidence=await get('/report');for(let [mode,m] of Object.entries(evidence.methods)){let r=document.createElement('tr');for(let v of [mode,m.availability_violations,m.train_rows,m.test_rows,m.test.accuracy.toFixed(3),m.test.f1.toFixed(3)])cell(r,v);el('metrics').appendChild(r)}el('late').textContent=evidence.late_append_old_snapshot_unchanged?'Passed: late append left the old safe snapshot unchanged.':'Not verified';await show()}catch(e){el('status').textContent=e.message}}init();
</script></html>'''
