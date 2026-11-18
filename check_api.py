"""Actual stored snapshot/API checks, not a browser or HTTP load test."""
from fastapi.testclient import TestClient
from app import app

def main():
    with TestClient(app) as client:
        assert 'FeatureLedger' in client.get('/').text
        report=client.get('/report').json()
        assert report['methods']['safe']['availability_violations']==0
        sid=report['methods']['safe']['snapshot_id']
        result=client.get('/snapshots/'+sid).json()
        assert len(result['rows'])==20
        assert client.get('/snapshots/'+sid,params={'limit':101}).status_code==422
        assert client.get('/snapshots/'+'0'*64).status_code==404
        assert client.get('/snapshots/'+sid+'/verify').json()['logical_rows_match']
        leaky=report['methods']['event_only']['snapshot_id']
        assert client.get('/snapshots/'+leaky).json()['manifest']['availability_violations']>0
    print('API CHECK PASSED: real snapshots, lineage, reconstruction and request limits. Browser layout not tested.')
if __name__=='__main__':main()
