"""Build an immutable snapshot or reconstruct one from its stored source records."""
import argparse,json,re
from ledger import PRIVATE,build,persist,rebuild

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--as-of',default='2025-05-15T00:00:00+00:00')
    parser.add_argument('--mode',choices=['safe','event_only'],default='safe')
    parser.add_argument('--reconstruct')
    args=parser.parse_args()
    if args.reconstruct:
        if not re.fullmatch('[a-f0-9]{64}',args.reconstruct):raise ValueError('A full snapshot ID is required')
        print(json.dumps(rebuild(PRIVATE/'snapshots'/args.reconstruct),indent=2))
    else:
        events=json.loads((PRIVATE/'events.json').read_text())
        observations=json.loads((PRIVATE/'observations.json').read_text())
        bundle=build(events,observations,args.as_of,args.mode)
        path=persist(bundle)
        print('SNAPSHOT READY:',path)
if __name__=='__main__':main()
