"""Seeded fictional subscription risk records with deliberately outcome-informed corrections."""
import json
from datetime import datetime,timedelta,timezone
import numpy as np
from ledger import PRIVATE,canonical

def generate(seed=42,entities=160,weeks=16):
    rng=np.random.default_rng(seed);events=[];observations=[]
    origin=datetime(2025,1,3,tzinfo=timezone.utc)
    iso=lambda dt:dt.isoformat(timespec='seconds')
    for entity in range(entities):
        entity_id=f'account-{entity:04d}';base=rng.normal()
        for week in range(weeks):
            prediction=origin+timedelta(days=week*7)
            risk=.6*base+rng.normal(scale=.7)
            label=int(rng.random()<1/(1+np.exp(-risk)))
            oid=f'{entity_id}-w{week:02d}'
            observations.append(dict(observation_id=oid,entity_id=entity_id,prediction_at=iso(prediction),
                label=label,label_available_at=iso(prediction+timedelta(days=3))))
            for feature in ('usage_risk','support_risk'):
                eid=oid+'-'+feature
                event_time=iso(prediction-timedelta(days=1))
                late=bool(rng.random()<.2)
                available=prediction+timedelta(days=2) if late else prediction-timedelta(hours=12)
                value=risk*.6+rng.normal(scale=.9)
                events.append(dict(event_id=eid,entity_id=entity_id,feature=feature,revision=1,
                                   event_time=event_time,available_at=iso(available),value=float(value)))
                if feature=='usage_risk':
                    # Deliberately constructed leakage stress test, not a real-world accuracy dataset.
                    events.append(dict(event_id=eid,entity_id=entity_id,feature=feature,revision=2,
                        event_time=event_time,available_at=iso(prediction+timedelta(days=5)),
                        value=float((2*label-1)*2+rng.normal(scale=.1))))
    return events,observations

def main():
    PRIVATE.mkdir(exist_ok=True)
    events,observations=generate()
    (PRIVATE/'events.json').write_bytes(canonical(events))
    (PRIVATE/'observations.json').write_bytes(canonical(observations))
    print('Generated fictional records:',len(events),'event revisions;',len(observations),'observations.')
if __name__=='__main__':main()
