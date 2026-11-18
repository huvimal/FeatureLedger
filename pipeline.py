"""Bounded local workflow; no provider, credentials or external data required."""
from generate import main as generate
from evaluate import main as evaluate
if __name__=='__main__':
    generate()
    evaluate()
    print('FEATURELEDGER PIPELINE COMPLETE. Snapshots and model evidence saved.')
