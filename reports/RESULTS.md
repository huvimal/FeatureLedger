# FeatureLedger measured results

Synthetic, engineered leakage stress test. Metrics measure the demonstration, not production forecast quality.

| Join | Availability violations | Train rows | Test rows | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| safe | 0 | 1600 | 960 | 0.5823 | 0.6141 |
| event_only | 3051 | 1600 | 960 | 1.0000 | 1.0000 |

Both snapshots reconstructed with identical logical row fingerprints. A later-available append did not change the original safe snapshot.
The leaky arm uses outcome-informed corrections intentionally created after prediction. Its apparent performance is invalid for historical deployment.
Parquet byte hashes detect artifact changes; logical row hashes establish content equality. No claim of cross-version byte-identical Parquet, external data authenticity, real-world accuracy, or production security.
