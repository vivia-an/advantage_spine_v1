# Paper D cross-cluster audit export

This directory is the author-attested, normalized audit export used with the
Paper D manuscript. Canonical run times include the documented -31-day
correction for the source cluster clock. The training and evaluation contract
is recorded in `evaluation_and_training_protocol.json` (PASP core-four avg@32).

## Verification boundary

Fields prefixed `reported_` are values attested in the source export. Fields
prefixed `local_` and the digests in `manifest_index.csv` are recomputed from
files physically present in this package. `PROVENANCE.json` defines the
verification boundary.

Local verification (`python verify_remote_audit.py audited_remote_export`)
checks package structure, internal identities, protocol fields, counts, and
local file integrity. It does **not** remount or byte-verify remote raw
checkpoints or logs (`remote_raw_artifacts_mounted_locally` is false).

## Qwen3-1.7B scores

`qwen17b_evaluation_counts.csv` is the authoritative Qwen3-1.7B score audit and
retains integer correct/total counts. Earlier Qwen3-1.7B AIME item labels in
`aime_sample_output_manifest.csv` did not include a corrected per-output
mapping, so their sample identities are retained but their `correct` field is
blank and marked superseded; no item label is fabricated to force the
corrected aggregate.
