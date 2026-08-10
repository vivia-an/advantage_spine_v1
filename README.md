# Advantage Spine

Open artifact for the Neurocomputing manuscript:

**The Advantage Spine: Dynamic Coordinate Selection for High-Rate RLVR**

Repository: https://github.com/vivia-an/advantage_spine_v1

## Contents

- `method/` — SparseAdamW drop-in for the Advantage Spine (dynamic top-40%
  support) and the exact loss-side meaning of the recipe's negative-advantage
  coefficient 0.1
- `audited_*.csv` — seed-level and aggregate result tables
- `verify_numbers.py` / `verify_remote_audit.py` — consistency guards
- `audited_remote_export/` — author-attested cross-cluster export
- `main_ne.tex` — manuscript source (Elsevier `elsarticle` 5p)
- figures, highlights, cover letter, declarations

## Quick verify

```bash
python3 verify_numbers.py
python3 verify_remote_audit.py audited_remote_export
```

## Claim boundary

The paper claims a **matched-protocol** high-rate operating region for the
Spine recipe (dynamic support plus negative-advantage down-weighting). An
exploratory tuned-dense 20× aggregate diagnostic recovers Mean4 from 0.340 to
0.383 but still trails the Spine recipe at 0.445. The dense-1× vs recipe-20× interval is a
**nominal descriptive endpoint contrast** after the full ladder.

The public audit contains verified mask identities, protocol fields, and hash
records. It does not redistribute the remote raw mask files or checkpoints.

## Citation

See `CITATION.cff`. The canonical public archive is the GitHub repository above.
