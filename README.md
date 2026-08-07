# Advantage Spine

Open artifact for the Neurocomputing manuscript:

**The Advantage Spine: Dynamic Coordinate Selection for High-Rate RLVR**

Repository: https://github.com/vivia-an/advantage_spine_v1

## Contents

- `method/` — SparseAdamW drop-in used by the Spine recipe (`topk` @ 40%, negative-channel weight 0.1)
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

The paper claims a **matched-protocol** high-rate operating region for Spine.
A lightly tuned dense 20× arm recovers Mean4 from 0.340 to 0.383 but still
trails the Spine recipe at 0.445. The dense-1× vs recipe-20× interval is a
**nominal descriptive endpoint contrast** after the full ladder.

## Citation

See `CITATION.cff`. This GitHub repository is the canonical public archive.
