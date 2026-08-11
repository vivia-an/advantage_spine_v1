# Advantage Spine — Neurocomputing submission bundle

PaperD studies whether dynamic update-coordinate selection and negative-channel
down-weighting enlarge the usable learning-rate range of RLVR. The primary
evidence is a complete 2×2×4 factorial on Qwen3-8B with three matched seeds in
all 16 cells and verified effective learning rates. The dense-1× versus
recipe-20× paired interval is reported as a nominal descriptive endpoint
contrast after the full ladder.

## Public code and artifact

- GitHub (public): https://github.com/vivia-an/advantage_spine_v1
- Local immutable snapshot: `release/advantage-spine-v1.0/`
- Zip: `release/advantage-spine-v1.0.zip`
- Citation metadata: `CITATION.cff`

Synthetic schema fixtures and reconstructed staging bundles are development
history only and are **not** experimental evidence or part of the public
release.

## Build and verify

```bash
latexmk -pdf main_ne.tex
python3 verify_numbers.py
python3 verify_refs.py
```

The manuscript uses `elsarticle` in the Neurocomputing 5p two-column layout.
The intended compiled artifact is `main_ne.pdf`.

## Audited evidence

- `audited_factorial_results.csv` — 16 cells and 48 seed-level observations.
- `audited_primary_benchmark_results.csv` — three-seed Dense/Recipe
  decomposition over Math500, AIME24, AIME25, Olympiad, and Mean4.
- `audited_primary_benchmark_counts.csv` — primary Qwen3-8B per-seed integer
  correct/total records and unrounded count-derived benchmark rates.
- `audited_lr_audit.csv` — requested/optimizer/scheduler/logged LR agreement.
- `audited_spine_results.csv` — per-seed support agreement metrics.
- `audited_specificity_results.csv` — full Spine–dense cross-seed matrix.
- `audited_channel_overlap_results.csv` — Figure 2 three-seed channel activity,
  overlap, sign conflict, Gini, and top-1% energy records, with means, sample
  SDs, per-seed ratios, and the separately identified ratio of aggregates.
- `audited_mask_controls.csv` — three-seed dynamic, frozen, oracle, static and
  null-mask controls.
- `audited_stability_results.csv` — three-seed endpoint reward/KL/entropy/clip.
- `audited_transfer_results.csv` — three-seed DAPO and cross-model Mean4.
- `audited_transfer_support_results.csv` — distinct three-seed support metrics
  for GRPO, DAPO, Qwen3-1.7B, and Llama3.1-8B.
- `audited_efficiency_results.csv` — three-seed step-time and peak-memory
  summaries.
- `audited_qwen17b_results.csv` — Qwen3-1.7B per-seed four-benchmark counts.
- `audited_tuned_dense_results.csv` — three-seed exploratory tuned-dense
  $20\times$ diagnostic (0.383) with its exact tuning fields.
- `verify_numbers.py` — recomputation and cross-file consistency guard.
- The evidence level is author-attested seed-level result reproduction; raw
  remote masks, checkpoints, and full logs are not redistributed.
- `LICENSE_AND_DATA_USE.md` — model/data/software redistribution boundaries.

## Submission files

- `main_ne.tex` / `main_ne.pdf` — manuscript and compiled PDF.
- `highlights.txt` — four Elsevier highlights, each below 85 characters.
- `cover_letter.txt` — Neurocomputing cover letter.
- `submission_metadata.txt` — paste-ready verified author and affiliation
  metadata, with unknown system-only fields explicitly left for author input.
- `graphical_abstract.pdf` / `.png` — graphical abstract.
- Primary figures: `fig_concept`, `fig_contam`, `fig_amplify`,
  `fig_support_evidence` (PDF/SVG + generators).
- `declarations/*.txt` — Editorial Manager paste-ready declarations.

Verified e-mails are `lishikai@wchscu.edu.cn` for Shikai Li and
`dr.shirui@hotmail.com` for corresponding author Rui Shi. Add street/postcode
fields only if the submission system requires them and the authors confirm them.
