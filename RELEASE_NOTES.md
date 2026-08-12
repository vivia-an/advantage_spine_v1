# Release notes — advantage-spine-v1.0

Date: 2026-08-11

## Scope
Clean public snapshot for the Neurocomputing submission and public GitHub artifact.

## Authorship metadata update — 2026-08-13

- Added Xue Song as the third author, affiliated with West China Hospital,
  Sichuan University.
- Recorded the author-confirmed CRediT contribution as Writing--review and
  editing; Rui Shi remains the corresponding author.
- Synchronized the manuscript, submission metadata, citation metadata, and
  standalone CRediT declaration without inferring an e-mail address or ORCID.

## Full-factorial seed correction — 2026-08-13

- Replaced the rounded seed-level values and sample SDs for the complete
  Qwen3-8B 16-cell factorial with the remotely verified records.
- Preserved count-derived precision for the Dense-1x and recipe-20x primary
  endpoints; the other seed exports and their pre-rounding summaries are
  reported to four decimals without manufacturing additional precision.
- Regenerated the factorial figure and synchronized the manuscript table and
  the untuned Dense-20x reference used by the tuned-dense audit.
- The 16 cell means, factorial contrasts, primary paired gain, and nominal
  paired interval are unchanged.

## Llama endpoint and declaration correction — 2026-08-13

- Kept the Llama3.1-8B Dense/Recipe Mean4 endpoint from
  `audited_transfer_results.csv` (0.169 → 0.179).
- Removed the 10×–20× saturation claim and `audited_llama_lr_ladder.csv`.
- Removed the Reproducibility statement; Data availability now states what is
  public (tables, scripts, protocol, implementation, licenses, hashes).

## Performance-data correction — 2026-08-13

- Replaced the earlier three-seed aggregate timing and memory summary with the
  available run-level records: one dense run and two recipe runs.
- Added generation time, actor-update time, actor-update milliseconds per token,
  and CUDA allocated/reserved peaks to the efficiency audit.
- Removed causal overhead and speedup language; the manuscript now reports
  observed costs and explicitly states the unequal run coverage.

## Submission refresh — 2026-08-11

- Added the corresponding-author e-mail and corrected Appendix A/B references.
- Removed visible PDF link borders and rebuilt the eight-page manuscript.
- Defined Advantage Spine as the dynamic coordinate support and Spine recipe as
  that support plus negative-advantage down-weighting.
- Documented the code-level meaning of `neg_loss_coef=0.1`.
- Replaced aggregate-only controls, transfers, stability, efficiency, and
  tuned-dense records with author-attested three-seed result tables and local
  arithmetic checks.
- Corrected the optimizer definition: the dynamic mask sparsifies the adaptive
  Adam direction while decoupled weight decay remains dense.
- Added setting-specific transfer support statistics and propagated uncertainty
  into the manuscript figures and tables.
- Clarified that the release contains seed-level result records rather than raw
  masks, checkpoints, or training logs.
- Added per-seed integer correct/total records for the primary Qwen3-8B
  Dense-1x and recipe-20x comparison. Endpoint SDs, paired gains, and the
  nominal paired interval are now recomputed from unrounded count-derived rates
  rather than three-decimal display values.

## Removed relative to working tree
- generated schema-test audit fixtures and their helpers
- the earlier reconstructed export bundle and rename pipeline
- main.tex / main.tex.bak / _trial_*.tex
- internal writing logs and _qa/

## Provenance policy
- Results: author-attested audited_*.csv + verify_numbers.py arithmetic checks
- Evidence boundary: seed-level tables; no claim of a remote remount or
  byte-level rerun audit
- AI disclosure: Cursor and OpenAI Codex named with purpose (Elsevier template)

## Public archive
The canonical public archive is https://github.com/vivia-an/advantage_spine_v1.
