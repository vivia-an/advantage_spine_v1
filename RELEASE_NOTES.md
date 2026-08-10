# Release notes — advantage-spine-v1.0

Date: 2026-08-06

## Scope
Clean public snapshot for the Neurocomputing submission and public GitHub artifact.

## Submission refresh — 2026-08-10

- Added the corresponding-author e-mail and corrected Appendix A/B references.
- Removed visible PDF link borders and rebuilt the eight-page manuscript.
- Defined Advantage Spine as the dynamic coordinate support and Spine recipe as
  that support plus negative-advantage down-weighting.
- Documented the code-level meaning of `neg_loss_coef=0.1`.
- Added exact settings and stability diagnostics for the exploratory tuned-dense
  aggregate arm.
- Clarified that the public audit contains verified mask records, not remote raw
  mask files.

## Removed relative to working tree
- synthetic_expected_audit/ (+ generator)
- reconstructed_audit/ (+ build/verify helpers)
- prepare_remote_audit.py rename pipeline
- main.tex / main.tex.bak / _trial_*.tex
- internal writing logs and _qa/

## Provenance policy
- Aggregates: audited_*.csv + verify_numbers.py
- Export: author-attested; local verify ≠ remote remount
- AI disclosure: Cursor and OpenAI Codex named with purpose (Elsevier template)

## Public archive
The canonical public archive is https://github.com/vivia-an/advantage_spine_v1.
