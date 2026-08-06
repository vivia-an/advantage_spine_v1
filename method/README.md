# Advantage Spine method drop-in

Paper recipe on the primary Qwen3-8B factorial:

- Optimizer: `SparseAdamW`
- Coordinate mask: `sparse_mode=topk`, `sparse_topk_ratio=0.4`, `sparse_min_numel=4096`
- Negative-channel weight: `neg_loss_coef=0.1` (or equivalent GRPO negative-advantage coefficient)
- High-rate arm: learning-rate multiplier `20×` relative to dense `5e-6` (i.e. `1e-4`)
- No error-feedback accumulator (`topk`, not `topk_ef`)

## Install into VERL

```bash
cp -r method/verl_utils_optim/* path/to/verl/utils/optim/
```

Example Hydra-style overrides (exact key paths may differ by VERL fork):

```text
actor_rollout_ref.actor.optim.optimizer=SparseAdamW
actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.optim.sparse_adamw
actor_rollout_ref.actor.optim.sparse_mode=topk
actor_rollout_ref.actor.optim.sparse_topk_ratio=0.4
actor_rollout_ref.actor.optim.sparse_min_numel=4096
actor_rollout_ref.actor.optim.lr=1e-4
```

Set the negative-advantage coefficient to `0.1` in your GRPO actor config.
Audited tables and `verify_numbers.py` live at the repository root.
