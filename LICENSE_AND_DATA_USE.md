# Model, dataset, and software terms

This file records the upstream terms used for the Paper D experiments. It is
an attribution and redistribution boundary, not legal advice.

| Resource | Role | Upstream terms | Archival treatment |
|---|---|---|---|
| Qwen3-8B and Qwen3-1.7B | Main and scale-transfer models | Apache License 2.0 | Model identifiers and initialization hashes are released; weights remain under the upstream license. |
| Llama-3.1-8B | Architecture-transfer model | Llama 3.1 Community License and Acceptable Use Policy | No Meta weights are redistributed in the paper bundle. |
| DAPO-Math-17K | RLVR training prompts | Apache License 2.0 | Dataset identifier and file hash are released; users obtain the data from the upstream host. |
| MATH500 | Evaluation | The HuggingFaceH4 dataset card does not declare a license | Only identifiers, hashes, counts, and aggregate results are released; problem text is not redistributed here. |
| AIME 2024 | Evaluation | The HuggingFaceH4 dataset card does not declare a license | Only identifiers, hashes, counts, and aggregate results are released; problem text is not redistributed here. |
| AIME 2025 | Evaluation | Apache License 2.0 on the referenced `math-ai/aime25` distribution | Dataset identifier and hash are released; users obtain the data upstream. |
| OlympiadBench | Evaluation | MIT License in the official OpenBMB repository | Dataset identifier and hash are released; users obtain the data upstream. |
| VERL | Training framework | Apache License 2.0 | Repository revision is recorded in the environment manifest. |

Upstream locations used to verify these terms:

- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/Qwen/Qwen3-1.7B
- https://huggingface.co/meta-llama/Llama-3.1-8B
- https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
- https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- https://huggingface.co/datasets/HuggingFaceH4/aime_2024
- https://huggingface.co/datasets/math-ai/aime25
- https://github.com/OpenBMB/OlympiadBench
- https://github.com/volcengine/verl

The audit export does not change any upstream license. Checkpoints and model
derivatives may be shared only when the corresponding upstream model and data
terms permit redistribution.
