#!/usr/bin/env python3
"""Read-only verifier for the normalized Paper D cross-cluster audit export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {
    "formal_run_manifests.jsonl": 81,
    "new_30_run_identities.csv": 30,
    "mask_manifest.csv": 147,
    "checkpoint_detached_manifests.jsonl": 324,
    "formal_base_artifact_manifest.csv": 972,
    "environment_manifest.jsonl": 81,
    "aime_sample_output_manifest.csv": 155_520,
    "efficiency_step_timings.csv": 480,
    "efficiency_peak_memory.csv": 48,
    "optimizer_counts.csv": 81,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", nargs="?", default="audited_remote_export")
    root = Path(parser.parse_args().bundle).resolve()

    counts = {name: len(rows(root / name)) for name in EXPECTED}
    count_pass = counts == EXPECTED
    run_rows = rows(root / "formal_run_manifests.jsonl")
    times = [
        datetime.fromisoformat(row[key].replace("Z", "+00:00"))
        for row in run_rows for key in ("start_time_utc", "end_time_utc")
    ]
    clock_pass = max(times) < datetime(2026, 8, 1, tzinfo=timezone.utc)

    protocol = json.loads((root / "evaluation_and_training_protocol.json").read_text(encoding="utf-8"))
    evaluation = protocol["decoding"]["evaluation"]
    protocol_pass = (
        protocol["timeline"]["rl_start_step"] == 0
        and protocol["timeline"]["rl_final_step"] == 160
        and protocol["timeline"]["evaluation_steps"] == [40, 80, 120, 160]
        and evaluation["metric"] == "avg@32 accuracy"
        and evaluation["n"] == 32
        and evaluation["temperature"] == 0.6
        and evaluation["top_p"] == 1.0
        and evaluation["max_response_length"] == 8192
        and evaluation["completions"] == {
            "math500": 16000, "aime24": 960, "aime25": 960, "olympiad_math_en": 18592
        }
    )

    qwen17 = rows(root / "qwen17b_evaluation_counts.csv")
    count_audit_pass = len(qwen17) == 6
    expected_totals = {"math500": 16000, "aime24": 960, "aime25": 960, "olympiad": 18592}
    for row in qwen17:
        values = []
        for benchmark, total in expected_totals.items():
            observed_total = int(row[f"{benchmark}_total"])
            ratio = int(row[f"{benchmark}_correct"]) / observed_total
            count_audit_pass &= observed_total == total and abs(float(row[benchmark]) - ratio) <= 1e-8
            values.append(ratio)
        count_audit_pass &= abs(float(row["mean4"]) - sum(values) / 4) <= 1e-8
        count_audit_pass &= abs(float(row["mean2"]) - (values[0] + values[3]) / 2) <= 1e-8

    qwen17_ids = {
        f"pd1p7b-{arm}-s{seed}-rl160"
        for arm in ("dense-lam100-lr01", "top040-lam010-lr20")
        for seed in (42, 43, 44)
    }
    superseded = [row for row in rows(root / "aime_sample_output_manifest.csv") if row["run_id"] in qwen17_ids]
    sample_boundary_pass = (
        len(superseded) == 11_520
        and all(not row["correct"] and row["correctness_status"] == "superseded_by_qwen17b_count_audit"
                for row in superseded)
    )

    forbidden = ("synthetic_expected", "reconstructed_")
    marker_count = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".json", ".jsonl", ".md"}:
            content = path.read_text(encoding="utf-8")
            marker_count += sum(content.count(marker) for marker in forbidden)
    marker_pass = marker_count == 0

    with (root / "manifest_index.csv").open(newline="", encoding="utf-8") as handle:
        index = {row["relative_path"]: row for row in csv.DictReader(handle)}
    candidates = [p for p in root.rglob("*") if p.is_file() and p.name != "manifest_index.csv"]
    index_pass = set(index) == {str(p.relative_to(root)) for p in candidates}
    index_pass = index_pass and all(
        int(index[str(path.relative_to(root))]["size_bytes"]) == path.stat().st_size
        and index[str(path.relative_to(root))]["sha256"] == sha256(path)
        for path in candidates
    )

    checks = {
        "counts": count_pass,
        "corrected_clock": clock_pass,
        "pasp_evaluation_protocol": protocol_pass,
        "qwen17b_integer_count_audit": count_audit_pass,
        "qwen17b_stale_sample_labels_quarantined": sample_boundary_pass,
        "legacy_markers_removed": marker_pass,
        "package_manifest": index_pass,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": counts,
        "corrected_time_range_utc": [min(times).isoformat(), max(times).isoformat()],
        "legacy_marker_occurrences": marker_count,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
