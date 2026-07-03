#!/usr/bin/env python3
"""
Benchmark SynthID detection and attack modes on bundled sample texts.
Outputs results/benchmark.json and prints a markdown table for README.

Usage:
  python scripts/eval_benchmark.py --detect-only   # CI-safe: detection only
  python scripts/eval_benchmark.py                   # + non-API attacks; API modes if GOOGLE_API_KEY set
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ENABLE_PERPLEXITY", "false")

from app.pipeline import AttackPipeline
from app.schemas import WatermarkRequest
from reverse_synthid.extended_detector import ExtendedDetector

SAMPLE_DIR = ROOT / "reverse_synthid"
RESULTS_DIR = ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark.json"

INLINE_SAMPLES = [
    {
        "name": "unicode_hidden",
        "text": "Hello\u200b World\uFEFF! SynthID logit biasing enables detection with high confidence.",
    },
    {
        "name": "gemini_style",
        "text": (
            "Harry Kane has established himself as one of the most prolific strikers in modern football. "
            "His movement off the ball, clinical finishing, and leadership qualities make him a constant "
            "threat in the penalty area. Analysts note his ability to link play and create chances for teammates."
        ),
    },
    {
        "name": "short_ai",
        "text": (
            "Artificial intelligence systems generate text by predicting the next token in a sequence. "
            "Watermarks can be embedded during this sampling process."
        ),
    },
]

NO_API_MODES = ["perturb", "homoglyph", "sanitize_perturb"]
API_MODES = ["paraphrase", "backtranslate", "combined", "auto"]
ALL_ATTACK_MODES = NO_API_MODES + API_MODES


def load_samples() -> list[dict]:
    samples = []
    for path, name in [
        (SAMPLE_DIR / "watermarked.txt", "watermarked"),
        (SAMPLE_DIR / "clean.txt", "clean"),
    ]:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                samples.append({"name": name, "text": text})
    samples.extend(INLINE_SAMPLES)
    return samples


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {
            "samples": 0,
            "avg_g_pre": 0.0,
            "avg_g_post": 0.0,
            "avg_drop_pct": 0.0,
            "success_rate": 0.0,
        }
    n = len(rows)
    avg_pre = sum(r["g_pre"] for r in rows) / n
    avg_post = sum(r["g_post"] for r in rows) / n
    avg_drop = sum(r["drop_pct"] for r in rows) / n
    success = sum(1 for r in rows if r["g_post"] < 0.55) / n
    return {
        "samples": n,
        "avg_g_pre": round(avg_pre, 4),
        "avg_g_post": round(avg_post, 4),
        "avg_drop_pct": round(avg_drop, 2),
        "success_rate": round(success * 100, 1),
    }


def run_detection_benchmark(detector: ExtendedDetector, samples: list[dict]) -> list[dict]:
    rows = []
    for sample in samples:
        res = detector.detect(sample["text"])
        rows.append({
            "sample": sample["name"],
            "g_value": round(res["g_value"], 4),
            "is_watermarked": res["is_watermarked"],
            "perplexity": round(res.get("perplexity") or 0.0, 2),
            "tokens": detector.count_tokens(sample["text"]),
        })
    return rows


def run_attack_benchmark(
    pipeline: AttackPipeline,
    samples: list[dict],
    modes: list[str],
) -> dict[str, dict]:
    summary = {}
    for mode in modes:
        rows = []
        for sample in samples:
            try:
                resp = pipeline.run(WatermarkRequest(text=sample["text"], attack_mode=mode))
                pre_g = resp.pre_attack.g_value
                post_g = resp.post_attack.g_value
                drop = resp.watermark_reduction_pct
                rows.append({
                    "sample": sample["name"],
                    "g_pre": pre_g,
                    "g_post": post_g,
                    "drop_pct": drop,
                })
            except Exception as e:
                print(f"  [{mode}] {sample['name']}: skipped ({e})", file=sys.stderr)
        summary[mode] = aggregate(rows)
    return summary


def markdown_table(detection_rows: list[dict], attack_summary: dict[str, dict]) -> str:
    lines = [
        "| Sample | G-value | Watermarked | Tokens | Perplexity |",
        "|---|---:|---|---:|---:|",
    ]
    for r in detection_rows:
        wm = "yes" if r["is_watermarked"] else "no"
        ppl = r["perplexity"] if r["perplexity"] else "—"
        lines.append(
            f"| {r['sample']} | {r['g_value']:.4f} | {wm} | {r['tokens']} | {ppl} |"
        )

    if attack_summary:
        lines.extend([
            "",
            "| Attack mode | Samples | Avg G (pre) | Avg G (post) | Avg drop % | Success rate (G_post < 0.55) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for mode, agg in attack_summary.items():
            lines.append(
                f"| {mode} | {agg['samples']} | {agg['avg_g_pre']:.4f} | "
                f"{agg['avg_g_post']:.4f} | {agg['avg_drop_pct']:.1f}% | {agg['success_rate']:.1f}% |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark detection and attack modes")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Run detection benchmark only (no attack pipeline, CI-safe)",
    )
    args = parser.parse_args()

    samples = load_samples()
    if not samples:
        print("No samples found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} sample texts.")
    detector = ExtendedDetector()

    detection_rows = run_detection_benchmark(detector, samples)
    attack_summary = {}

    if not args.detect_only:
        pipeline = AttackPipeline()
        modes = list(NO_API_MODES)
        if os.getenv("GOOGLE_API_KEY"):
            modes.extend(API_MODES)
            print("GOOGLE_API_KEY set — including API attack modes.")
        else:
            print("No GOOGLE_API_KEY — running non-API attack modes only.")
        attack_summary = run_attack_benchmark(pipeline, samples, modes)

    output = {
        "detection": detection_rows,
        "attacks": attack_summary,
        "sample_count": len(samples),
        "detect_only": args.detect_only,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_FILE}\n")
    print(markdown_table(detection_rows, attack_summary))


if __name__ == "__main__":
    main()
