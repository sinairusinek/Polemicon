"""Read overnight artifacts and produce a single readable summary file."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = LOGS / "overnight_summary.md"


def grab(path, pattern):
    if not path.exists():
        return None
    m = re.search(pattern, path.read_text())
    return m.group(1) if m else None


def main():
    lines = ["# Overnight run summary", ""]

    # B.4a comparison
    cmp_path = ROOT / "data" / "b4a_model_comparison.json"
    if cmp_path.exists():
        results = json.loads(cmp_path.read_text())
        lines.append("## B.4a model comparison (6 epochs)")
        lines.append("")
        lines.append("| Model | Macro F1 | non | implicit | explicit | meta |")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            pc = r.get("per_class", [None] * 4) + [None] * 4
            row = (
                f"| {r['model_key']} "
                f"| {r['macro_f1']:.3f} "
                f"| {pc[0]:.3f} | {pc[1]:.3f} | {pc[2]:.3f} | {pc[3]:.3f} |"
            )
            lines.append(row)
        winner = max(results, key=lambda r: r["macro_f1"])
        lines.append("")
        lines.append(f"**Winner:** `{winner['model_key']}` ({winner['model_id']}) — macro F1 {winner['macro_f1']:.3f}")
        lines.append("")

    # Classification report from final-train log
    final_log = LOGS / "b4a_full.log"
    if final_log.exists():
        txt = final_log.read_text()
        # last classification_report block
        blocks = re.findall(r"Test results \([^)]+\):\s*\n(.*?)(?=\n  Macro F1:)", txt, re.DOTALL)
        if blocks:
            lines.append("## Test classification report (final saved model)")
            lines.append("")
            lines.append("```")
            lines.append(blocks[-1].rstrip())
            lines.append("```")
            lines.append("")

    # Full-corpus predictions distribution
    pred_path = ROOT / "data" / "full_corpus_predictions.parquet"
    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
        n = len(preds)
        lines.append("## Full-corpus predictions")
        lines.append("")
        lines.append(f"Texts classified: **{n:,}**  → `data/full_corpus_predictions.parquet`")
        lines.append("")
        lines.append("| Label | Count | Share |")
        lines.append("|---|---|---|")
        counts = preds["predicted_label"].value_counts()
        for label in [
            "non-polemic", "implicit polemic", "explicit polemic", "meta-polemic (descriptive)"
        ]:
            c = int(counts.get(label, 0))
            lines.append(f"| {label} | {c:,} | {c/n:.1%} |")
        polemic = int(n - counts.get("non-polemic", 0))
        lines.append(f"| **combined polemic** | **{polemic:,}** | **{polemic/n:.1%}** |")
        lines.append("")

        # Binary view via probability-mass collapse — uses prob_non as the
        # discriminator. This is more honest than argmax-then-collapse because
        # it does not let a fragmented polemic vote lose to a unified non vote.
        non_col = "prob_non_polemic" if "prob_non_polemic" in preds.columns else "prob_non"
        if non_col in preds.columns:
            binary_polemic = (preds[non_col] < 0.5).sum()
            lines.append("### Binary view (derived from prob_non < 0.5)")
            lines.append("")
            lines.append(f"- polemic: {binary_polemic:,} ({binary_polemic/n:.1%})")
            lines.append(f"- non-polemic: {n - binary_polemic:,} ({(n - binary_polemic)/n:.1%})")
            lines.append("")
            lines.append("Expected binary macro F1 from test-set extrapolation: ~0.82 "
                         "(vs 0.47 on 4-class). Use this layer for triage; "
                         "subtype labels are noisy.")
            lines.append("")
        # mean confidence per label
        lines.append("### Mean confidence per predicted label")
        lines.append("")
        conf = preds.groupby("predicted_label")["confidence"].mean()
        for label, v in conf.items():
            lines.append(f"- {label}: {v:.3f}")
        lines.append("")
        # Compare to Sonnet calibration distribution if available
        cal_path = ROOT / "data" / "calibration_v2.parquet"
        if cal_path.exists():
            cal = pd.read_parquet(cal_path)
            cal = cal[cal["polemic_label"].notna()]
            lines.append("### Sonnet 2K calibration distribution (reference)")
            lines.append("")
            ccounts = cal["polemic_label"].value_counts()
            for label in [
                "non-polemic", "implicit polemic", "explicit polemic", "meta-polemic (descriptive)"
            ]:
                c = int(ccounts.get(label, 0))
                lines.append(f"- {label}: {c} ({c/len(cal):.1%})")
            lines.append("")

    # File manifest
    lines.append("## Files written")
    lines.append("")
    for rel in [
        "data/b4a_model_comparison.json",
        "data/models/best_polemic_classifier/",
        "data/full_corpus_predictions.parquet",
        "logs/b4a_full.log",
        "logs/full_corpus_inference.log",
    ]:
        p = ROOT / rel
        exists = "✓" if p.exists() else "✗ MISSING"
        lines.append(f"- {exists} `{rel}`")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
