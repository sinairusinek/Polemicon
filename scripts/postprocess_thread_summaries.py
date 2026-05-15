"""
postprocess_thread_summaries.py — derive per-(thread,model) quality fields and
identify which threads need Opus arbitration before the 141-thread production run.

Inputs:
  data/thread_llm_summaries.parquet  (per-(thread,model) verdicts)
  data/threads.parquet               (cluster n_docs)

Outputs:
  data/thread_summaries_derived.parquet   per-(thread,model) + core_doc_count, thread_purity, effective_polemic_strength
  data/thread_arbitration_targets.parquet threads that need Opus arbitration with reason
  logs/arbitration_targets.md             human-readable report

Run:
  python scripts/postprocess_thread_summaries.py
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"

SCORE_GAP_TRIGGER = 0.25
OUTLIER_RATIO_TRIGGER = 2.0


def n_outliers(s: str) -> int:
    if not s or pd.isna(s):
        return 0
    try:
        return len(json.loads(s))
    except Exception:
        return 0


def main():
    llm = pd.read_parquet(DATA / "thread_llm_summaries.parquet")
    threads = pd.read_parquet(DATA / "threads.parquet")[["thread_id", "n_docs"]]

    df = llm.merge(threads, on="thread_id", how="left")
    df["n_outliers"] = df["outlier_docs"].apply(n_outliers)
    df["core_doc_count"] = (df["n_docs"] - df["n_outliers"]).clip(lower=0)
    df["thread_purity"] = (df["core_doc_count"] / df["n_docs"].clip(lower=1)).round(3)
    df["effective_polemic_strength"] = (
        df["polemic_score"].fillna(0.0) * df["core_doc_count"].apply(lambda c: math.sqrt(max(0, c)))
    ).round(3)

    derived_cols = [
        "thread_id", "model", "n_docs", "n_outliers", "core_doc_count", "thread_purity",
        "is_polemic_thread", "polemic_score", "polemic_direction", "polemic_type",
        "effective_polemic_strength",
    ]
    if "sub_thread_signal" in df.columns:
        derived_cols.append("sub_thread_signal")
    derived = df[derived_cols].sort_values(["thread_id", "model"])
    out_path = DATA / "thread_summaries_derived.parquet"
    derived.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(derived)} rows)")

    # --- Arbitration targets ---
    # Pivot per-thread across models
    pivot = df.pivot_table(
        index="thread_id",
        columns="model",
        values=["polemic_score", "polemic_direction", "is_polemic_thread", "n_outliers"],
        aggfunc="first",
    )
    models = sorted(df["model"].unique())
    cheap_models = [m for m in models if m != "cli_opus"]

    rows = []
    for tid in pivot.index:
        reasons = []
        # Pull per-cheap-model values
        scores = {m: pivot.loc[tid, ("polemic_score", m)] for m in cheap_models if ("polemic_score", m) in pivot.columns}
        scores = {m: v for m, v in scores.items() if pd.notna(v)}
        dirs = {m: pivot.loc[tid, ("polemic_direction", m)] for m in cheap_models if ("polemic_direction", m) in pivot.columns}
        dirs = {m: v for m, v in dirs.items() if pd.notna(v)}
        flags = {m: pivot.loc[tid, ("is_polemic_thread", m)] for m in cheap_models if ("is_polemic_thread", m) in pivot.columns}
        flags = {m: v for m, v in flags.items() if pd.notna(v)}
        outs = {m: pivot.loc[tid, ("n_outliers", m)] for m in cheap_models if ("n_outliers", m) in pivot.columns}
        outs = {m: v for m, v in outs.items() if pd.notna(v)}

        if len(scores) >= 2:
            gap = max(scores.values()) - min(scores.values())
            if gap > SCORE_GAP_TRIGGER:
                reasons.append(f"score_gap={gap:.2f}")
        if len(dirs) >= 2 and len(set(dirs.values())) > 1:
            reasons.append(f"direction_disagree={dict(dirs)}")
        if len(flags) >= 2 and len(set(bool(v) for v in flags.values())) > 1:
            reasons.append(f"is_polemic_disagree={dict(flags)}")
        if len(outs) >= 2:
            mx, mn = max(outs.values()), max(1.0, min(outs.values()))
            ratio = mx / mn
            if ratio >= OUTLIER_RATIO_TRIGGER and mx >= 4:
                reasons.append(f"outlier_ratio={ratio:.1f} ({dict(outs)})")
        already = pivot.columns.get_level_values(1).tolist().count("cli_opus") and \
                  pd.notna(pivot.loc[tid].get(("polemic_score", "cli_opus"), float("nan")))
        if reasons and not already:
            rows.append({
                "thread_id": tid,
                "reasons": "; ".join(reasons),
                "score_gap": (max(scores.values()) - min(scores.values())) if len(scores) >= 2 else 0.0,
                **{f"score_{m}": scores.get(m) for m in cheap_models},
                **{f"dir_{m}": dirs.get(m) for m in cheap_models},
                **{f"n_outliers_{m}": outs.get(m) for m in cheap_models},
            })

    targets = pd.DataFrame(rows).sort_values("score_gap", ascending=False) if rows else pd.DataFrame()
    t_path = DATA / "thread_arbitration_targets.parquet"
    targets.to_parquet(t_path, index=False)
    print(f"wrote {t_path} ({len(targets)} rows)")

    # Human-readable report
    LOGS.mkdir(exist_ok=True)
    lines = [
        "# Opus arbitration targets",
        "",
        f"Generated by `scripts/postprocess_thread_summaries.py`. Trigger rules: ",
        f"score_gap > {SCORE_GAP_TRIGGER}; direction disagreement; is_polemic_thread disagreement; ",
        f"outlier-count ratio ≥ {OUTLIER_RATIO_TRIGGER}× with max ≥ 4.",
        "",
        f"**{len(targets)} threads need Opus arbitration** (currently arbitrated: see thread_llm_summaries.parquet).",
        "",
    ]
    if len(targets):
        lines.append("| thread | reasons |")
        lines.append("|---|---|")
        for _, r in targets.iterrows():
            lines.append(f"| {int(r['thread_id'])} | {r['reasons']} |")
    (LOGS / "arbitration_targets.md").write_text("\n".join(lines))
    print(f"wrote {LOGS / 'arbitration_targets.md'}")

    # Console summary
    print("\n=== Derived (top 10 by effective_polemic_strength) ===")
    print(
        derived.sort_values("effective_polemic_strength", ascending=False).head(15)
        [["thread_id", "model", "n_docs", "n_outliers", "core_doc_count", "thread_purity",
          "polemic_score", "effective_polemic_strength"]]
        .to_string(index=False)
    )
    if len(targets):
        print("\n=== Arbitration targets ===")
        print(targets[["thread_id", "score_gap", "reasons"]].to_string(index=False))


if __name__ == "__main__":
    main()
