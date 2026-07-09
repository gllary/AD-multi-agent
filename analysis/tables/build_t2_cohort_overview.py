"""Build Table 2 — Cohort overview.

One row per cohort with: role (training vs zero-shot external), site, n,
prevalence, label SOP, and a short note on intended role in the study.

Outputs:
    paper_figures/tables/T2_cohort_overview.csv
    paper_figures/tables/T2_cohort_overview.md
    paper_figures/docs/T2_cohort_overview.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("AD_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "analysis" / "figures"))
from _data import COHORT_META, COHORT_ORDER, load_predictions  # noqa: E402

OUT = ROOT / "paper_figures" / "tables"
DOCS = ROOT / "paper_figures" / "docs"
OUT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


COHORT_DISPLAY = {
    "cohort_D":      "D",
    "cohort_V1":   "V1",
    "cohort_V2": "V2",
}

COHORT_FULL = {
    "cohort_D":      "Development cohort",
    "cohort_V1":   "External validation 1",
    "cohort_V2": "External validation 2",
}

COHORT_NOTE = {
    "cohort_D":      "Model development — all LightGBM stage risk models, "
                     "the CP3 logistic-regression text model, policy thresholds, "
                     "safety-layer hyperparameters, and the specialist and "
                     "coordinator LLM prompts were trained or iterated on this "
                     "cohort and then frozen before any external-cohort exposure.",
    "cohort_V1":   "Strict zero-shot external validation, same province as the "
                     "training cohort.",
    "cohort_V2": "Strict zero-shot post-suspicion suspected-AD work-up cohort "
                 "used for the headline external analysis at scale.",
}


def main() -> None:
    rows: list[dict] = []
    for cohort in COHORT_ORDER:
        meta = COHORT_META[cohort]
        # Compute exact counts from the data
        pred = load_predictions(cohort, "multi_agent")
        n = len(pred)
        pos = int((pred.y == 1).sum())
        neg = int((pred.y == 0).sum())
        prev = pos / n
        rows.append({
            "cohort_internal": cohort,
            "cohort_id":      COHORT_DISPLAY[cohort],
            "full_name":      COHORT_FULL[cohort],
            "role":           meta["role"],
            "site":           meta["site"],
            "site_label":     meta["site_label"],
            "n":              n,
            "n_AD_pos":       pos,
            "n_AD_neg":       neg,
            "prevalence":     prev,
            "label_sop":      meta["label_sop"],
            "study_role_note": COHORT_NOTE[cohort],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "T2_cohort_overview.csv", index=False)

    # ---- Markdown table ----
    md: list[str] = []
    md.append("# Table 2 — Cohort overview\n\n")
    md.append("One development cohort plus two independent zero-shot external "
              "validation cohorts. The external cohorts were not consulted "
              "during model training, prompt iteration, threshold calibration, "
              "or hyperparameter selection.\n\n")
    md.append("| ID | Cohort name | Role | Site | n | AD+ | AD− | AD prevalence | Label generation | Notes |\n")
    md.append("|---|---|---|---|---:|---:|---:|---:|---|---|\n")
    for _, r in df.iterrows():
        md.append(
            f"| **{r['cohort_id']}** | {r['full_name']} | {r['role']} | {r['site']} | "
            f"{int(r['n']):,} | {int(r['n_AD_pos']):,} | "
            f"{int(r['n_AD_neg']):,} | {r['prevalence']*100:.1f}% | "
            f"{r['label_sop']} | {r['study_role_note']} |\n"
        )
    md.append("\n")
    md.append("*AD, acute aortic dissection. The retained CSV schema uses `AD` as "
              "the binary reference-label column.*\n")

    (OUT / "T2_cohort_overview.md").write_text("".join(md))

    # ---- Doc ----
    doc = [
        "# T2 — Cohort overview | explanation document\n\n",
        "## Purpose\n",
        "Establishes the current three-cohort study design with a clear development vs. "
        "zero-shot boundary. This is the first table the reader meets and "
        "must communicate three things at a glance: (i) Cohort D alone was "
        "used for all model fitting and prompt iteration; (ii) Cohorts V1 and "
        "V2 are zero-shot external evaluations; (iii) Cohort V2 is the "
        "post-suspicion suspected-AD work-up cohort used for the headline "
        "external analysis.\n\n",
        "## Recommended placement\n",
        "Manuscript Methods §2 or Results §1 — must precede Table 1.\n\n",
        "## Reading the table\n",
        "- *Role* column makes the training vs. zero-shot boundary explicit.\n",
        "- *Site* column shows the three-centre design.\n",
        "- *Label generation* column distinguishes the three label SOPs.\n",
        "- *Notes* column states what each cohort tests in plain English.\n\n",
        "## Sources\n",
        "- Counts derived directly from per-patient prediction frames via "
        "`paper_figures._data.load_predictions(cohort, 'multi_agent')`.\n",
        "- Site, role and SOP metadata in `paper_figures._data.COHORT_META`.\n",
        "Generated by `paper_figures/tables/build_t2_cohort_overview.py`.\n",
    ]
    (DOCS / "T2_cohort_overview.md").write_text("".join(doc))
    print(f"Wrote {OUT/'T2_cohort_overview.csv'}")
    print(f"Wrote {OUT/'T2_cohort_overview.md'}")
    print(f"Wrote {DOCS/'T2_cohort_overview.md'}")


if __name__ == "__main__":
    main()
