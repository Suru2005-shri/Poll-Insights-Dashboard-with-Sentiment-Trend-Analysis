"""
stats_testing.py
----------------
Upgrade 1: Statistical significance testing using chi-square tests.
Proves that differences between regions/age groups are REAL, not random noise.
This separates serious analysts from people who just make charts.

Run:
    python src/stats_testing.py
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from src.analysis import load_and_clean


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def interpret_p(p):
    if p < 0.001: return "*** Highly significant (p<0.001)"
    if p < 0.01:  return "**  Significant (p<0.01)"
    if p < 0.05:  return "*   Significant (p<0.05)"
    return "    Not significant (p≥0.05) — could be random"

def cramers_v(chi2, n, r, c):
    """Effect size for chi-square: 0=no effect, 1=perfect association."""
    return round(np.sqrt(chi2 / (n * (min(r, c) - 1))), 4)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Region vs Product preference
# ══════════════════════════════════════════════════════════════════════════════

def test_region_vs_product(df):
    """
    H0: Product preference is INDEPENDENT of region (no association).
    H1: Product preference DEPENDS on region.
    """
    ct = pd.crosstab(df["region"], df["preferred_product"])
    chi2, p, dof, expected = chi2_contingency(ct)
    v = cramers_v(chi2, len(df), *ct.shape)

    result = {
        "test":        "Chi-square: Region × Product",
        "chi2":        round(chi2, 4),
        "p_value":     round(p, 6),
        "dof":         dof,
        "cramers_v":   v,
        "effect":      "Small" if v < 0.1 else "Medium" if v < 0.3 else "Large",
        "conclusion":  interpret_p(p),
        "crosstab":    ct,
        "expected":    pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(1),
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Age group vs Product preference
# ══════════════════════════════════════════════════════════════════════════════

def test_age_vs_product(df):
    ct = pd.crosstab(df["age_group"], df["preferred_product"])
    chi2, p, dof, expected = chi2_contingency(ct)
    v = cramers_v(chi2, len(df), *ct.shape)

    return {
        "test":       "Chi-square: Age Group × Product",
        "chi2":       round(chi2, 4),
        "p_value":    round(p, 6),
        "dof":        dof,
        "cramers_v":  v,
        "effect":     "Small" if v < 0.1 else "Medium" if v < 0.3 else "Large",
        "conclusion": interpret_p(p),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Gender vs Product preference
# ══════════════════════════════════════════════════════════════════════════════

def test_gender_vs_product(df):
    ct = pd.crosstab(df["gender"], df["preferred_product"])
    chi2, p, dof, _ = chi2_contingency(ct)
    v = cramers_v(chi2, len(df), *ct.shape)

    return {
        "test":       "Chi-square: Gender × Product",
        "chi2":       round(chi2, 4),
        "p_value":    round(p, 6),
        "dof":        dof,
        "cramers_v":  v,
        "effect":     "Small" if v < 0.1 else "Medium" if v < 0.3 else "Large",
        "conclusion": interpret_p(p),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — ANOVA: Do ratings differ significantly across products?
# ══════════════════════════════════════════════════════════════════════════════

def test_rating_anova(df):
    """
    H0: Mean ratings are equal across all products.
    H1: At least one product has a different mean rating.
    One-way ANOVA.
    """
    groups = [df[df["preferred_product"] == p]["rating"].values
              for p in df["preferred_product"].unique()]
    f_stat, p = f_oneway(*groups)

    means = df.groupby("preferred_product")["rating"].mean().round(2)

    return {
        "test":       "One-way ANOVA: Rating across Products",
        "f_statistic": round(f_stat, 4),
        "p_value":    round(p, 6),
        "conclusion": interpret_p(p),
        "group_means": means,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PAIRWISE RESIDUALS — which region/product combos are over/under expected?
# ══════════════════════════════════════════════════════════════════════════════

def residual_analysis(df):
    """
    Standardised residuals > ±2 suggest a cell deviates significantly
    from what chi-square independence would predict.
    """
    ct = pd.crosstab(df["region"], df["preferred_product"])
    _, _, _, expected = chi2_contingency(ct)
    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)

    residuals = (ct - expected_df) / np.sqrt(expected_df)
    residuals = residuals.round(2)
    return residuals


# ══════════════════════════════════════════════════════════════════════════════
# FULL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def run_full_significance_report(df):
    sep = "─" * 60
    print("\n" + sep)
    print("  STATISTICAL SIGNIFICANCE REPORT")
    print(sep)

    tests = [
        test_region_vs_product(df),
        test_age_vs_product(df),
        test_gender_vs_product(df),
    ]

    for t in tests:
        print(f"\n{t['test']}")
        print(f"   χ² = {t['chi2']}  |  p = {t['p_value']}  |  dof = {t['dof']}")
        print(f"   Cramér's V = {t['cramers_v']} ({t['effect']} effect)")
        print(f"   → {t['conclusion']}")

    print(f"\nOne-way ANOVA — Ratings by Product")
    anova = test_rating_anova(df)
    print(f"   F = {anova['f_statistic']}  |  p = {anova['p_value']}")
    print(f"   → {anova['conclusion']}")
    print(f"   Mean ratings: {anova['group_means'].to_dict()}")

    print(f"\nStandardised Residuals (Region × Product)")
    print("   Cells > +2: over-represented  |  < -2: under-represented")
    residuals = residual_analysis(df)
    print(residuals.to_string())

    print(f"\n{sep}")
    print("  KEY STATISTICAL INSIGHTS")
    print(sep)
    region_test = tests[0]
    if region_test["p_value"] < 0.05:
        print(f" Regional preference differences are STATISTICALLY SIGNIFICANT")
        print(f"     (p={region_test['p_value']}, Cramér's V={region_test['cramers_v']})")
        print(f"     → Safe to report regional breakdowns as real patterns.")
    else:
        print(f" Regional differences are NOT significant — may be sampling noise.")

    age_test = tests[1]
    if age_test["p_value"] < 0.05:
        print(f"Age group differences are STATISTICALLY SIGNIFICANT")
        print(f"     → Demographic targeting is data-backed.")
    else:
        print(f"Age group differences are NOT significant.")

    print(sep + "\n")

    return {
        "region_test": tests[0],
        "age_test":    tests[1],
        "gender_test": tests[2],
        "anova":       anova,
        "residuals":   residuals,
    }


if __name__ == "__main__":
    df = load_and_clean("data/poll_data.csv")
    run_full_significance_report(df)
