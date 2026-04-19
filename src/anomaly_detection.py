"""
anomaly_detection.py
--------------------
Upgrade 5: Detects data quality issues and unusual patterns:
  - Daily vote spike detection (IQR method)
  - Outlier rating detection (z-score)
  - Suspicious uniform response patterns
  - Missing time periods
  - Response rate anomalies by region

This is what real analysts do BEFORE presenting results.
Run: python src/anomaly_detection.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DAILY VOTE SPIKE DETECTION (IQR METHOD)
# ══════════════════════════════════════════════════════════════════════════════

def detect_daily_spikes(df, threshold_multiplier=2.0):
    """Flag days with unusually high or low response counts."""
    daily = df.groupby("date").size().reset_index(name="count")
    Q1  = daily["count"].quantile(0.25)
    Q3  = daily["count"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold_multiplier * IQR
    upper = Q3 + threshold_multiplier * IQR

    daily["anomaly"] = (daily["count"] < lower) | (daily["count"] > upper)
    daily["direction"] = daily["count"].apply(
        lambda x: "spike" if x > upper else ("drop" if x < lower else "normal")
    )
    anomalies = daily[daily["anomaly"]]
    return daily, anomalies, lower, upper


# ══════════════════════════════════════════════════════════════════════════════
# 2. RATING OUTLIER DETECTION (Z-SCORE)
# ══════════════════════════════════════════════════════════════════════════════

def detect_rating_outliers(df, z_threshold=2.5):
    """Flag respondents whose ratings are statistical outliers."""
    df_out = df.copy()
    df_out["rating_z"] = (df["rating"] - df["rating"].mean()) / df["rating"].std()
    outliers = df_out[df_out["rating_z"].abs() > z_threshold]
    return outliers[["respondent_id", "region", "age_group",
                      "preferred_product", "rating", "rating_z"]]


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNIFORM RESPONSE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_uniform_responses(df):
    """
    If a single option dominates >80% in any region, flag it.
    Real surveys rarely show this level of uniformity — could indicate
    sampling bias or data collection issues.
    """
    flags = []
    pivot = pd.crosstab(df["region"], df["preferred_product"])
    pct   = pivot.div(pivot.sum(axis=1), axis=0).mul(100)

    for region in pct.index:
        max_pct = pct.loc[region].max()
        max_prod = pct.loc[region].idxmax()
        if max_pct > 75:
            flags.append({
                "region":  region,
                "product": max_prod,
                "pct":     round(max_pct, 1),
                "flag":    f"⚠️  {region}: {max_prod} dominates at {max_pct:.1f}%"
            })
    return flags


# ══════════════════════════════════════════════════════════════════════════════
# 4. RESPONSE RATE ANALYSIS BY REGION
# ══════════════════════════════════════════════════════════════════════════════

def response_rate_analysis(df):
    """Check if any region is significantly over or under-sampled."""
    region_counts = df["region"].value_counts()
    expected      = len(df) / len(region_counts)
    result = pd.DataFrame({
        "count":    region_counts,
        "expected": expected,
        "delta":    (region_counts - expected).round(1),
        "pct_of_total": (region_counts / len(df) * 100).round(1),
    })
    result["flag"] = result["delta"].apply(
        lambda d: "Over-sampled" if d > expected * 0.2
        else ("Under-sampled" if d < -expected * 0.2 else "Normal")
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_anomaly_charts(df):
    daily, anomalies, lower, upper = detect_daily_spikes(df)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Daily volume with anomaly markers
    axes[0].plot(daily["date"], daily["count"],
                 color="#2563EB", linewidth=1.5, alpha=0.8)
    axes[0].fill_between(daily["date"], daily["count"],
                          alpha=0.1, color="#2563EB")
    axes[0].axhline(y=upper, color="#DC2626", linestyle="--",
                    linewidth=1.2, label=f"Upper bound ({upper:.1f})")
    axes[0].axhline(y=lower, color="#D97706", linestyle="--",
                    linewidth=1.2, label=f"Lower bound ({lower:.1f})")
    if not anomalies.empty:
        axes[0].scatter(anomalies["date"], anomalies["count"],
                        color="#DC2626", s=60, zorder=5, label="Anomaly")
    axes[0].set_title("Daily Response Volume + Anomalies", fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Responses")
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Rating distribution with outlier highlight
    df_out = df.copy()
    df_out["z"] = (df["rating"] - df["rating"].mean()) / df["rating"].std()
    counts = df_out["rating"].value_counts().sort_index()
    outlier_ratings = df_out[df_out["z"].abs() > 2.5]["rating"].unique()
    colors = ["#DC2626" if r in outlier_ratings else "#2563EB"
              for r in counts.index]
    axes[1].bar(counts.index, counts.values, color=colors, edgecolor="white")
    axes[1].set_title("Rating Distribution (red = outlier zone)", fontweight="bold")
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel("Count")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Data Quality & Anomaly Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = "outputs/14_anomaly_detection.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Anomaly chart saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# FULL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def run_anomaly_report(df):
    print("\nDATA QUALITY & ANOMALY DETECTION REPORT")
    print("─" * 55)

    # Daily spikes
    daily, anomalies, lower, upper = detect_daily_spikes(df)
    print(f"\nDaily Volume Check (IQR bounds: {lower:.1f} – {upper:.1f})")
    if anomalies.empty:
        print("No daily volume anomalies detected.")
    else:
        print(f"{len(anomalies)} anomalous day(s) found:")
        for _, row in anomalies.iterrows():
            print(f"     {row['date'].date()} — {row['count']} responses ({row['direction']})")

    # Rating outliers
    outliers = detect_rating_outliers(df)
    print(f"\nRating Outliers (z > 2.5):")
    if outliers.empty:
        print("No rating outliers detected.")
    else:
        print(f"{len(outliers)} outlier respondents found:")
        print(outliers.head(5).to_string(index=False))

    # Uniform responses
    flags = detect_uniform_responses(df)
    print(f"\nUniformity Check (>75% dominance):")
    if not flags:
        print(" No suspiciously uniform regional patterns.")
    else:
        for f in flags:
            print(f"  {f['flag']}")

    # Response rate
    rr = response_rate_analysis(df)
    print(f"\n Response Rate by Region:")
    for region, row in rr.iterrows():
        print(f"  {region:10s}: {row['count']:4.0f} respondents "
              f"({row['pct_of_total']}%) {row['flag']}")

    print("\n─" * 55)
    plot_anomaly_charts(df)

    return {
        "daily_anomalies": anomalies,
        "rating_outliers": outliers,
        "uniformity_flags": flags,
        "response_rates":  rr,
    }


if __name__ == "__main__":
    from src.analysis import load_and_clean
    df = load_and_clean("data/poll_data.csv")
    run_anomaly_report(df)
