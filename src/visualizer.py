"""
visualizer.py
-------------
Generates and saves all charts for the Poll Results Visualizer.
Run standalone or call generate_all_charts(df) from main.py.
"""

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

from src.analysis import (
    load_and_clean, overall_product_votes, region_wise_analysis,
    age_wise_analysis, gender_wise_analysis, satisfaction_analysis,
    nps_analysis, monthly_trend, avg_rating_by_product,
    would_buy_analysis, generate_insights
)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PALETTE   = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Overall Vote Share (Horizontal Bar)
# ══════════════════════════════════════════════════════════════════════════════
def chart_overall_votes(df):
    votes = overall_product_votes(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(votes["product"][::-1], votes["percentage"][::-1],
                   color=PALETTE[::-1], edgecolor="white", height=0.55)
    for bar, pct in zip(bars, votes["percentage"][::-1]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{pct}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Vote Share (%)", fontsize=11)
    ax.set_title("Overall Product Preference — Vote Share", fontsize=14, fontweight="bold")
    ax.set_xlim(0, votes["percentage"].max() + 8)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "01_overall_votes.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Pie Chart of Vote Distribution
# ══════════════════════════════════════════════════════════════════════════════
def chart_vote_pie(df):
    votes = overall_product_votes(df)
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        votes["votes"], labels=votes["product"],
        autopct="%1.1f%%", colors=PALETTE,
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.set_title("Vote Distribution — Pie Chart", fontsize=14, fontweight="bold")
    return _save(fig, "02_vote_pie.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Stacked Bar: Region-wise Preference
# ══════════════════════════════════════════════════════════════════════════════
def chart_region_stacked(df):
    _, pct = region_wise_analysis(df)
    fig, ax = plt.subplots(figsize=(11, 6))
    pct.plot(kind="bar", stacked=True, ax=ax, color=PALETTE, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Region", fontsize=11)
    ax.set_ylabel("Vote Share (%)", fontsize=11)
    ax.set_title("Region-wise Product Preference (Stacked %)", fontsize=14, fontweight="bold")
    ax.legend(title="Product", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "03_region_stacked.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Heatmap: Region × Product (%)
# ══════════════════════════════════════════════════════════════════════════════
def chart_region_heatmap(df):
    _, pct = region_wise_analysis(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Vote Share (%)"})
    ax.set_title("Region × Product Preference Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Product")
    ax.set_ylabel("Region")
    return _save(fig, "04_region_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Age Group Analysis (Grouped Bar)
# ══════════════════════════════════════════════════════════════════════════════
def chart_age_grouped(df):
    _, pct = age_wise_analysis(df)
    fig, ax = plt.subplots(figsize=(12, 6))
    pct.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="white", width=0.75)
    ax.set_xlabel("Age Group", fontsize=11)
    ax.set_ylabel("Vote Share (%)", fontsize=11)
    ax.set_title("Age Group-wise Product Preference", fontsize=14, fontweight="bold")
    ax.legend(title="Product", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "05_age_grouped.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Satisfaction Distribution
# ══════════════════════════════════════════════════════════════════════════════
def chart_satisfaction(df):
    sat = satisfaction_analysis(df)
    colors = ["#16A34A", "#84CC16", "#EAB308", "#F97316", "#DC2626"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(sat["satisfaction"], sat["percentage"], color=colors, edgecolor="white", width=0.55)
    for bar, pct in zip(bars, sat["percentage"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{pct}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Respondents (%)", fontsize=11)
    ax.set_title("Customer Satisfaction Distribution", fontsize=14, fontweight="bold")
    ax.set_xticklabels(sat["satisfaction"], rotation=15, ha="right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "06_satisfaction.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — NPS Donut
# ══════════════════════════════════════════════════════════════════════════════
def chart_nps_donut(df):
    counts, nps_score = nps_analysis(df)
    nps_colors = {"Promoter": "#16A34A", "Passive": "#EAB308", "Detractor": "#DC2626"}
    colors = [nps_colors.get(n, "#999") for n in counts["nps"]]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax.pie(
        counts["count"], labels=counts["nps"],
        autopct="%1.1f%%", colors=colors,
        startangle=90, pctdistance=0.80,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.text(0, 0, f"NPS\n{nps_score}", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#1E293B")
    ax.set_title("Net Promoter Score (NPS) — Donut Chart", fontsize=14, fontweight="bold")
    return _save(fig, "07_nps_donut.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Monthly Trend Line
# ══════════════════════════════════════════════════════════════════════════════
def chart_monthly_trend(df):
    trend = monthly_trend(df)
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, col in enumerate(trend.columns):
        ax.plot(trend.index, trend[col], marker="o", linewidth=2.2,
                color=PALETTE[i % len(PALETTE)], label=col)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Number of Votes", fontsize=11)
    ax.set_title("Monthly Voting Trend by Product", fontsize=14, fontweight="bold")
    ax.legend(title="Product", fontsize=9)
    ax.set_xticklabels(trend.index, rotation=30, ha="right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "08_monthly_trend.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 9 — Average Rating by Product
# ══════════════════════════════════════════════════════════════════════════════
def chart_avg_rating(df):
    ratings = avg_rating_by_product(df).sort_values("rating", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(ratings["preferred_product"], ratings["rating"],
                  color=PALETTE, edgecolor="white", width=0.55)
    ax.axhline(y=df["rating"].mean(), color="#64748B", linestyle="--", linewidth=1.4, label="Overall Avg")
    for bar, val in zip(bars, ratings["rating"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Rating (out of 10)", fontsize=11)
    ax.set_title("Average Rating by Product", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "09_avg_rating.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 10 — Would Buy Again (Stacked %)
# ══════════════════════════════════════════════════════════════════════════════
def chart_would_buy(df):
    pct = would_buy_analysis(df)
    buy_colors = {"Yes": "#16A34A", "Maybe": "#EAB308", "No": "#DC2626"}
    colors = [buy_colors.get(c, "#999") for c in pct.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    pct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Respondents (%)", fontsize=11)
    ax.set_xlabel("Product", fontsize=11)
    ax.set_title("Would Buy Again? — by Product", fontsize=14, fontweight="bold")
    ax.legend(title="Response", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "10_would_buy_again.png")


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def generate_all_charts(df):
    print("\nGenerating charts...")
    paths = [
        chart_overall_votes(df),
        chart_vote_pie(df),
        chart_region_stacked(df),
        chart_region_heatmap(df),
        chart_age_grouped(df),
        chart_satisfaction(df),
        chart_nps_donut(df),
        chart_monthly_trend(df),
        chart_avg_rating(df),
        chart_would_buy(df),
    ]
    print(f"\nAll {len(paths)} charts saved to /{OUT_DIR}/")
    return paths


if __name__ == "__main__":
    df = load_and_clean()
    generate_all_charts(df)
