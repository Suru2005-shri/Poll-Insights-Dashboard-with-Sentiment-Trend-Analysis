"""
analysis.py
-----------
Loads, cleans, and analyses the poll dataset.
Returns clean DataFrames and insight dictionaries used by the visualizer.
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(filepath="data/poll_data.csv"):
    df = pd.read_csv(filepath)

    # Parse date
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # Drop full duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"🧹 Removed {before - after} duplicate rows.")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # Handle missing values (none expected in synthetic data, but good practice)
    df.dropna(subset=["preferred_product", "region", "age_group"], inplace=True)

    print(f"✅ Clean dataset: {len(df)} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def overall_product_votes(df):
    """Counts and % share for each product."""
    counts = df["preferred_product"].value_counts().reset_index()
    counts.columns = ["product", "votes"]
    counts["percentage"] = (counts["votes"] / counts["votes"].sum() * 100).round(2)
    counts["rank"] = range(1, len(counts) + 1)
    return counts


def region_wise_analysis(df):
    """Pivot: region × product, with row-% (share within each region)."""
    pivot = pd.crosstab(df["region"], df["preferred_product"])
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(2)
    return pivot, pivot_pct


def age_wise_analysis(df):
    """Pivot: age_group × product (% within age group)."""
    pivot = pd.crosstab(df["age_group"], df["preferred_product"])
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(2)
    return pivot, pivot_pct


def gender_wise_analysis(df):
    pivot = pd.crosstab(df["gender"], df["preferred_product"])
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(2)
    return pivot, pivot_pct


def satisfaction_analysis(df):
    order = ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"]
    counts = df["satisfaction"].value_counts().reindex(order).reset_index()
    counts.columns = ["satisfaction", "count"]
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    return counts


def nps_analysis(df):
    counts = df["nps_category"].value_counts().reset_index()
    counts.columns = ["nps", "count"]
    promoters  = counts.loc[counts.nps == "Promoter",  "count"].sum()
    detractors = counts.loc[counts.nps == "Detractor", "count"].sum()
    total      = counts["count"].sum()
    nps_score  = round((promoters - detractors) / total * 100, 1)
    return counts, nps_score


def monthly_trend(df):
    """Votes per product per month."""
    trend = df.groupby(["month", "preferred_product"]).size().unstack(fill_value=0)
    return trend


def avg_rating_by_product(df):
    return df.groupby("preferred_product")["rating"].mean().round(2).reset_index()


def would_buy_analysis(df):
    pivot = pd.crosstab(df["preferred_product"], df["would_buy_again"])
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(2)
    return pivot_pct


# ══════════════════════════════════════════════════════════════════════════════
# 3. INSIGHTS GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_insights(df):
    votes        = overall_product_votes(df)
    _, sat       = region_wise_analysis(df)
    _, nps_score = nps_analysis(df)
    ratings      = avg_rating_by_product(df)

    winner      = votes.iloc[0]["product"]
    winner_pct  = votes.iloc[0]["percentage"]
    runner_up   = votes.iloc[1]["product"]
    lowest      = votes.iloc[-1]["product"]
    best_region = sat[winner].idxmax()
    avg_rating  = df["rating"].mean().round(2)
    top_rated   = ratings.iloc[ratings["rating"].idxmax()]["preferred_product"]

    insights = [
        f"🏆 {winner} leads overall with {winner_pct}% of all votes.",
        f"🥈 {runner_up} is a close runner-up — watch this space.",
        f"📉 {lowest} has the lowest preference — needs attention.",
        f"🌍 {winner} is most popular in the {best_region} region.",
        f"⭐ Average respondent rating across all products: {avg_rating}/10.",
        f"🌟 Highest-rated product by average score: {top_rated}.",
        f"📊 Net Promoter Score (NPS): {nps_score} — {'Good' if nps_score > 30 else 'Needs Improvement'}.",
        f"👥 Survey covered {len(df)} respondents across {df['region'].nunique()} regions.",
    ]
    return insights


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_and_clean()
    print("\n--- Overall Votes ---")
    print(overall_product_votes(df))
    print("\n--- Insights ---")
    for i in generate_insights(df):
        print(i)
