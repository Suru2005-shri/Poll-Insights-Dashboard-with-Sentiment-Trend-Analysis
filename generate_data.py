"""
generate_data.py
----------------
Generates a realistic synthetic poll dataset for the Poll Results Visualizer.
Simulates a national product preference survey across regions, age groups, and dates.
"""

import pandas as pd
import numpy as np
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
N_RESPONDENTS = 1000

REGIONS     = ["North", "South", "East", "West", "Central"]
AGE_GROUPS  = ["18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS     = ["Male", "Female", "Non-binary"]
EDUCATION   = ["High School", "Bachelor's", "Master's", "PhD"]
PRODUCTS    = ["Product A", "Product B", "Product C", "Product D", "Product E"]

# Regional biases make data more realistic (higher weight = more preferred in that region)
REGIONAL_PREFS = {
    "North":   [0.35, 0.25, 0.20, 0.12, 0.08],
    "South":   [0.20, 0.35, 0.15, 0.20, 0.10],
    "East":    [0.15, 0.20, 0.40, 0.15, 0.10],
    "West":    [0.25, 0.15, 0.20, 0.10, 0.30],
    "Central": [0.20, 0.20, 0.20, 0.20, 0.20],
}

# Age-group biases
AGE_PREFS = {
    "18-24": [0.10, 0.20, 0.15, 0.15, 0.40],
    "25-34": [0.25, 0.30, 0.20, 0.15, 0.10],
    "35-44": [0.35, 0.25, 0.20, 0.12, 0.08],
    "45-54": [0.40, 0.20, 0.20, 0.15, 0.05],
    "55+":   [0.45, 0.25, 0.15, 0.10, 0.05],
}

SATISFACTION_OPTIONS = ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"]
NPS_OPTIONS          = ["Promoter", "Passive", "Detractor"]

def weighted_choice(options, weights):
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return np.random.choice(options, p=weights)

def generate_poll_data(n=N_RESPONDENTS):
    records = []
    dates   = pd.date_range("2024-01-01", "2024-06-30", periods=n)
    dates_arr = dates.to_numpy().copy()
    np.random.shuffle(dates_arr)
    dates = pd.DatetimeIndex(dates_arr)

    for i in range(n):
        region    = np.random.choice(REGIONS)
        age_group = np.random.choice(AGE_GROUPS)
        gender    = np.random.choice(GENDERS, p=[0.48, 0.48, 0.04])
        education = np.random.choice(EDUCATION, p=[0.20, 0.45, 0.25, 0.10])

        # Blend regional + age biases 50/50
        r_w = np.array(REGIONAL_PREFS[region])
        a_w = np.array(AGE_PREFS[age_group])
        blended = (r_w + a_w) / 2

        product      = weighted_choice(PRODUCTS, blended)
        satisfaction = weighted_choice(SATISFACTION_OPTIONS, [0.25, 0.35, 0.20, 0.12, 0.08])
        nps          = weighted_choice(NPS_OPTIONS,          [0.40, 0.35, 0.25])
        rating       = np.random.randint(1, 11)
        would_buy    = np.random.choice(["Yes", "No", "Maybe"], p=[0.55, 0.20, 0.25])

        records.append({
            "respondent_id":    f"R{1000 + i}",
            "date":             dates[i].strftime("%Y-%m-%d"),
            "region":           region,
            "age_group":        age_group,
            "gender":           gender,
            "education":        education,
            "preferred_product": product,
            "satisfaction":     satisfaction,
            "nps_category":     nps,
            "rating":           rating,
            "would_buy_again":  would_buy,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_poll_data()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/poll_data.csv", index=False)
    print(f"✅ Dataset created: {len(df)} rows × {len(df.columns)} columns")
    print(df.head())
