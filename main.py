"""
main.py
-------
Entry point for the Poll Results Visualizer.

Run:
    python main.py

Steps performed automatically:
  1. Generate synthetic poll data  →  data/poll_data.csv
  2. Load and clean the dataset
  3. Run all analysis functions
  4. Generate and save 10 charts   →  outputs/
  5. Print summary insights to console
"""

import os
import sys

# Make sure src/ is on the path when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.generate_data import generate_poll_data
from src.analysis      import load_and_clean, generate_insights
from src.visualizer    import generate_all_charts

BANNER = """
╔══════════════════════════════════════════════════════╗
║         POLL  RESULTS  VISUALIZER  v1.0              ║
║   Data Analysis · Visualization · Insights           ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── STEP 1: Generate data (skip if already exists) ────────────────────
    csv_path = "data/poll_data.csv"
    if not os.path.exists(csv_path):
        print("📝 Generating synthetic poll dataset …")
        os.makedirs("data", exist_ok=True)
        df_raw = generate_poll_data()
        df_raw.to_csv(csv_path, index=False)
        print(f"   Saved → {csv_path}")
    else:
        print(f"📂 Dataset already exists at {csv_path}")

    # ── STEP 2: Load & clean ──────────────────────────────────────────────
    print("\n🧹 Loading and cleaning data …")
    df = load_and_clean(csv_path)
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── STEP 3: Generate all charts ───────────────────────────────────────
    generate_all_charts(df)

    # ── STEP 4: Print insights ────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("💡  KEY INSIGHTS")
    print("─" * 55)
    for insight in generate_insights(df):
        print(" ", insight)
    print("─" * 55)

    print("\n🎉  Done!  Open the outputs/ folder to view all charts.\n")


if __name__ == "__main__":
    main()
