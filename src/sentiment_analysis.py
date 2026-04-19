"""
sentiment_analysis.py
---------------------
Upgrade 3: Adds a free-text feedback column to the dataset, then:
  - Generates a word cloud of common feedback terms
  - Runs VADER sentiment analysis (positive / neutral / negative)
  - Correlates sentiment with product preference and rating
  - Saves word cloud image to outputs/

Install:  pip install wordcloud vaderSentiment nltk
Run:      python src/sentiment_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Handle optional imports gracefully ────────────────────────────────────────
try:
    from wordcloud import WordCloud, STOPWORDS
    HAS_WC = True
except ImportError:
    HAS_WC = False
    print("wordcloud not installed. Run: pip install wordcloud")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("vaderSentiment not installed. Run: pip install vaderSentiment")

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. GENERATE SYNTHETIC FEEDBACK TEXT
# ══════════════════════════════════════════════════════════════════════════════

POSITIVE_COMMENTS = [
    "Excellent product, highly recommend it",
    "Great quality and amazing value for money",
    "Very satisfied with the performance",
    "Best purchase I have made, outstanding",
    "Fantastic experience, will buy again",
    "Love the design and the features",
    "Superb quality, exceeded my expectations",
    "Really happy with this product",
]

NEUTRAL_COMMENTS = [
    "It is okay, nothing special",
    "Average product, does the job",
    "Decent enough for the price",
    "Works as expected, no complaints",
    "Not bad, not great either",
    "Meets basic requirements",
    "Can be improved in some areas",
    "Acceptable but room for improvement",
]

NEGATIVE_COMMENTS = [
    "Disappointed with the quality",
    "Does not meet my expectations at all",
    "Poor performance, not worth the price",
    "Had issues from the beginning",
    "Would not recommend to anyone",
    "Very bad experience overall",
    "Terrible quality, waste of money",
    "Unreliable and frustrating to use",
]

def add_feedback_column(df):
    """Add synthetic open-text feedback correlated with satisfaction."""
    comments = []
    for _, row in df.iterrows():
        sat = row["satisfaction"]
        if sat in ["Very Satisfied", "Satisfied"]:
            c = np.random.choice(POSITIVE_COMMENTS)
        elif sat == "Neutral":
            c = np.random.choice(NEUTRAL_COMMENTS)
        else:
            c = np.random.choice(NEGATIVE_COMMENTS)
        comments.append(c)
    df = df.copy()
    df["feedback"] = comments
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. WORD CLOUD
# ══════════════════════════════════════════════════════════════════════════════

def generate_word_cloud(df, product_filter=None, save_path="outputs/11_word_cloud.png"):
    if not HAS_WC:
        print("  Skipping word cloud — wordcloud not installed.")
        return None

    df_wc = df.copy()
    if product_filter:
        df_wc = df_wc[df_wc["preferred_product"] == product_filter]

    text = " ".join(df_wc["feedback"].tolist())
    stopwords = STOPWORDS | {"the", "and", "for", "with", "from", "this", "that", "will"}

    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="Blues",
        stopwords=stopwords,
        max_words=80,
        prefer_horizontal=0.9,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    title = f"Word Cloud — {product_filter} Feedback" if product_filter else "Word Cloud — All Feedback"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Word cloud saved → {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 3. VADER SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_sentiment_analysis(df):
    if not HAS_VADER:
        print("  Skipping sentiment — vaderSentiment not installed.")
        return df

    analyzer = SentimentIntensityAnalyzer()

    def classify(text):
        score = analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:  return "Positive"
        if score <= -0.05: return "Negative"
        return "Neutral"

    def get_score(text):
        return round(analyzer.polarity_scores(text)["compound"], 4)

    df = df.copy()
    df["sentiment_label"] = df["feedback"].apply(classify)
    df["sentiment_score"]  = df["feedback"].apply(get_score)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. SENTIMENT CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_sentiment_charts(df):
    if "sentiment_label" not in df.columns:
        print("  Run run_sentiment_analysis() first.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    SENT_COLORS = {"Positive": "#16A34A", "Neutral": "#EAB308", "Negative": "#DC2626"}

    # 1 — Overall sentiment distribution
    counts = df["sentiment_label"].value_counts()
    axes[0].bar(counts.index, counts.values,
                color=[SENT_COLORS[s] for s in counts.index],
                edgecolor="white", width=0.55)
    for i, (label, v) in enumerate(counts.items()):
        axes[0].text(i, v + 5, f"{v}", ha="center", fontweight="bold", fontsize=11)
    axes[0].set_title("Overall Sentiment", fontweight="bold")
    axes[0].set_ylabel("Respondents")
    axes[0].spines[["top", "right"]].set_visible(False)

    # 2 — Sentiment by product
    pivot = pd.crosstab(df["preferred_product"], df["sentiment_label"])
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)
    pivot_pct[["Positive", "Neutral", "Negative"]].plot(
        kind="bar", stacked=True, ax=axes[1],
        color=[SENT_COLORS["Positive"], SENT_COLORS["Neutral"], SENT_COLORS["Negative"]],
        edgecolor="white", linewidth=0.5
    )
    axes[1].set_title("Sentiment by Product", fontweight="bold")
    axes[1].set_ylabel("Share (%)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title="Sentiment", fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    # 3 — Avg sentiment score by product
    avg_score = df.groupby("preferred_product")["sentiment_score"].mean().round(3)
    colors = ["#16A34A" if v > 0 else "#DC2626" for v in avg_score]
    axes[2].bar(avg_score.index, avg_score.values, color=colors, edgecolor="white", width=0.55)
    axes[2].axhline(y=0, color="gray", linestyle="--", linewidth=1)
    for i, v in enumerate(avg_score.values):
        axes[2].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    axes[2].set_title("Avg Sentiment Score by Product", fontweight="bold")
    axes[2].set_ylabel("VADER Compound Score")
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Sentiment Analysis — Open-Text Feedback", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = "outputs/12_sentiment_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Sentiment chart saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def sentiment_summary(df):
    if "sentiment_label" not in df.columns:
        return
    print("\nSENTIMENT ANALYSIS SUMMARY")
    print("─" * 40)
    dist = df["sentiment_label"].value_counts(normalize=True).mul(100).round(1)
    for label, pct in dist.items():
        print(f"  {label:10s}: {pct}%")
    print(f"\n  Avg sentiment score: {df['sentiment_score'].mean():.4f}")
    print(f"  Most positive product: {df.groupby('preferred_product')['sentiment_score'].mean().idxmax()}")
    print(f"  Most negative product: {df.groupby('preferred_product')['sentiment_score'].mean().idxmin()}")

    corr = df[["sentiment_score", "rating"]].corr().iloc[0, 1]
    print(f"\n  Correlation (sentiment ↔ rating): {corr:.4f}")
    print("─" * 40 + "\n")


if __name__ == "__main__":
    from src.analysis import load_and_clean
    df = load_and_clean("data/poll_data.csv")
    df = add_feedback_column(df)
    df = run_sentiment_analysis(df)
    generate_word_cloud(df)
    plot_sentiment_charts(df)
    sentiment_summary(df)
