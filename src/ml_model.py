"""
ml_model.py
-----------
Upgrade 4: Trains a Random Forest classifier to PREDICT which product
a new respondent will prefer based on their demographic profile.

Shows:
  - Feature engineering (label encoding)
  - Train/test split
  - Model training + hyperparameter tuning
  - Evaluation: accuracy, classification report, confusion matrix
  - Feature importance chart
  - Prediction function (usable in the Streamlit dashboard)

Install:  pip install scikit-learn
Run:      python src/ml_model.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = ["region", "age_group", "gender", "education"]
TARGET_COL   = "preferred_product"

def encode_features(df):
    df_enc = df[FEATURE_COLS + [TARGET_COL]].copy()

    encoders = {}
    for col in FEATURE_COLS + [TARGET_COL]:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    return df_enc, encoders


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_model(df):
    df_enc, encoders = encode_features(df)

    X = df_enc[FEATURE_COLS]
    y = df_enc[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    return model, encoders, X_test, y_test, cv_scores


# ══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, encoders, X_test, y_test, cv_scores):
    y_pred    = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    products  = encoders[TARGET_COL].classes_

    print("\n🤖 ML MODEL EVALUATION — Random Forest")
    print("─" * 55)
    print(f"  Test accuracy:            {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  5-fold CV accuracy:       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Baseline (random guess):  {1/len(products):.4f} ({100/len(products):.1f}%)")
    print(f"  Improvement over baseline:{(accuracy - 1/len(products))*100:+.1f} pp")
    print()
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=products))
    print("─" * 55)

    return accuracy, y_pred, products


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_ml_results(model, encoders, X_test, y_test, y_pred, products):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances.sort_values().plot(kind="barh", ax=axes[0],
                                   color="#2563EB", edgecolor="white")
    axes[0].set_title("Feature Importance", fontweight="bold")
    axes[0].set_xlabel("Importance Score")
    axes[0].spines[["top", "right"]].set_visible(False)
    for bar in axes[0].patches:
        axes[0].text(bar.get_width() + 0.002,
                     bar.get_y() + bar.get_height() / 2,
                     f"{bar.get_width():.3f}", va="center", fontsize=10)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=products)
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion Matrix", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle("ML Model — Product Preference Predictor", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = "outputs/13_ml_model_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 ML chart saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 5. PREDICTION FUNCTION (use in Streamlit dashboard)
# ══════════════════════════════════════════════════════════════════════════════

def predict_preference(model, encoders, region, age_group, gender, education):
    """
    Predict product preference for a new respondent.

    Returns:
        dict with predicted product and probability for each product.

    Example:
        predict_preference(model, encoders, "North", "25-34", "Female", "Bachelor's")
    """
    input_data = {}
    for col, val in zip(FEATURE_COLS, [region, age_group, gender, education]):
        try:
            input_data[col] = encoders[col].transform([val])[0]
        except ValueError:
            input_data[col] = 0  # fallback for unseen label

    X_new = pd.DataFrame([input_data])
    pred_encoded   = model.predict(X_new)[0]
    pred_product   = encoders[TARGET_COL].inverse_transform([pred_encoded])[0]
    proba          = model.predict_proba(X_new)[0]
    products       = encoders[TARGET_COL].classes_
    proba_dict     = {p: round(float(pr) * 100, 1) for p, pr in zip(products, proba)}
    proba_sorted   = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))

    return {
        "predicted_product": pred_product,
        "probabilities":     proba_sorted,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, encoders):
    joblib.dump(model,    "models/rf_model.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    print("  💾 Model saved → models/rf_model.pkl")

def load_model():
    model    = joblib.load("models/rf_model.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return model, encoders


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.analysis import load_and_clean

    df = load_and_clean("data/poll_data.csv")

    print("🔧 Training Random Forest model...")
    model, encoders, X_test, y_test, cv_scores = train_model(df)

    accuracy, y_pred, products = evaluate_model(model, encoders, X_test, y_test, cv_scores)
    plot_ml_results(model, encoders, X_test, y_test, y_pred, products)
    save_model(model, encoders)

    # Demo prediction
    print("\n🎯 Demo prediction:")
    result = predict_preference(model, encoders, "North", "25-34", "Female", "Bachelor's")
    print(f"  Input: North | 25-34 | Female | Bachelor's")
    print(f"  → Predicted: {result['predicted_product']}")
    print(f"  → Probabilities: {result['probabilities']}")
