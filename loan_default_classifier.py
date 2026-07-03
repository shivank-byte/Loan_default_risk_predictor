"""
Loan Default Prediction — Binary Classification
Dataset: Kaggle Loan Default (255,347 rows, 18 columns)
Author: Shivank Thakur

Goal: Predict whether a borrower will default (1) or not (0) on a loan,
using borrower demographic, financial, and loan-specific features.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# ── 1. LOAD & CLEAN ────────────────────────────────────────────────
df = pd.read_csv('Loan_default.csv')
df = df.drop(columns=['LoanID'])   # identifier, not predictive

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Default rate:", round(df['Default'].mean(), 4))
# Dataset is imbalanced: ~11.6% default rate. Accuracy alone will be
# misleading here — a model that always predicts "no default" would
# score ~88% accuracy while being useless. Precision/recall/ROC-AUC
# matter more than accuracy for this problem.

# ── 2. ENCODE & SPLIT ───────────────────────────────────────────────
X = df.drop(columns=['Default'])
y = df['Default']
X = pd.get_dummies(X, drop_first=True)  # one-hot encode categoricals
# drop_first=True avoids the "dummy variable trap" (redundant column
# that's a linear combination of the others)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y keeps the same default rate in both train and test sets
)

# ── 3. TRAIN TWO MODELS (never trust one model's word for it) ──────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline: Logistic Regression (simple, interpretable)
log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]

# Comparison: Random Forest (captures non-linear interactions)
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# class_weight='balanced' tells the model to weight the minority
# (default) class more heavily during training, instead of defaulting
# to the lazy majority-class prediction.

# ── 4. EVALUATE — precision/recall/ROC-AUC, NOT just accuracy ──────
print("\n=== LOGISTIC REGRESSION ===")
print(classification_report(y_test, log_preds, target_names=['No Default', 'Default']))
print("ROC-AUC:", round(roc_auc_score(y_test, log_probs), 4))

print("\n=== RANDOM FOREST ===")
print(classification_report(y_test, rf_preds, target_names=['No Default', 'Default']))
print("ROC-AUC:", round(roc_auc_score(y_test, rf_probs), 4))

# ── 5. VISUALIZE: confusion matrix + ROC curve ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test, rf_preds)
axes[0].imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['No Default', 'Default'])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(['No Default', 'Default'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('Random Forest — Confusion Matrix')

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
axes[1].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, rf_probs):.3f})')
axes[1].plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC={roc_auc_score(y_test, log_probs):.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random guess')
axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve Comparison')
axes[1].legend()

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150)

# ── 6. FEATURE IMPORTANCE — what actually predicts default? ────────
importances = pd.Series(
    rf_model.feature_importances_, index=X_test.columns
).sort_values(ascending=False).head(10)
print("\nTop 10 predictors:\n", importances)

plt.figure(figsize=(8, 5))
importances.sort_values().plot(kind='barh', color='#2c6e91')
plt.title('Top 10 Predictors of Loan Default (Random Forest)')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)

# NOTE: Age is the single strongest predictor here. This is a
# correlation the model found in the data, not a causal claim -
# it may reflect genuine risk differences by age, or it may be a
# proxy for something the model can't see directly (career stage,
# loan-purpose mix, etc.). Worth stating explicitly, not overclaiming.
