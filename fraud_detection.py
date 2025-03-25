import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                             precision_recall_curve, roc_auc_score, 
                             average_precision_score)
0
# Load fraud dataset
file_path = "fraud_preprocessed.csv"
fraud_data = pd.read_csv(file_path)

# --------------------------+
# Enhanced Feature Engineering
# --------------------------
X = fraud_data.drop(columns=["Defect rates"])
y = (fraud_data["Defect rates"] > 0.5).astype(int)

# Create interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = pd.DataFrame(poly.fit_transform(X), 
                     columns=poly.get_feature_names_out(X.columns))

# --------------------------
# Advanced Resampling
# --------------------------
resampler = SMOTEENN(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = resampler.fit_resample(X_poly, y)

# --------------------------
# Feature Selection
# --------------------------
class_ratio = len(y[y == 0]) / len(y[y == 1])

selector = SelectFromModel(
    XGBClassifier(random_state=42, scale_pos_weight=class_ratio),
    threshold='median'
).fit(X_resampled, y_resampled)

X_selected = selector.transform(X_resampled)
selected_features = X_poly.columns[selector.get_support()]

# --------------------------
# Data Splitting with Stratification
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_resampled, 
    test_size=0.2, 
    stratify=y_resampled,
    random_state=42
)

# --------------------------
# XGBoost Optimization
# --------------------------
xgb_params = {
    'n_estimators': [500, 700, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1],
    'scale_pos_weight': [class_ratio]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_search = RandomizedSearchCV(
    xgb, xgb_params, n_iter=25, 
    cv=cv, scoring='roc_auc', 
    n_jobs=-1, random_state=42
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

# --------------------------
# Random Forest Optimization
# --------------------------
rf_params = {
    'n_estimators': [300, 500],
    'max_depth': [5, 7, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', {0: 1, 1: 4}]
}

rf_model = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params, n_iter=15, cv=cv,
    scoring='roc_auc', n_jobs=-1
)
rf_model.fit(X_train, y_train)
best_rf = rf_model.best_estimator_

# --------------------------
# Ensemble Model with Voting Classifier
# --------------------------
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', best_rf)
    ],
    voting='soft',
    weights=[3, 2]  # Adjust based on validation performance
)
ensemble_model.fit(X_train, y_train)

# --------------------------
# Probability Predictions
# --------------------------
y_probs = ensemble_model.predict_proba(X_test)[:, 1]

# --------------------------
# Cost-Sensitive Threshold Optimization
# --------------------------
def custom_cost(y_true, y_prob, fp_cost=1, fn_cost=5):
    thresholds = np.linspace(0, 1, 100)
    costs = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        costs.append(fp * fp_cost + fn * fn_cost)
    return thresholds[np.argmin(costs)]

best_threshold = custom_cost(y_test, y_probs)
y_pred_adjusted = (y_probs >= best_threshold).astype(int)

# --------------------------
# Comprehensive Evaluation
# --------------------------
accuracy = accuracy_score(y_test, y_pred_adjusted)
roc_auc = roc_auc_score(y_test, y_probs)
avg_precision = average_precision_score(y_test, y_probs)

print(f"✅ Enhanced Fraud Detection Accuracy: {accuracy * 100:.2f}%")
print(f"🎯 ROC-AUC: {roc_auc:.3f}")
print(f"📈 Average Precision: {avg_precision:.3f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred_adjusted))

# --------------------------
# Advanced Model Analysis
# --------------------------
# Feature Importance Plot
plt.figure(figsize=(12, 6))
plot_importance(best_xgb, max_num_features=15)
plt.title("XGBoost Feature Importance")
plt.savefig("xgboost_feature_importance.png", bbox_inches='tight')

# Error Analysis
error_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred_adjusted,
    'probability': y_probs
}).reset_index(drop=True)

# Save error analysis
error_df.to_csv("error_analysis.csv", index=False)
