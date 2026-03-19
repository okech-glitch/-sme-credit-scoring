"""
===================================================================================
  data.org Financial Health Prediction Challenge — API Export Script
  
  This script trains a SINGLE production instance of the 3 V2 models (LGB, CAT, XGB)
  on 100% of the training data using the hyperparameters found by Optuna in v2, 
  and saves them to disk via joblib for the FastAPI endpoint.
===================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent
MODEL_DIR = DATA_DIR / "api_models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  Exporting Final Production Models for FastAPI")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & BASE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading and transforming 100% of train data...")
train = pd.read_csv(DATA_DIR / "Train.csv")
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}
train["target_enc"] = train["Target"].map(TARGET_MAP)

yn_map = {
    "yes": 1.0, "no": 0.0, "Yes": 1.0, "No": 0.0,
    "Have now": 2.0, "Used to have but don't have now": 1.0, "Never had": 0.0,
    "Don't know or N/A": np.nan, "Don't know": np.nan,
    "Don't Know": np.nan, "Don't know (Do not show)": np.nan, "Refused": np.nan,
}

def build_base(df):
    df = df.copy()
    str_cols = [c for c in df.columns if df[c].dtype == object and c not in ["ID", "country", "Target"]]
    for col in str_cols: df[col] = df[col].map(yn_map)
    df["country_rank"] = df["country"].str.lower().map({"lesotho":0, "malawi":1, "zimbabwe":2, "eswatini":3}).fillna(1)

    df["factor_credit_access"] = df[[c for c in df.columns if any(k in c for k in ["bank","credit","loan","borrow","mfi","sacco","microfinance","mobile_money"])]].mean(axis=1)
    df["factor_resilience"] = df[[c for c in df.columns if any(k in c for k in ["insurance","savings","emergency","shock","resilience","cope","support"])]].mean(axis=1)
    df["factor_debt_burden"] = df[[c for c in df.columns if any(k in c for k in ["debt","repay","default","overdue","informal_lender","friends_family"])]].mean(axis=1)
    df["factor_stability"] = df[[c for c in df.columns if any(k in c for k in ["stable","worried","shutdown","consistent","income","profit","revenue","turnover","compliance","tax","register"])]].mean(axis=1)
    
    df["composite_fhi"] = df["factor_credit_access"] * 0.30 + df["factor_resilience"] * 0.30 + df["factor_debt_burden"] * 0.20 + df["factor_stability"] * 0.20
    
    if "owner_age" in df.columns:
        df["age_group"] = pd.cut(df["owner_age"], bins=[0,25,35,50,70,999], labels=[0,1,2,3,4]).astype(float)
    if "personal_income" in df.columns and "business_expenses" in df.columns:
        df["income_expense_ratio"] = (df["personal_income"] / (df["business_expenses"].replace(0, np.nan) + 1)).clip(0, 100)
        df["log_income"] = np.log1p(df["personal_income"].clip(0))
        df["log_expenses"] = np.log1p(df["business_expenses"].clip(0))

    df["country_fhi"] = df["country_rank"] * df["composite_fhi"]
    df["country_credit"] = df["country_rank"] * df["factor_credit_access"]
    df["country_resilience"] = df["country_rank"] * df["factor_resilience"]

    num_cols = df.select_dtypes(include="number").columns
    for c in num_cols:
        if df[c].isnull().sum() > 0: df[f"{c}_missing"] = df[c].isnull().astype(int)
    return df.fillna(-999)

train_df = build_base(train)

# ─────────────────────────────────────────────────────────────────────────────
# 2. ENCODERS & TARGET ENCODING FITTING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Fitting Encoders and Target Encoding on full dataset...")
drop_cols = ["ID", "Target", "target_enc", "country"]
feat_cols = [c for c in train_df.columns if c not in drop_cols]
cat_cols = [c for c in feat_cols if train_df[c].dtype == object]

label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    train_df[c] = le.fit_transform(train_df[c].astype(str))
    label_encoders[c] = le

joblib.dump(label_encoders, MODEL_DIR / "label_encoders.pkl")
joblib.dump(feat_cols, MODEL_DIR / "base_feat_cols.pkl")

X_raw = train_df[feat_cols].values
y = train_df["target_enc"].values

# Fit global target encoding on 100% data
n_classes = 3
global_mean = np.array([(y == c).mean() for c in range(n_classes)])
SMOOTHING = 10
country_rank_arr = train_df["country_rank"].values

te_mapping = {cls: {} for cls in range(n_classes)}
te_train = np.zeros((len(X_raw), n_classes))

for cls in range(n_classes):
    for cid in range(4):
        mask = country_rank_arr == cid
        n = mask.sum()
        target_mean = (y[mask] == cls).mean() if n > 0 else global_mean[cls]
        smoothed = (n * target_mean + SMOOTHING * global_mean[cls]) / (n + SMOOTHING)
        te_mapping[cls][cid] = smoothed
    te_train[:, cls] = np.array([te_mapping[cls].get(c, global_mean[cls]) for c in country_rank_arr])

joblib.dump({"mapping": te_mapping, "global_mean": global_mean}, MODEL_DIR / "target_encoding.pkl")

X = np.concatenate([X_raw, te_train], axis=1)

class_counts = np.bincount(y)
class_weight_map = {i: len(y) / (len(class_counts) * cnt) for i, cnt in enumerate(class_counts)}
sample_weights = np.array([class_weight_map[lbl] for lbl in y])
cat_cw = [class_weight_map[i] for i in range(3)]

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN SINGLE PRODUCTION MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Training final production models...")

# Best params found in v2 Optuna
LGB_PARAMS = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "learning_rate": 0.0674, "num_leaves": 202, "min_child_samples": 42,
    "feature_fraction": 0.96, "bagging_fraction": 0.77, "bagging_freq": 1,
    "lambda_l1": 0.2, "lambda_l2": 0.5, "verbose": -1, "seed": SEED, "n_jobs": -1,
}
print("  - LightGBM...")
ds_tr = lgb.Dataset(X, label=y, weight=sample_weights)
model_lgb = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=100) # smaller ensemble for API latency

print("  - CatBoost...")
model_cat = cb.CatBoostClassifier(
    iterations=150, learning_rate=0.0524, depth=7, l2_leaf_reg=3.5,
    loss_function="MultiClass", class_weights=cat_cw, random_seed=SEED, verbose=False)
model_cat.fit(X, y)

print("  - XGBoost...")
XGB_PARAMS = {
    "objective":"multi:softprob", "num_class":3, "eval_metric":"mlogloss",
    "learning_rate":0.05, "max_depth":6, "min_child_weight":5,
    "subsample":0.8, "colsample_bytree":0.8, "reg_alpha":0.5, "reg_lambda":2.0,
    "seed":SEED, "nthread":-1, "verbosity":0
}
dtrain = xgb.DMatrix(X, label=y, weight=sample_weights)
model_xgb = xgb.train(XGB_PARAMS, dtrain, num_boost_round=100)

# Save models
joblib.dump(model_lgb, MODEL_DIR / "lgb.pkl")
joblib.dump(model_cat, MODEL_DIR / "cat.pkl")
joblib.dump(model_xgb, MODEL_DIR / "xgb.pkl")

# Save the ensemble weights discovered in v2 (60% LGB, 20% CAT, 20% XGB)
joblib.dump([0.6, 0.2, 0.2], MODEL_DIR / "ensemble_weights.pkl")

print("\n[4/4] Models exported to 'api_models/'!")
print("✅ Ready to run app.py")
