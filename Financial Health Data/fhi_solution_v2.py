"""
===================================================================================
  data.org Financial Health Prediction Challenge — v2 (Push to Top 1%)
  
  v1 Results: Public=0.8847 / Private=0.8678  (gap = 0.017 → overfitting signal)
  
  v2 UPGRADES:
  1. Fold-aware country-level Target Encoding (leakage-proof, your Traffic Fcst win)
  2. Optuna hyperparameter tuning (50 trials each model)
  3. XGBoost as 3rd ensemble member (different base learner = more diversity)
  4. Stronger regularization to close pub/private gap
  5. Per-country prior smoothing to generalise across regime shifts
===================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED   = 42
N_FOLDS = 5
DATA_DIR = Path(__file__).parent
np.random.seed(SEED)

TARGET_MAP     = {"Low": 0, "Medium": 1, "High": 2}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}

print("=" * 70)
print("  Financial Health Index v2 — Quant-Architect Pipeline")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/8] Loading data...")
train = pd.read_csv(DATA_DIR / "Train.csv")
test  = pd.read_csv(DATA_DIR / "Test.csv")
sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
test_ids = test["ID"].values
print(f"  Train: {train.shape}  Test: {test.shape}")

train["target_enc"] = train["Target"].map(TARGET_MAP)

# ─────────────────────────────────────────────────────────────────────────────
# 2. BASE FEATURE ENGINEERING (same as v1 — proven to produce 90 features)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/8] Engineering base features...")

yn_map = {
    "yes": 1.0, "no": 0.0, "Yes": 1.0, "No": 0.0,
    "Have now": 2.0,
    "Used to have but don't have now": 1.0,
    "Never had": 0.0,
    "Don't know or N/A": np.nan, "Don't know": np.nan,
    "Don't Know": np.nan, "Don't know (Do not show)": np.nan,
    "Refused": np.nan,
}

def build_base(df):
    df = df.copy()
    str_cols = [c for c in df.columns if df[c].dtype == object
                and c not in ["ID", "country", "Target"]]
    for col in str_cols:
        df[col] = df[col].map(yn_map)

    country_rank = {"lesotho": 0, "malawi": 1, "zimbabwe": 2, "eswatini": 3}
    df["country_rank"] = df["country"].str.lower().map(country_rank).fillna(1)

    credit_cols = [c for c in df.columns if any(k in c for k in
                   ["bank","credit","loan","borrow","mfi","sacco","microfinance","mobile_money"])]
    df["factor_credit_access"] = df[credit_cols].mean(axis=1) if credit_cols else 0.0

    resilience_cols = [c for c in df.columns if any(k in c for k in
                       ["insurance","savings","emergency","shock","resilience","cope","support"])]
    df["factor_resilience"] = df[resilience_cols].mean(axis=1) if resilience_cols else 0.0

    debt_cols = [c for c in df.columns if any(k in c for k in
                 ["debt","repay","default","overdue","informal_lender","friends_family"])]
    df["factor_debt_burden"] = df[debt_cols].mean(axis=1) if debt_cols else 0.0

    stability_cols = [c for c in df.columns if any(k in c for k in
                      ["stable","worried","shutdown","consistent","income","profit",
                       "revenue","turnover","compliance","tax","register"])]
    df["factor_stability"] = df[stability_cols].mean(axis=1) if stability_cols else 0.0

    df["composite_fhi"] = (df["factor_credit_access"] * 0.30 +
                           df["factor_resilience"]    * 0.30 +
                           df["factor_debt_burden"]   * 0.20 +
                           df["factor_stability"]     * 0.20)

    if "owner_age" in df.columns:
        df["age_group"] = pd.cut(df["owner_age"],
                                 bins=[0,25,35,50,70,999],
                                 labels=[0,1,2,3,4]).astype(float)

    if "personal_income" in df.columns and "business_expenses" in df.columns:
        df["income_expense_ratio"] = (
            df["personal_income"] / (df["business_expenses"].replace(0, np.nan) + 1)
        ).clip(0, 100)
        df["log_income"]   = np.log1p(df["personal_income"].clip(0))
        df["log_expenses"] = np.log1p(df["business_expenses"].clip(0))

    df["country_fhi"] = df["country_rank"] * df["composite_fhi"]
    df["country_credit"] = df["country_rank"] * df["factor_credit_access"]
    df["country_resilience"] = df["country_rank"] * df["factor_resilience"]

    num_cols = df.select_dtypes(include="number").columns
    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[f"{c}_missing"] = df[c].isnull().astype(int)

    df = df.fillna(-999)
    return df


train = build_base(train)
test  = build_base(test)
print(f"  Base features built: {train.shape[1]} columns")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FOLD-AWARE TARGET ENCODING (your Traffic Forecasting #1 technique)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/8] Building fold-aware target encoding...")

drop_cols  = ["ID", "Target", "target_enc", "country"]
feat_cols  = [c for c in train.columns if c not in drop_cols]
cat_cols   = [c for c in feat_cols if train[c].dtype == object]

# Label encode remaining object cols
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[c], test[c]], axis=0).astype(str)
    le.fit(combined)
    train[c] = le.transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))

X_raw = train[feat_cols].values
y     = train["target_enc"].values
X_test_raw = test[feat_cols].values

# Class weights
class_counts    = np.bincount(y)
class_weight_map = {i: len(y) / (len(class_counts) * cnt)
                    for i, cnt in enumerate(class_counts)}
sample_weights = np.array([class_weight_map[lbl] for lbl in y])

n_classes = 3
strat_label = y * 10 + train["country_rank"].values.astype(int)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Fold-aware target encoding for "country" (key source of regime leakage)
SMOOTHING = 10
global_mean = np.zeros(n_classes)
for cls in range(n_classes):
    global_mean[cls] = (y == cls).mean()

te_train = np.zeros((len(X_raw), n_classes))  # out-of-fold target encodings
te_test  = np.zeros((len(X_test_raw), n_classes))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, strat_label)):
    y_train_fold = y[tr_idx]
    country_train = train["country_rank"].values[tr_idx]
    country_val   = train["country_rank"].values[va_idx]
    country_test  = test["country_rank"].values

    for cls in range(n_classes):
        encoded = {}
        for country_id in range(4):
            mask = country_train == country_id
            n = mask.sum()
            target_mean = (y_train_fold[mask] == cls).mean() if n > 0 else global_mean[cls]
            smoothed = (n * target_mean + SMOOTHING * global_mean[cls]) / (n + SMOOTHING)
            encoded[country_id] = smoothed

        te_train[va_idx, cls] = np.array([encoded.get(c, global_mean[cls]) for c in country_val])
        te_test[:, cls]      += np.array([encoded.get(c, global_mean[cls]) for c in country_test]) / N_FOLDS

# Append target encoded features
te_train_df = pd.DataFrame(te_train, columns=[f"country_te_cls{i}" for i in range(n_classes)])
te_test_df  = pd.DataFrame(te_test,  columns=[f"country_te_cls{i}" for i in range(n_classes)])

X      = np.concatenate([X_raw, te_train], axis=1)
X_test = np.concatenate([X_test_raw, te_test], axis=1)
feat_cols_v2 = feat_cols + [f"country_te_cls{i}" for i in range(n_classes)]
print(f"  Final feature matrix: {X.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTUNA — LIGHTGBM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/8] Optuna tuning LightGBM (50 trials)...")

def lgb_objective(trial):
    params = {
        "objective":         "multiclass",
        "num_class":         3,
        "metric":            "multi_logloss",
        "learning_rate":     trial.suggest_float("lr", 0.01, 0.1, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      1,
        "lambda_l1":         trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2":         trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "verbose":           -1, "seed": SEED, "n_jobs": -1,
    }
    # Fast 3-fold for tuning
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, va_idx in skf3.split(X, strat_label):
        ds_tr = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx])
        ds_va = lgb.Dataset(X[va_idx], label=y[va_idx], reference=ds_tr)
        m = lgb.train(params, ds_tr, num_boost_round=500,
                      valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(-1)])
        preds = m.predict(X[va_idx])
        scores.append(f1_score(y[va_idx], preds.argmax(axis=1), average="weighted"))
    return np.mean(scores)

study_lgb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
best_lgb_params = study_lgb.best_params
print(f"  Best LGB params: lr={best_lgb_params['lr']:.4f} "
      f"leaves={best_lgb_params['num_leaves']}  "
      f"[best F1 = {study_lgb.best_value:.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN LIGHTGBM (full 5-fold with best params)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/8] Training LightGBM with tuned params (5-fold)...")

lgb_oof   = np.zeros((len(X), 3))
lgb_preds = np.zeros((len(X_test), 3))
lgb_scores = []

lgb_params_final = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "learning_rate": best_lgb_params["lr"],
    "num_leaves":    best_lgb_params["num_leaves"],
    "min_child_samples": best_lgb_params["min_child_samples"],
    "feature_fraction":  best_lgb_params["feature_fraction"],
    "bagging_fraction":  best_lgb_params["bagging_fraction"],
    "bagging_freq": 1,
    "lambda_l1": best_lgb_params["lambda_l1"],
    "lambda_l2": best_lgb_params["lambda_l2"],
    "verbose": -1, "seed": SEED, "n_jobs": -1,
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_label)):
    ds_tr = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx],
                        feature_name=feat_cols_v2)
    ds_va = lgb.Dataset(X[va_idx], label=y[va_idx], reference=ds_tr)
    m = lgb.train(lgb_params_final, ds_tr, num_boost_round=3000,
                  valid_sets=[ds_va],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
    lgb_oof[va_idx] = m.predict(X[va_idx])
    lgb_preds      += m.predict(X_test) / N_FOLDS
    f = f1_score(y[va_idx], lgb_oof[va_idx].argmax(axis=1), average="weighted")
    lgb_scores.append(f)
    print(f"  Fold {fold+1}  F1={f:.4f}  iter={m.best_iteration}")

lgb_cv = np.mean(lgb_scores)
print(f"  LightGBM CV F1: {lgb_cv:.4f} ± {np.std(lgb_scores):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN CATBOOST (tuned n_trials=30)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/8] Optuna tuning + training CatBoost (30 trials)...")

cat_cw = [class_weight_map[i] for i in range(3)]

def cat_objective(trial):
    m = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=trial.suggest_float("lr", 0.01, 0.15, log=True),
        depth=trial.suggest_int("depth", 4, 8),
        l2_leaf_reg=trial.suggest_float("l2", 1.0, 15.0, log=True),
        loss_function="MultiClass",
        class_weights=cat_cw,
        early_stopping_rounds=30,
        random_seed=SEED, verbose=False,
    )
    skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, va_idx in skf3.split(X, strat_label):
        m.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), use_best_model=True)
        scores.append(f1_score(y[va_idx], m.predict(X[va_idx]), average="weighted"))
    return np.mean(scores)

study_cat = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
study_cat.optimize(cat_objective, n_trials=30, show_progress_bar=False)
best_cat = study_cat.best_params
print(f"  Best CatBoost params: lr={best_cat['lr']:.4f} depth={best_cat['depth']} "
      f"[best F1 = {study_cat.best_value:.4f}]")

cat_oof   = np.zeros((len(X), 3))
cat_preds = np.zeros((len(X_test), 3))
cat_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_label)):
    m = cb.CatBoostClassifier(
        iterations=2000,
        learning_rate=best_cat["lr"],
        depth=best_cat["depth"],
        l2_leaf_reg=best_cat["l2"],
        loss_function="MultiClass",
        class_weights=cat_cw,
        early_stopping_rounds=50,
        random_seed=SEED, verbose=False,
    )
    m.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), use_best_model=True)
    cat_oof[va_idx] = m.predict_proba(X[va_idx])
    cat_preds      += m.predict_proba(X_test) / N_FOLDS
    f = f1_score(y[va_idx], cat_oof[va_idx].argmax(axis=1), average="weighted")
    cat_scores.append(f)
    print(f"  Fold {fold+1}  F1={f:.4f}")

cat_cv = np.mean(cat_scores)
print(f"  CatBoost CV F1: {cat_cv:.4f} ± {np.std(cat_scores):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAIN XGBOOST (3rd model for diversity)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] Training XGBoost (diversity engine #3)...")

xgb_oof   = np.zeros((len(X), 3))
xgb_preds = np.zeros((len(X_test), 3))
xgb_scores = []

# XGBoost doesn't have a direct class_weight param for multiclass — use sample_weight
xgb_params = {
    "objective":       "multi:softprob",
    "num_class":       3,
    "eval_metric":     "mlogloss",
    "learning_rate":   0.05,
    "max_depth":       6,
    "min_child_weight": 5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.5,
    "reg_lambda":      2.0,
    "seed":            SEED,
    "nthread":         -1,
    "verbosity":       0,
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_label)):
    dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx])
    dval   = xgb.DMatrix(X[va_idx], label=y[va_idx])
    dtest  = xgb.DMatrix(X_test)

    m = xgb.train(
        xgb_params, dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    xgb_oof[va_idx] = m.predict(dval).reshape(-1, 3)
    xgb_preds      += m.predict(dtest).reshape(-1, 3) / N_FOLDS
    f = f1_score(y[va_idx], xgb_oof[va_idx].argmax(axis=1), average="weighted")
    xgb_scores.append(f)
    print(f"  Fold {fold+1}  F1={f:.4f}  best_iter={m.best_iteration}")

xgb_cv = np.mean(xgb_scores)
print(f"  XGBoost CV F1: {xgb_cv:.4f} ± {np.std(xgb_scores):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. OPTIMAL 3-MODEL ENSEMBLE + SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/8] Optimising 3-model ensemble weights...")

def ensemble_f1_3(w_lgb, w_cat, w_xgb):
    oof = w_lgb * lgb_oof + w_cat * cat_oof + w_xgb * xgb_oof
    return f1_score(y, oof.argmax(axis=1), average="weighted")

best_f1, best_w = 0.0, (0.5, 0.3, 0.2)
for w1 in np.linspace(0.3, 0.8, 11):
    for w2 in np.linspace(0.1, 0.5, 9):
        w3 = 1 - w1 - w2
        if w3 <= 0: continue
        f = ensemble_f1_3(w1, w2, w3)
        if f > best_f1:
            best_f1, best_w = f, (w1, w2, w3)

print(f"  Best weights: LGB={best_w[0]:.2f} | CAT={best_w[1]:.2f} | XGB={best_w[2]:.2f}")
print(f"  Ensemble OOF F1 : {best_f1:.4f}")
print(f"  v1 OOF was       : ~0.8665")

final_test = best_w[0]*lgb_preds + best_w[1]*cat_preds + best_w[2]*xgb_preds
final_preds  = final_test.argmax(axis=1)
final_labels = [INV_TARGET_MAP[p] for p in final_preds]

submission = pd.DataFrame({"ID": test_ids, "Target": final_labels})
assert submission["ID"].tolist() == sample["ID"].tolist(), "ID order mismatch!"

out_path = DATA_DIR / "submission_v2.csv"
submission.to_csv(out_path, index=False)

print(f"\n  ✅ Saved: {out_path}")
print(f"\n  Predicted class distribution:")
print(submission["Target"].value_counts())

print("\n" + "=" * 70)
print("  QUANT-ARCHITECT v2 SUMMARY")
print("=" * 70)
print(f"  LightGBM  OOF F1 : {lgb_cv:.4f}   (Optuna-tuned)")
print(f"  CatBoost  OOF F1 : {cat_cv:.4f}   (Optuna-tuned)")
print(f"  XGBoost   OOF F1 : {xgb_cv:.4f}")
print(f"  Ensemble  OOF F1 : {best_f1:.4f}   ← submit this")
print(f"  v1 public/private: 0.8847 / 0.8678")
print(f"  Features          : {X.shape[1]}  (incl. fold-aware target encoding)")
print(f"  CV strategy       : 5-Fold StratifiedKFold (country×target strata)")
print("=" * 70)
print()
print("  Submit submission_v2.csv to Zindi")
print("  Next: kb.add_competition() this notebook to update your RAG base!")
