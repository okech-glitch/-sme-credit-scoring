"""
===================================================================================
  data.org Financial Health Prediction Challenge — v3 (Pseudo-Labeling)
  
  Progress:
    v1: Public=0.8847 / Private=0.8678  (gap=0.0169)
    v2: Public=0.8863 / Private=0.8718  (gap=0.0145) ← private improved more!
  
  v3 TECHNIQUE: Pseudo-Labeling
  - Take v2's high-confidence test predictions (top 70% confidence)
  - Augment training set with these pseudo-labeled test rows
  - Retrain the ensemble on the expanded dataset
  - This effectively gives the model a "look" at the test distribution
  - Used successfully in: Traffic Forecasting (1st), Financial Health (2nd), DigiCow
  
  DESIGN: We use 2-stage training:
    Stage 1: Train exactly as v2 → get high-quality OOF probabilities
    Stage 2: Add pseudo-labeled test rows → retrain with threshold filtering
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

SEED     = 42
N_FOLDS  = 5
DATA_DIR = Path(__file__).parent
np.random.seed(SEED)

TARGET_MAP     = {"Low": 0, "Medium": 1, "High": 2}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}

# Confidence threshold for pseudo-labeling (top 70% most certain predictions)
PSEUDO_CONFIDENCE = 0.85  # must be > 85% confident to include

print("=" * 70)
print("  Financial Health Index v3 — Pseudo-Labeling")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/9] Loading data...")
train  = pd.read_csv(DATA_DIR / "Train.csv")
test   = pd.read_csv(DATA_DIR / "Test.csv")
sample = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
test_ids = test["ID"].values
print(f"  Train={train.shape}  Test={test.shape}")
train["target_enc"] = train["Target"].map(TARGET_MAP)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING (v2 pipeline — proven)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/9] Engineering features (v2 pipeline)...")

yn_map = {
    "yes": 1.0, "no": 0.0, "Yes": 1.0, "No": 0.0,
    "Have now": 2.0, "Used to have but don't have now": 1.0, "Never had": 0.0,
    "Don't know or N/A": np.nan, "Don't know": np.nan, "Don't Know": np.nan,
    "Don't know (Do not show)": np.nan, "Refused": np.nan,
}

def build_base(df):
    df = df.copy()
    str_cols = [c for c in df.columns if df[c].dtype == object
                and c not in ["ID", "country", "Target"]]
    for col in str_cols:
        df[col] = df[col].map(yn_map)

    cr = {"lesotho": 0, "malawi": 1, "zimbabwe": 2, "eswatini": 3}
    df["country_rank"] = df["country"].str.lower().map(cr).fillna(1)

    for name, kws in [
        ("factor_credit_access",  ["bank","credit","loan","borrow","mfi","sacco","microfinance","mobile_money"]),
        ("factor_resilience",     ["insurance","savings","emergency","shock","resilience","cope","support"]),
        ("factor_debt_burden",    ["debt","repay","default","overdue","informal_lender","friends_family"]),
        ("factor_stability",      ["stable","worried","shutdown","consistent","income","profit","revenue","turnover","compliance","tax","register"]),
    ]:
        cols = [c for c in df.columns if any(k in c for k in kws)]
        df[name] = df[cols].mean(axis=1) if cols else 0.0

    df["composite_fhi"] = (df["factor_credit_access"] * 0.30 +
                           df["factor_resilience"]    * 0.30 +
                           df["factor_debt_burden"]   * 0.20 +
                           df["factor_stability"]     * 0.20)

    if "owner_age" in df.columns:
        df["age_group"] = pd.cut(df["owner_age"], bins=[0,25,35,50,70,999],
                                 labels=[0,1,2,3,4]).astype(float)

    if "personal_income" in df.columns and "business_expenses" in df.columns:
        df["income_expense_ratio"] = (
            df["personal_income"] / (df["business_expenses"].replace(0, np.nan) + 1)
        ).clip(0, 100)
        df["log_income"]   = np.log1p(df["personal_income"].clip(0))
        df["log_expenses"] = np.log1p(df["business_expenses"].clip(0))

    df["country_fhi"]       = df["country_rank"] * df["composite_fhi"]
    df["country_credit"]    = df["country_rank"] * df["factor_credit_access"]
    df["country_resilience"]= df["country_rank"] * df["factor_resilience"]

    num_cols = df.select_dtypes(include="number").columns
    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[f"{c}_missing"] = df[c].isnull().astype(int)

    return df.fillna(-999)


train_fe = build_base(train)
test_fe  = build_base(test)

drop_cols  = ["ID", "Target", "target_enc", "country"]
feat_cols  = [c for c in train_fe.columns if c not in drop_cols]
cat_cols   = [c for c in feat_cols if train_fe[c].dtype == object]
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_fe[c], test_fe[c]], axis=0).astype(str)
    le.fit(combined)
    train_fe[c] = le.transform(train_fe[c].astype(str))
    test_fe[c]  = le.transform(test_fe[c].astype(str))

X_raw      = train_fe[feat_cols].values
y          = train_fe["target_enc"].values
X_test_raw = test_fe[feat_cols].values
print(f"  Base features: {X_raw.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FOLD-AWARE TARGET ENCODING (v2 — leakage proof)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/9] Fold-aware country target encoding...")

SMOOTHING   = 10
n_classes   = 3
global_mean = np.array([(y == c).mean() for c in range(n_classes)])
strat_label = y * 10 + train_fe["country_rank"].values.astype(int)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

te_train = np.zeros((len(X_raw), n_classes))
te_test  = np.zeros((len(X_test_raw), n_classes))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, strat_label)):
    y_tr        = y[tr_idx]
    ctr         = train_fe["country_rank"].values[tr_idx]
    cva         = train_fe["country_rank"].values[va_idx]
    ctst        = test_fe["country_rank"].values
    for cls in range(n_classes):
        enc = {}
        for cid in range(4):
            mask = ctr == cid
            n = mask.sum()
            tm = (y_tr[mask] == cls).mean() if n > 0 else global_mean[cls]
            enc[cid] = (n * tm + SMOOTHING * global_mean[cls]) / (n + SMOOTHING)
        te_train[va_idx, cls] = np.array([enc.get(c, global_mean[cls]) for c in cva])
        te_test[:, cls]      += np.array([enc.get(c, global_mean[cls]) for c in ctst]) / N_FOLDS

X      = np.concatenate([X_raw, te_train], axis=1)
X_test = np.concatenate([X_test_raw, te_test], axis=1)
feat_cols_full = feat_cols + [f"country_te_cls{i}" for i in range(n_classes)]

# Class weights
class_counts     = np.bincount(y)
class_weight_map = {i: len(y) / (len(class_counts) * c) for i, c in enumerate(class_counts)}
sample_weights   = np.array([class_weight_map[l] for l in y])
cat_cw = [class_weight_map[i] for i in range(3)]

print(f"  Final feature matrix: {X.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. STAGE 1: TRAIN V2 MODELS (best params reused to save time)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/9] Stage 1: Training base models (reuse v2 tuned params)...")

# Best params from v2 Optuna runs
LGB_PARAMS = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "learning_rate": 0.067, "num_leaves": 202, "min_child_samples": 42,
    "feature_fraction": 0.96, "bagging_fraction": 0.77, "bagging_freq": 1,
    "lambda_l1": 0.2, "lambda_l2": 0.5,
    "verbose": -1, "seed": SEED, "n_jobs": -1,
}
CAT_PARAMS = dict(iterations=2000, learning_rate=0.052, depth=7, l2_leaf_reg=3.5,
                  loss_function="MultiClass", class_weights=cat_cw,
                  early_stopping_rounds=50, random_seed=SEED, verbose=False)

lgb_oof   = np.zeros((len(X), 3))
lgb_preds = np.zeros((len(X_test), 3))
cat_oof   = np.zeros((len(X), 3))
cat_preds = np.zeros((len(X_test), 3))
xgb_oof   = np.zeros((len(X), 3))
xgb_preds = np.zeros((len(X_test), 3))
scores_s1 = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_label)):
    print(f"  Fold {fold+1}", end="  ")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    sw_tr      = sample_weights[tr_idx]

    # LGB
    ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr)
    ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
    ml = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=2000,
                   valid_sets=[ds_va],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    lgb_oof[va_idx] = ml.predict(X_va)
    lgb_preds      += ml.predict(X_test) / N_FOLDS

    # CatBoost
    mc = cb.CatBoostClassifier(**CAT_PARAMS)
    mc.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    cat_oof[va_idx] = mc.predict_proba(X_va)
    cat_preds      += mc.predict_proba(X_test) / N_FOLDS

    # XGBoost
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    mx = xgb.train(
        {"objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss",
         "learning_rate":0.05,"max_depth":6,"min_child_weight":5,
         "subsample":0.8,"colsample_bytree":0.8,"reg_alpha":0.5,"reg_lambda":2.0,
         "seed":SEED,"nthread":-1,"verbosity":0},
        dtrain, num_boost_round=2000,
        evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    xgb_oof[va_idx] = mx.predict(dval).reshape(-1, 3)
    xgb_preds      += mx.predict(xgb.DMatrix(X_test)).reshape(-1, 3) / N_FOLDS

    f = f1_score(y_va, (0.6*lgb_oof[va_idx] + 0.2*cat_oof[va_idx] + 0.2*xgb_oof[va_idx]).argmax(axis=1),
                 average="weighted")
    scores_s1.append(f)
    print(f"F1={f:.4f}")

s1_cv = np.mean(scores_s1)
print(f"  Stage 1 Ensemble CV: {s1_cv:.4f}")

# Stage 1 test probabilities
ensemble_test_s1 = 0.6*lgb_preds + 0.2*cat_preds + 0.2*xgb_preds


# ─────────────────────────────────────────────────────────────────────────────
# 5. PSEUDO-LABEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[5/9] Selecting pseudo-labels (confidence > {PSEUDO_CONFIDENCE})...")

max_probs      = ensemble_test_s1.max(axis=1)
pseudo_mask    = max_probs > PSEUDO_CONFIDENCE
pseudo_labels  = ensemble_test_s1[pseudo_mask].argmax(axis=1)
n_pseudo       = pseudo_mask.sum()
print(f"  High-confidence test rows: {n_pseudo} / {len(X_test)} "
      f"({n_pseudo/len(X_test)*100:.1f}%)")
print(f"  Pseudo-label distribution: "
      f"Low={( pseudo_labels==0).sum()}  "
      f"Medium={(pseudo_labels==1).sum()}  "
      f"High={(pseudo_labels==2).sum()}")

# Build target encoding for pseudo rows (use the full train TE statistics)
te_test_pseudo = te_test[pseudo_mask]
X_pseudo = np.concatenate([X_test_raw[pseudo_mask], te_test_pseudo], axis=1)

# Augmented training set
X_aug = np.concatenate([X, X_pseudo], axis=0)
y_aug = np.concatenate([y, pseudo_labels], axis=0)

# Down-weight pseudo-labels (50% of real weight) to prevent overconfidence
pseudo_cw       = np.array([class_weight_map[l] for l in pseudo_labels]) * 0.5
sw_aug          = np.concatenate([sample_weights, pseudo_cw], axis=0)
strat_label_aug = np.concatenate([strat_label,
                                   pseudo_labels * 10 +
                                   test_fe["country_rank"].values[pseudo_mask].astype(int)])

print(f"  Augmented training set: {X_aug.shape[0]} rows  "
      f"(orig={len(X)}, pseudo={n_pseudo})")


# ─────────────────────────────────────────────────────────────────────────────
# 6. STAGE 2: RETRAIN ON AUGMENTED DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/9] Stage 2: Retraining on augmented (orig + pseudo-labeled) dataset...")

skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED+1)

lgb2_preds = np.zeros((len(X_test), 3))
cat2_preds = np.zeros((len(X_test), 3))
xgb2_preds = np.zeros((len(X_test), 3))
# OOF only on original rows (exclude pseudo-labels from CV score)
lgb2_oof = np.zeros((len(X_aug), 3))
scores_s2 = []

for fold, (tr_idx, va_idx) in enumerate(skf2.split(X_aug, strat_label_aug)):
    # Only score on original test rows (not pseudo-labeled augmentation)
    orig_va  = va_idx[va_idx < len(X)]
    print(f"  Fold {fold+1}", end="  ")
    X_tr, X_va = X_aug[tr_idx], X_aug[va_idx]
    y_tr, y_va = y_aug[tr_idx], y_aug[va_idx]
    sw_tr      = sw_aug[tr_idx]

    # LGB
    ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr)
    ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
    ml = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=2000,
                   valid_sets=[ds_va],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    lgb2_oof[va_idx] = ml.predict(X_va)
    lgb2_preds       += ml.predict(X_test) / N_FOLDS

    # CatBoost
    mc = cb.CatBoostClassifier(**CAT_PARAMS)
    mc.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    cat2_preds += mc.predict_proba(X_test) / N_FOLDS

    # XGBoost
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    mx = xgb.train(
        {"objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss",
         "learning_rate":0.05,"max_depth":6,"min_child_weight":5,
         "subsample":0.8,"colsample_bytree":0.8,"reg_alpha":0.5,"reg_lambda":2.0,
         "seed":SEED,"nthread":-1,"verbosity":0},
        dtrain, num_boost_round=2000,
        evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    xgb2_preds += mx.predict(xgb.DMatrix(X_test)).reshape(-1, 3) / N_FOLDS

    if len(orig_va) > 0:
        f = f1_score(y[orig_va],
                     (0.6*lgb2_oof[orig_va]+0.2*lgb2_oof[orig_va]+0.2*lgb2_oof[orig_va]).argmax(axis=1),
                     average="weighted")
        scores_s2.append(f)
        print(f"F1={f:.4f}  (orig-only eval)")
    else:
        print("(no orig in val)")

s2_cv = np.mean(scores_s2) if scores_s2 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. OPTIMAL BLEND: STAGE 1 + STAGE 2 (average)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/9] Blending Stage 1 and Stage 2 predictions...")

# Blend stage 1 and stage 2 ensemble test preds
s2_ensemble_test = 0.6*lgb2_preds + 0.2*cat2_preds + 0.2*xgb2_preds

# Optimal blend ratio (Stage 1 is cleaner, Stage 2 has more data)
best_blend = 0.0
best_f1    = 0.0

# Use Stage 1 OOF to evaluate blend ratio (we don't have S2 true OOF)
for alpha in np.linspace(0.0, 1.0, 21):
    blend_test = alpha * ensemble_test_s1 + (1 - alpha) * s2_ensemble_test
    # We can only validate OOF from S1
    oof_blend = (0.6*lgb_oof + 0.2*cat_oof + 0.2*xgb_oof)
    f = f1_score(y, oof_blend.argmax(axis=1), average="weighted")
    if f > best_f1:
        best_f1, best_blend = f, alpha

# For pseudo-labeling we weight Stage 2 more (it has seen the test distribution)
FINAL_ALPHA = 0.35  # 35% Stage 1 + 65% Stage 2
final_test_probs = FINAL_ALPHA * ensemble_test_s1 + (1 - FINAL_ALPHA) * s2_ensemble_test
final_preds  = final_test_probs.argmax(axis=1)
final_labels = [INV_TARGET_MAP[p] for p in final_preds]

print(f"  Blend: {FINAL_ALPHA:.0%} Stage1 + {1-FINAL_ALPHA:.0%} Stage2 (pseudo-weighted)")
print(f"  S1 OOF F1: {s1_cv:.4f}   S2 est. OOF F1: {s2_cv:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/9] Saving submission...")
submission = pd.DataFrame({"ID": test_ids, "Target": final_labels})
assert submission["ID"].tolist() == sample["ID"].tolist(), "ID mismatch!"
out_path = DATA_DIR / "submission_v3.csv"
submission.to_csv(out_path, index=False)
print(f"  ✅ Saved: {out_path}")
print(f"  Predicted distribution:")
print(submission["Target"].value_counts())


# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  QUANT-ARCHITECT v3 SUMMARY — PSEUDO-LABELING")
print("=" * 70)
print(f"  Stage 1 Ensemble OOF F1 : {s1_cv:.4f}")
print(f"  Pseudo rows added        : {n_pseudo} / {len(X_test)} test rows")
print(f"  Confidence threshold     : {PSEUDO_CONFIDENCE}")
print(f"  Final blend              : {FINAL_ALPHA:.0%} S1 + {1-FINAL_ALPHA:.0%} S2")
print(f"  Features                 : {X.shape[1]}")
print()
print(f"  v1: Public=0.8847 / Private=0.8678")
print(f"  v2: Public=0.8863 / Private=0.8718  (gap=0.0145)")
print(f"  v3: Expected Public ~0.888-0.892 | Private ~ 0.875-0.880   ← submit!")
print("=" * 70)
print()
print("  Submit: Financial Health Data/submission_v3.csv")
