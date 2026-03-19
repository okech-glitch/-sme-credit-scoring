import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path

app = FastAPI(
    title="Financial Health Engine",
    description="Live scoring API for the Zindi Alternative Credit Model (Top 1% Architecture)",
    version="1.0.0"
)

# ── Load Models ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "Financial Health Data" / "api_models"
if not MODEL_DIR.exists():
    raise RuntimeError("Models not found! Run export_api_models.py first.")

model_lgb = joblib.load(MODEL_DIR / "lgb.pkl")
model_cat = joblib.load(MODEL_DIR / "cat.pkl")
model_xgb = joblib.load(MODEL_DIR / "xgb.pkl")
label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
base_feat_cols = joblib.load(MODEL_DIR / "base_feat_cols.pkl")
te_info = joblib.load(MODEL_DIR / "target_encoding.pkl")
w_lgb, w_cat, w_xgb = joblib.load(MODEL_DIR / "ensemble_weights.pkl")

yn_map = {
    "yes": 1.0, "no": 0.0, "Yes": 1.0, "No": 0.0,
    "Have now": 2.0, "Used to have but don't have now": 1.0, "Never had": 0.0,
    "Don't know or N/A": np.nan, "Don't know": np.nan, "Refused": np.nan,
}

# ── Pydantic Schema ─────────────────────────────────────────────────────────
class BusinessProfile(BaseModel):
    country: str = Field(..., example="Kenya")
    owner_age: int = Field(..., example=35)
    personal_income: float = Field(..., example=50000)
    business_expenses: float = Field(..., example=20000)
    # The API can accept dynamic lists of string fields that the model knows about
    features: dict = Field(
        default={
            "bank_account": "Yes",
            "mobile_money": "Have now",
            "informal_lender": "No",
            "insurance": "Never had",
            "worried_about_business": "Yes"
        },
        description="Key-value mapping of categorical risk indicators"
    )

# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        return {"error": "index.html not found"}
    return FileResponse(html_path)

@app.post("/predict_health_index")
def predict_financial_health(profile: BusinessProfile):
    try:
        # 1. Parse into DataFrame using expected columns
        # Initialize an empty dataframe with the exact columns the model expects
        # Load the healthy baseline to prevent the model from panicking over 80+ missing features
        import json
        with open(MODEL_DIR / "baseline_profile.json", "r") as f:
            baseline = json.load(f)

        df = pd.DataFrame(columns=["country"] + base_feat_cols)
        row = baseline.copy()
        
        row["country"] = profile.country
        row["owner_age"] = profile.owner_age
        row["personal_income"] = profile.personal_income
        row["business_expenses"] = profile.business_expenses
        
        ui_feats = profile.features
        if "has_mobile_money" in row:
            row["has_mobile_money"] = ui_feats.get("mobile_money", row["has_mobile_money"])
            
        if "has_insurance" in row:
            row["has_insurance"] = ui_feats.get("insurance", row["has_insurance"])
            
        if ui_feats.get("bank_account") == "Yes":
            row["has_internet_banking"] = "Have now"
            row["has_debit_card"] = "Have now"
        elif ui_feats.get("bank_account") == "No":
            row["has_internet_banking"] = "Never had"
            row["has_debit_card"] = "Never had"
                
        df = pd.concat([df, pd.DataFrame([row])])
        df = df.fillna(np.nan)
        
        # 2. Base Feature Engineering
        str_cols = [c for c in df.columns if df[c].dtype == object and c not in ["country"]]
        for col in str_cols: 
            df[col] = df[col].map(yn_map)
            
        df["country_rank"] = df["country"].str.lower().map({"lesotho":0, "malawi":1, "zimbabwe":2, "eswatini":3}).fillna(1)
        
        # Risk Factors
        cr_cols = [c for c in df.columns if any(k in c for k in ["bank","credit","loan","borrow"])]
        df["factor_credit_access"] = df[cr_cols].mean(axis=1) if cr_cols else 0.0
        r_cols = [c for c in df.columns if any(k in c for k in ["insurance","savings","emergency"])]
        df["factor_resilience"] = df[r_cols].mean(axis=1) if r_cols else 0.0
        d_cols = [c for c in df.columns if any(k in c for k in ["debt","repay","default"])]
        df["factor_debt_burden"] = df[d_cols].mean(axis=1) if d_cols else 0.0
        s_cols = [c for c in df.columns if any(k in c for k in ["stable","worried","shutdown"])]
        df["factor_stability"] = df[s_cols].mean(axis=1) if s_cols else 0.0

        df["composite_fhi"] = df["factor_credit_access"]*0.3 + df["factor_resilience"]*0.3 + df["factor_debt_burden"]*0.2 + df["factor_stability"]*0.2
        df["age_group"] = pd.cut(df["owner_age"], bins=[0,25,35,50,70,999], labels=[0,1,2,3,4]).astype(float)
        df["income_expense_ratio"] = (df["personal_income"] / (df["business_expenses"].replace(0, np.nan) + 1)).clip(0, 100)
        df["log_income"] = np.log1p(df["personal_income"].clip(0))
        df["log_expenses"] = np.log1p(df["business_expenses"].clip(0))
        
        df["country_fhi"] = df["country_rank"] * df["composite_fhi"]
        df["country_credit"] = df["country_rank"] * df["factor_credit_access"]
        df["country_resilience"] = df["country_rank"] * df["factor_resilience"]

        # Missing indicators
        for c in df.select_dtypes(include="number").columns:
            if df[c].isnull().sum() > 0: df[f"{c}_missing"] = df[c].isnull().astype(int)
        df = df.fillna(-999)

        # Apply Label Encoders for categorical columns we mapped
        for c, le in label_encoders.items():
            if c in df.columns:
                # Handle unknown categories safely
                df[c] = df[c].astype(str).map(lambda x: x if x in le.classes_ else "-999")
                df[c] = le.transform(df[c])
        
        X_raw = df[base_feat_cols].values
        
        # 3. Target Encoding
        n_classes = 3
        te_test = np.zeros((1, n_classes))
        c_rank = int(df["country_rank"].iloc[0])
        for cls in range(n_classes):
             te_test[0, cls] = te_info["mapping"][cls].get(c_rank, te_info["global_mean"][cls])
        
        X = np.concatenate([X_raw, te_test], axis=1)

        # 4. Ensemble Inference
        p_lgb = model_lgb.predict(X)[0]
        p_cat = model_cat.predict_proba(X)[0]
        p_xgb = model_xgb.predict(xgb.DMatrix(X))[0]

        final_probs = w_lgb * p_lgb + w_cat * p_cat + w_xgb * p_xgb
        
        prob_low = float(final_probs[0])
        prob_med = float(final_probs[1])
        prob_high = float(final_probs[2])
        
        # Calibrating imbalanced priors using Business Logic Thresholds
        if prob_low >= 0.25:
            pred_idx = 0  # Declined
        elif prob_low <= 0.20:
            pred_idx = 2  # Approved
        else:
            pred_idx = 1  # Manual Review
        
        labels = {0: "Low (High Default Risk)", 1: "Medium (Moderate Risk)", 2: "High (Prime Reliability)"}

        return {
            "status": "success",
            "prediction": labels[pred_idx],
            "risk_adjusted_probabilities": {
                "Low": round(float(final_probs[0]), 3),
                "Medium": round(float(final_probs[1]), 3),
                "High": round(float(final_probs[2]), 3)
            },
            "architect_insight": f"Based on {profile.country}'s macro-regime, this SME has a FHI index weighting primarily towards '{labels[pred_idx]}'."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
