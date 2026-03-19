"""
ZINDI GRANDMASTER CREW - Multi-Agent Competition System
Based on Koleshjr's proven winning patterns from 20+ competitions
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from .env file (must be called before importing LLM clients)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import pandas as pd
import numpy as np

# ─── LLM Selection ────────────────────────────────────────────────────────────
# Priority: Anthropic → OpenAI → fail loudly (Ollama handled inside ZindiGrandmasterCrew)
def _pick_llm():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key    = os.getenv("OPENAI_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku")
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

    if use_local:
        from langchain_community.chat_models import ChatOllama
        print("🦙 Using local Ollama (llama3)")
        return ChatOllama(model="llama3", temperature=0.1)
    elif openrouter_key and openrouter_key != "your_openrouter_key_here":
        print(f"🌐 Using OpenRouter → {openrouter_model}")
        return ChatOpenAI(
            model=openrouter_model,
            temperature=0.1,
            openai_api_key=openrouter_key,
            openai_api_base=openrouter_url,
        )
    elif anthropic_key and anthropic_key != "your_anthropic_key_here":
        print("🤖 Using Claude claude-sonnet-4-5")
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=0.1)
    elif openai_key and openai_key != "your_openai_key_here":
        print("🤖 Using GPT-4o-mini")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    else:
        raise EnvironmentError(
            "❌ No LLM configured!\n"
            "Fix: Edit .env and set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY\n"
            "  OR set USE_LOCAL_LLM=true in .env and install Ollama"
        )

llm = _pick_llm()


class ZindiGrandmasterCrew:
    """
    4-Agent system based on proven Zindi winning patterns:
    - Agent 1: EDA Analyst (identifies data patterns)
    - Agent 2: Feature Factory (target encoding + rolling features)
    - Agent 3: Model Zoo (CatBoost + LightGBM ensemble)
    - Agent 4: Ensemble & Validator (stacking + CV scoring)
    """

    def __init__(self, dataset_path, target_column, problem_type="regression"):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.problem_type = problem_type
        self.df = pd.read_csv(dataset_path)

    def create_agents(self):
        """Create 4 specialized agents based on winning competition patterns"""

        # AGENT 1: EDA Analyst - Like your Traffic Forecasting exploratory phase
        eda_agent = Agent(
            role='Zindi EDA Specialist',
            goal=f'Analyze {self.dataset_path} and identify patterns that separate winners from top-30%',
            backstory="""You are a Zindi Grandmaster with 20+ top-3 finishes.
            You NEVER trust initial impressions. You always check:
            - Target distribution (skewness, need for sqrt/log transform?)
            - Missing value patterns (MAR vs MCAR)
            - Temporal leakage (future info in training?)
            - Categorical cardinality (which need target encoding?)
            - Correlation with target (which features are gold?)

            You output ONLY actionable insights, no fluff.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # AGENT 2: Feature Factory - Your proven engineering patterns
        feature_agent = Agent(
            role='Zindi Feature Engineering Grandmaster',
            goal='Generate features using PROVEN Zindi winning patterns only',
            backstory="""You are Koleshjr's feature engineering brain.

            YOUR PROVEN TOOLKIT (from 15+ winning solutions):

            1. TARGET ENCODING (fold-aware, prevents leakage):
               - Mean, std, skewness, min, max, 25th/75th percentiles
               - Always group by multiple categorical combinations
               - NEVER encode test set with test statistics

            2. ROLLING FEATURES (for time series):
               - Windows: 168h (1 week), 336h (2 weeks)
               - Statistics: mean, median, std, quantiles
               - ALWAYS shift to prevent leakage

            3. EXPANDING FEATURES (cumulative stats):
               - Multiple shift periods (168, 336, 504 hours)
               - Captures long-term evolution

            4. INTERACTION FEATURES:
               - Ratio features (X/Y where correlated)
               - Product features (X*Y for categorical combos)
               - Difference features (X-Y for competing metrics)

            5. DOMAIN-SPECIFIC:
               - Agricultural: NDVI, EVI for satellite
               - Time series: lag features, EWMA
               - Transaction: recency, frequency, monetary

            You NEVER suggest RandomForest.
            You ALWAYS output exact pandas/polars code.
            You ALWAYS prevent leakage with proper CV-aware encoding.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # AGENT 3: Model Zoo - Your proven GBDT stack
        model_agent = Agent(
            role='Zindi GBDT Modeling Specialist',
            goal='Build CatBoost + LightGBM ensemble with proven hyperparameters',
            backstory="""You are the modeling brain behind 15+ Zindi GBDT wins.

            YOUR DEFAULT STACK (proven in Traffic Forecasting 1st place):

            1. LightGBM:
               - learning_rate: 0.02-0.08
               - num_leaves: 150-254
               - max_depth: 9-10
               - feature_fraction: 0.65-0.75
               - bagging_fraction: 0.72-0.94
               - lambda_l1: 1e-6 to 1e-1
               - n_estimators: 1000-5000

            2. CatBoost:
               - learning_rate: 0.02
               - depth: 9
               - l2_leaf_reg: 1e-6
               - bagging_temperature: 0.38
               - iterations: 15000

            3. CROSS-VALIDATION:
               - 10-fold StratifiedKFold (stratify on important categorical)
               - For time series: TimeSeriesSplit with gap
               - ALWAYS save OOF predictions for meta-models

            4. ENSEMBLE WEIGHTS (from competition history):
               - 60% CatBoost + 40% LightGBM (Traffic Forecasting)
               - 50% each (Farm to Feed)
               - Test both, pick by CV score

            You NEVER use default hyperparameters.
            You ALWAYS apply sqrt/log transform if target is skewed.
            You ALWAYS track feature importance.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # AGENT 4: Ensemble & Validator - Final submission optimizer
        ensemble_agent = Agent(
            role='Zindi Ensemble Architect',
            goal='Create optimal ensemble and validate no leakage exists',
            backstory="""You are the final validator in a Zindi winning pipeline.

            YOUR PROVEN ENSEMBLE TECHNIQUES:

            1. WEIGHTED AVERAGING:
               - Test weights: 0.5/0.5, 0.6/0.4, 0.7/0.3
               - Pick by best OOF MAE/RMSE

            2. STACKING (if time permits):
               - Level 1: LightGBM + CatBoost + TabNet
               - Level 2: Ridge/Lasso on OOF predictions

            3. VALIDATION CHECKLIST:
               - CV score matches public LB? (±5% is normal)
               - Feature importance makes domain sense?
               - No future leakage in time series?
               - No test statistics in train encoding?

            4. SHAP ANALYSIS:
               - Top 10 features explainable?
               - Any suspicious perfect predictors?

            You NEVER submit without CV validation.
            You ALWAYS check for leakage before final submission.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        return eda_agent, feature_agent, model_agent, ensemble_agent

    def create_tasks(self, agents):
        """Create tasks that mirror your competition workflow"""
        eda_agent, feature_agent, model_agent, ensemble_agent = agents

        # TASK 1: EDA Analysis
        eda_task = Task(
            description=f"""
            Analyze the dataset at {self.dataset_path}.
            Target column: {self.target_column}
            Problem type: {self.problem_type}

            Output a Python dict with:
            1. target_distribution: {{skewness, need_transform, suggested_transform}}
            2. missing_patterns: {{columns_with_missing, percentage, pattern_type}}
            3. categorical_features: {{high_cardinality_cols, need_target_encoding}}
            4. temporal_features: {{is_time_series, temporal_columns, leak_risk}}
            5. top_correlations: {{top_5_correlated_features}}
            6. recommended_cv_strategy: {{cv_type, n_splits, stratify_on}}

            Base insights on Zindi winning patterns.
            """,
            agent=eda_agent,
            expected_output="Python dictionary with EDA insights"
        )

        # TASK 2: Feature Engineering
        feature_task = Task(
            description="""
            Based on EDA results, generate feature engineering code.

            MUST INCLUDE (based on proven patterns):
            1. Target encoding (fold-aware) on identified categorical features
            2. Interaction features (top 15 combinations by correlation)
            3. If time series: rolling features (168h, 336h windows)
            4. If time series: expanding features (shift 168, 336, 504)
            5. Domain-specific features (based on problem type)

            Output EXACT pandas code that:
            - Prevents leakage (uses only train data for encoding)
            - Is copy-paste ready
            - Includes comments explaining each feature
            - Returns engineered train and test DataFrames

            Format: ```python
            # Feature Engineering Code
            def create_features(train, test, target_col):
                # ... your code ...
                return train_fe, test_fe
            ```
            """,
            agent=feature_agent,
            expected_output="Python function with feature engineering code",
            context=[eda_task]
        )

        # TASK 3: Model Training
        model_task = Task(
            description="""
            Generate training code for CatBoost + LightGBM ensemble.

            REQUIREMENTS:
            1. Use proven hyperparameters from Traffic Forecasting competition
            2. Implement 10-fold StratifiedKFold (or appropriate CV from EDA)
            3. Apply sqrt transform if target is skewed (from EDA)
            4. Save OOF predictions for ensembling
            5. Track feature importance

            Output EXACT code:
            ```python
            from sklearn.model_selection import StratifiedKFold
            from lightgbm import LGBMRegressor
            from catboost import CatBoostRegressor
            import numpy as np

            def train_models(X_train, y_train, X_test):
                # LightGBM with proven params
                # CatBoost with proven params
                # 10-fold CV
                # Return: lgb_preds, cat_preds, oof_scores, feature_importance
                pass
            ```
            """,
            agent=model_agent,
            expected_output="Python function with model training code",
            context=[eda_task, feature_task]
        )

        # TASK 4: Ensemble & Validation
        ensemble_task = Task(
            description="""
            Create final ensemble and validate results.

            TASKS:
            1. Test ensemble weights: [0.5, 0.5], [0.6, 0.4], [0.4, 0.6]
            2. Pick best by OOF MAE/RMSE
            3. Run SHAP on final model to explain top 10 features
            4. Validate no leakage (check feature importance for red flags)
            5. Generate submission file

            Output:
            ```python
            def create_ensemble(lgb_preds, cat_preds, oof_lgb, oof_cat, y_true):
                # Test different weights
                # Return: best_weight, final_predictions, validation_score
                pass

            def validate_and_submit(final_preds, feature_importance):
                # SHAP analysis
                # Leakage check
                # Create submission.csv
                pass
            ```

            Also output a summary:
            - Expected CV score
            - Top 10 most important features with SHAP explanation
            - Any red flags or warnings
            - Recommended next steps if score is not competitive
            """,
            agent=ensemble_agent,
            expected_output="Ensemble code + validation summary",
            context=[eda_task, feature_task, model_task]
        )

        return [eda_task, feature_task, model_task, ensemble_task]

    @staticmethod
    def save_outputs(result, output_dir: str = "crew_output"):
        """Save agent-generated code blocks to individual .py files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result_text = str(result)

        # Save full output
        full_out = output_path / "full_crew_output.txt"
        full_out.write_text(result_text, encoding="utf-8")
        print(f"💾 Full output → {full_out}")

        # Extract code blocks
        code_blocks = []
        if "```python" in result_text:
            parts = result_text.split("```python")
            for part in parts[1:]:
                code = part.split("```")[0].strip()
                if code:
                    code_blocks.append(code)

        labels = ["feature_engineering", "model_training", "ensemble_validation"]
        for i, code in enumerate(code_blocks):
            label = labels[i] if i < len(labels) else f"step_{i+1}"
            file_path = output_path / f"{label}.py"
            file_path.write_text(code, encoding="utf-8")
            print(f"💾 Code block {i+1} → {file_path}")

        print(f"\n✅ All outputs saved to: {output_path}/")
        return output_path

    def run(self):
        """Execute the full agent workflow"""
        print("🚀 Starting Zindi Grandmaster Crew...")
        print(f"📊 Dataset: {self.dataset_path}")
        print(f"🎯 Target: {self.target_column}")
        print(f"🏆 Problem: {self.problem_type}\n")

        agents = self.create_agents()
        tasks = self.create_tasks(agents)

        crew = Crew(
            agents=list(agents),
            tasks=tasks,
            process=Process.sequential,  # EDA → Features → Models → Ensemble
            verbose=True
        )

        result = crew.kickoff()

        print("\n✅ Crew execution complete!")
        print("\n" + "=" * 80)
        print("FINAL OUTPUT:")
        print("=" * 80)
        print(result)

        return result


# ─── USAGE EXAMPLE ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Replace with your actual Zindi competition data
    crew_system = ZindiGrandmasterCrew(
        dataset_path="train.csv",   # ← your training CSV
        target_column="target",     # ← your target column name
        problem_type="regression"   # ← "regression" or "classification"
    )

    result = crew_system.run()

    # Save all generated code files
    ZindiGrandmasterCrew.save_outputs(result, output_dir="crew_output")
    print("\n🎯 Copy code blocks from crew_output/ and paste into your notebook!")
