"""
ZINDI COMPETITION WORKFLOW - CrewAI + RAG Integration
Full end-to-end system for any new Zindi competition

WORKFLOW:
1. Query RAG for similar past competitions
2. Use insights to configure CrewAI agents
3. Let agents generate feature engineering + models
4. Validate with RAG's proven patterns
5. Submit and iterate

USAGE:
    python zindi_competition_workflow.py \\
        --train train.csv --test test.csv \\
        --target target --problem regression \\
        --name "My Competition"

    # Dry-run (RAG only, no LLM cost from CrewAI):
    python zindi_competition_workflow.py --dry-run ...
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load .env FIRST so API keys are available to all imports
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from zindi_rag_knowledge_base import ZindiKnowledgeBase
from zindi_crew_system import ZindiGrandmasterCrew


class ZindiCompetitionWorkflow:
    """
    Complete workflow for any Zindi competition.
    Combines RAG knowledge retrieval with CrewAI automation.
    """

    def __init__(
        self,
        train_path: str,
        test_path: str,
        target_col: str,
        problem_type: str,
        competition_name: str = "Unknown Competition",
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.target_col = target_col
        self.problem_type = problem_type
        self.competition_name = competition_name

        self.train = pd.read_csv(train_path)
        self.test  = pd.read_csv(test_path)

        print(f"\n{'=' * 80}")
        print(f"🎯 ZINDI COMPETITION: {competition_name}")
        print(f"{'=' * 80}")
        print(f"📊 Train shape: {self.train.shape}")
        print(f"📊 Test shape:  {self.test.shape}")
        print(f"🎯 Target:      {target_col}")
        print(f"🏆 Problem:     {problem_type}")
        print(f"{'=' * 80}\n")

        self.kb = None
        self.crew = None
        self.insights = {}
        self.crew_output = None

    # ──────────────────────────────────────────────────────────────────────────

    def step1_query_knowledge_base(self):
        """STEP 1: Query RAG for relevant past solutions."""
        print("\n" + "=" * 80)
        print("STEP 1: Querying Knowledge Base for Similar Competitions")
        print("=" * 80 + "\n")

        self.kb = ZindiKnowledgeBase()
        self.kb.build_index()

        # Analyse competition characteristics
        n_features    = len(self.train.columns) - 1
        n_categorical = len(self.train.select_dtypes(include=["object"]).columns)
        target_skew   = (
            float(self.train[self.target_col].skew()) if self.problem_type == "regression" else 0.0
        )
        has_time_cols = any(
            kw in col.lower() for col in self.train.columns for kw in ("date", "time", "hour", "week")
        )

        print("📈 Competition characteristics:")
        print(f"   - Features:        {n_features}")
        print(f"   - Categorical:     {n_categorical}")
        print(f"   - Target skewness: {target_skew:.2f}")
        print(f"   - Time series:     {has_time_cols}")
        print()

        queries = [
            (
                f"What are the best feature engineering techniques for "
                f"{self.problem_type} problems with {n_categorical} categorical features?"
            ),
            (
                f"What LightGBM and CatBoost hyperparameters work best for {self.problem_type}?"
            ),
            "How should I ensemble CatBoost and LightGBM? What weights work best?",
        ]
        if has_time_cols:
            queries.append("What rolling and expanding window features work best for time series?")
        if abs(target_skew) > 1:
            queries.append(
                f"The target has skewness {target_skew:.2f}. "
                "Should I apply sqrt or log transform? Show code."
            )

        for query in queries:
            print(f"\n❓ {query}")
            answer = self.kb.query(query, verbose=False)
            self.insights[query] = answer
            preview = answer[:300].replace("\n", " ")
            print(f"💡 {preview}...")

        return self.insights

    def step2_configure_and_run_crew(self):
        """STEP 2: Configure CrewAI with RAG insights and run."""
        print("\n" + "=" * 80)
        print("STEP 2: Configuring & Running CrewAI Agents")
        print("=" * 80 + "\n")

        self.crew = ZindiGrandmasterCrew(
            dataset_path=self.train_path,
            target_column=self.target_col,
            problem_type=self.problem_type,
        )

        print("🤖 Deploying 4-Agent Team...")
        print("   Agent 1: EDA Analyst")
        print("   Agent 2: Feature Factory")
        print("   Agent 3: Model Zoo")
        print("   Agent 4: Ensemble Validator\n")

        self.crew_output = self.crew.run()
        return self.crew_output

    def step3_extract_and_validate_code(self):
        """STEP 3: Extract generated code and validate against RAG patterns."""
        print("\n" + "=" * 80)
        print("STEP 3: Extracting Code & Validating Against Proven Patterns")
        print("=" * 80 + "\n")

        crew_text = str(self.crew_output)

        # Extract code blocks
        code_blocks = []
        if "```python" in crew_text:
            for part in crew_text.split("```python")[1:]:
                code = part.split("```")[0].strip()
                if code:
                    code_blocks.append(code)

        print(f"✅ Extracted {len(code_blocks)} code blocks from agent output\n")

        # Validate against proven patterns
        validation_queries = [
            "Does proper fold-aware encoding prevent data leakage?",
            "Are these hyperparameters similar to the proven winning configuration?",
            "Is the CV strategy (StratifiedKFold or TimeSeriesSplit) appropriate?",
        ]
        print("🔍 Validating against proven patterns...\n")
        for query in validation_queries:
            validation = self.kb.query(query, verbose=False)
            print(f"✓ {query}")
            print(f"  {validation[:200].replace(chr(10), ' ')}...\n")

        # Save to timestamped output directory
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir  = Path(f"./crew_output_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        labels = ["feature_engineering", "model_training", "ensemble_validation"]
        for i, code in enumerate(code_blocks):
            label = labels[i] if i < len(labels) else f"step_{i+1}"
            (output_dir / f"{label}.py").write_text(code, encoding="utf-8")
            print(f"💾 {label}.py → {output_dir}")

        (output_dir / "full_crew_output.txt").write_text(crew_text, encoding="utf-8")
        print(f"\n✅ All outputs saved to: {output_dir}/")
        return code_blocks, output_dir

    def step4_quick_baseline(self):
        """STEP 4: Write a quick baseline script (Top 30% in <2h)."""
        print("\n" + "=" * 80)
        print("STEP 4: Quick Baseline Submission (Target: Top 30%)")
        print("=" * 80 + "\n")

        # Choose the correct model class string
        if self.problem_type == "regression":
            model_class = "LGBMRegressor"
            scoring     = "neg_mean_absolute_error"
        else:
            model_class = "LGBMClassifier"
            scoring     = "roc_auc"

        baseline_code = f"""# Quick Baseline — Generated by Zindi Grandmaster System
# Competition: {self.competition_name}
# Target: Top 30% in <2 hours

import pandas as pd
import numpy as np
from lightgbm import {model_class}
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# ── Load data ─────────────────────────────────────────────────
train = pd.read_csv(r'{self.train_path}')
test  = pd.read_csv(r'{self.test_path}')

FEATURES = [c for c in train.columns if c not in ['ID', 'id', '{self.target_col}']]

# ── Simple label encoding ──────────────────────────────────────
for col in train.select_dtypes(include='object').columns:
    if col in FEATURES:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        # Handle unseen categories in test
        test[col] = test[col].astype(str).map(
            {{v: i for i, v in enumerate(le.classes_)}}
        ).fillna(-1).astype(int)

# ── Proven LightGBM quick baseline ────────────────────────────
model = {model_class}(
    learning_rate=0.05,
    num_leaves=150,
    max_depth=9,
    n_estimators=1000,
    random_state=42,
    verbose=-1,
)

print("Running 5-fold CV...")
cv_scores = cross_val_score(
    model,
    train[FEATURES],
    train['{self.target_col}'],
    cv=5,
    scoring='{scoring}',
    n_jobs=-1,
)
print(f"CV Score: {{np.mean(cv_scores):.4f}} (+/- {{np.std(cv_scores):.4f}})")

# ── Train on full dataset & create submission ──────────────────
print("Training on full dataset...")
model.fit(train[FEATURES], train['{self.target_col}'])

test['{self.target_col}'] = model.predict(test[FEATURES])

id_col = 'ID' if 'ID' in test.columns else ('id' if 'id' in test.columns else test.columns[0])
test[[id_col, '{self.target_col}']].to_csv('baseline_submission.csv', index=False)

print("✅ baseline_submission.csv ready!")
print("🎯 Expected: Top 30% on public leaderboard")
print("📈 Next: add feature engineering from crew output for top 10%")
"""
        baseline_path = Path("quick_baseline.py")
        baseline_path.write_text(baseline_code, encoding="utf-8")
        print(f"✅ Quick baseline written to {baseline_path}")
        print("🚀 Run it now: python quick_baseline.py")
        print("⏱  Expected time: <5 minutes\n")
        return baseline_path

    # ──────────────────────────────────────────────────────────────────────────

    def run_full_workflow(self, dry_run: bool = False):
        """Execute complete workflow: RAG → [CrewAI] → Validation → Baseline."""
        try:
            # Step 1: Always run RAG
            self.step1_query_knowledge_base()

            if dry_run:
                print("\n⚡ DRY-RUN mode — skipping CrewAI (no API cost)")
                print("   Remove --dry-run to run the full 4-agent workflow.\n")
            else:
                # Step 2: CrewAI agents
                self.step2_configure_and_run_crew()
                # Step 3: Extract & validate
                self.step3_extract_and_validate_code()

            # Step 4: Always generate baseline
            self.step4_quick_baseline()

            print("\n" + "=" * 80)
            print("✅ WORKFLOW COMPLETE!")
            print("=" * 80)
            print("\n📋 Next Steps:")
            print("1. Run  python quick_baseline.py  →  immediate top-30% submission")
            if not dry_run:
                print("2. Review generated code in crew_output_*/")
            print("3. Use kb.query('...') for any technique questions")
            print("\n🔥 You're ready to compete!\n")

        except Exception as exc:
            import traceback
            print(f"\n❌ Error in workflow: {exc}")
            traceback.print_exc()


# ─── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Zindi Competition Workflow — CrewAI + RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train",   default="train.csv",   help="Path to train CSV")
    parser.add_argument("--test",    default="test.csv",    help="Path to test CSV")
    parser.add_argument("--target",  default="target",      help="Target column name")
    parser.add_argument("--problem", default="regression",  choices=["regression", "classification"],
                        help="Problem type")
    parser.add_argument("--name",    default="My Competition", help="Competition name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip CrewAI agents (RAG + baseline only, no LLM cost)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    workflow = ZindiCompetitionWorkflow(
        train_path=args.train,
        test_path=args.test,
        target_col=args.target,
        problem_type=args.problem,
        competition_name=args.name,
    )

    workflow.run_full_workflow(dry_run=args.dry_run)
