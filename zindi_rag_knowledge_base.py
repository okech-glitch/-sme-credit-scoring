"""
ZINDI RAG KNOWLEDGE BASE - Your Personal Winning Solutions Library
Indexes all your past competitions and winning techniques for instant retrieval
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# ─── Load API keys from .env ───────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# LlamaIndex for RAG (lighter than LangChain)
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def _build_llm():
    """Select LLM based on available API keys. Priority: OpenRouter → Anthropic → OpenAI → Ollama."""
    use_local        = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    openrouter_key   = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_url   = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku")
    anthropic_key    = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key       = os.getenv("OPENAI_API_KEY", "")

    if use_local:
        from llama_index.llms.ollama import Ollama
        print("🦙 RAG using local Ollama (llama3)")
        return Ollama(model="llama3", request_timeout=120.0)
    elif openrouter_key and openrouter_key != "your_openrouter_key_here":
        from llama_index.llms.openai_like import OpenAILike
        print(f"🌐 RAG using OpenRouter → {openrouter_model}")
        return OpenAILike(
            model=openrouter_model,
            api_key=openrouter_key,
            api_base=openrouter_url,
            is_chat_model=True,
            context_window=100000,
        )
    elif anthropic_key and anthropic_key != "your_anthropic_key_here":
        from llama_index.llms.anthropic import Anthropic
        print("🤖 RAG using Claude claude-sonnet-4-5")
        return Anthropic(model="claude-sonnet-4-5")
    elif openai_key and openai_key != "your_openai_key_here":
        from llama_index.llms.openai import OpenAI
        print("🤖 RAG using OpenAI gpt-4o-mini")
        return OpenAI(model="gpt-4o-mini")
    else:
        from llama_index.llms.ollama import Ollama
        print("⚠️  No LLM API key found. Falling back to Ollama (must be running).")
        print("   Add OPENROUTER_API_KEY to .env for cloud generation.")
        return Ollama(model="llama3", request_timeout=120.0)


class ZindiKnowledgeBase:
    """
    RAG system containing all your Zindi winning patterns:
    - Traffic Forecasting (1st place) — rolling/expanding features
    - DigiCow (16th place) — target encoding patterns
    - Bank Transaction (1st place) — NER + transaction features
    - Farm to Feed (2nd place) — customer-product interactions
    - Floods (2nd place) — deep learning tabular + GBDT blend
    - Vision competitions (FastAI + YOLO approaches)
    """

    # All notebooks present in the files folder
    _NOTEBOOK_REGISTRY = [
        # (filename,                          domain,       competition_name)
        ("fastai_winningsolution.ipynb",       "vision",     "fastai_winning_vision"),
        ("LOAN_DEFAULT_CATBOOST.ipynb",        "tabular",    "loan_default_catboost"),
        ("predicting_carbon_emission.ipynb",   "regression", "carbon_emission"),
        ("digicow-done-correctly.ipynb",       "tabular",    "digicow_target_encoding"),
        ("removenans_40.ipynb",                "tabular",    "removenans_fe_pipeline"),
        ("Solution1.ipynb",                    "tabular",    "solution1_ensemble"),
        ("fastai_bollworm_vit384.ipynb",       "vision",     "bollworm_vit_detection"),
        ("vegetation-mapping-notebook-best.ipynb", "vision", "vegetation_mapping"),
        ("tree-image-regression.ipynb",        "vision",     "tree_image_regression"),
        ("Noise_data_simple_exploration.ipynb","tabular",    "noise_data_eda"),
    ]

    def __init__(
        self,
        knowledge_dir: Optional[str] = None,
        persist_dir: str = "./zindi_rag_storage",
        use_local_llm: bool = False,  # kept for backwards compat; .env takes priority
    ):
        # Default knowledge dir: same folder as this script (where notebooks live)
        if knowledge_dir is None:
            knowledge_dir = str(Path(__file__).parent)
        self.knowledge_dir = Path(knowledge_dir)
        self.persist_dir = Path(persist_dir)

        # ── Embeddings (no API key needed — runs offline after first download) ──
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        )

        # ── LLM ──
        Settings.llm = _build_llm()

        # ── Chunking optimized for code blocks ──
        Settings.node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200
        )

        self.index = None
        self.query_engine = None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_core_patterns_doc(self) -> Document:
        """Hard-coded proven patterns — always available, even with no notebooks."""
        return Document(
            text="""
# KOLESHJR'S PROVEN ZINDI WINNING PATTERNS

## Core Philosophy
- NEVER use RandomForest — it never appears in winning solutions
- Default stack: CatBoost + LightGBM ensemble (proven in 15+ competitions)
- ALWAYS use stratified cross-validation (5-fold or 10-fold)
- Target encoding is SACRED — implement fold-aware to prevent leakage

## Feature Engineering Patterns

### 1. Target Encoding with Statistics (Bank Transaction 1st, DigiCow solutions)
```python
def target_encode_fold_aware(train, test, group_cols, target_col, stratify_col, n_folds=10):
    '''
    Fold-aware target encoding: mean, std, skewness, min, max, 25th/75th percentiles.
    Test set is encoded using FULL train statistics (no leakage risk for test).
    Train is encoded OOF to prevent leakage.
    '''
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    stats = ['mean', 'std', 'min', 'max', 'skew']
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Encode test using full train stats
    for stat in stats:
        col_name = f"{'_'.join(group_cols)}_{target_col}_{stat}"
        agg = train.groupby(group_cols)[target_col].agg(stat).reset_index()
        agg.columns = group_cols + [col_name]
        test = test.merge(agg, on=group_cols, how='left')

    # Encode train OOF (prevents leakage)
    train_enc = train.copy()
    for stat in stats:
        col_name = f"{'_'.join(group_cols)}_{target_col}_{stat}"
        train_enc[col_name] = np.nan
        for _, (tr_idx, val_idx) in enumerate(skf.split(train, train[stratify_col])):
            agg = train.iloc[tr_idx].groupby(group_cols)[target_col].agg(stat).reset_index()
            agg.columns = group_cols + [col_name]
            merged = train.iloc[val_idx].merge(agg, on=group_cols, how='left')
            train_enc.iloc[val_idx, train_enc.columns.get_loc(col_name)] = merged[col_name].values

    return train_enc, test
```

### 2. Rolling Window Features (Traffic Forecasting 1st place)
```python
def create_rolling_features(df, group_cols, value_cols, windows=[168, 336]):
    '''
    Proven windows: 168 hours (1 week), 336 hours (2 weeks).
    ALWAYS shift by 168h (or equivalent) to prevent leakage.
    '''
    shift_period = 168

    for col in value_cols:
        for window in windows:
            base = df.groupby(group_cols)[col].shift(shift_period)
            df[f'{col}_rolling_mean_{window}h'] = base.rolling(window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window}h']  = base.rolling(window, min_periods=1).std()
            df[f'{col}_rolling_q25_{window}h']  = base.rolling(window, min_periods=1).quantile(0.25)
            df[f'{col}_rolling_q75_{window}h']  = base.rolling(window, min_periods=1).quantile(0.75)
    return df
```

### 3. Expanding Features (Traffic Forecasting)
```python
def create_expanding_features(df, group_cols, value_cols, shifts=[168, 336, 504]):
    '''Captures long-term evolution with multiple shift periods.'''
    for col in value_cols:
        for shift in shifts:
            base = df.groupby(group_cols)[col].shift(shift)
            df[f'{col}_expanding_mean_shift{shift}'] = base.expanding(min_periods=1).mean()
            df[f'{col}_expanding_std_shift{shift}']  = base.expanding(min_periods=1).std()
    return df
```

### 4. Interaction Features (Farm to Feed 2nd place)
```python
def create_interactions(df, cat_cols, num_cols):
    '''Customer-product interactions and ratio features.'''
    for i, col1 in enumerate(num_cols):
        for col2 in num_cols[i+1:]:
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)
            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

    for cat1 in cat_cols:
        for cat2 in cat_cols:
            if cat1 != cat2:
                for num in num_cols:
                    df[f'{cat1}_{cat2}_{num}_mean'] = df.groupby([cat1, cat2])[num].transform('mean')
    return df
```

## GBDT Ensemble (proven in 15+ competitions)

```python
# Proven ensemble weights:
#  60% CatBoost + 40% LightGBM  →  Traffic Forecasting (1st place)
#  50% CatBoost + 50% LightGBM  →  Farm to Feed (2nd place)

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
import numpy as np

lgb_params = {
    'learning_rate': 0.02,
    'num_leaves': 254,
    'max_depth': 10,
    'feature_fraction': 0.67,
    'bagging_fraction': 0.72,
    'bagging_freq': 8,
    'min_child_samples': 100,
    'lambda_l1': 2.4e-6,
    'lambda_l2': 1.7e-8,
    'n_estimators': 1000,
    'verbose': -1,
    'random_state': 42,
}

cat_params = {
    'learning_rate': 0.02,
    'depth': 9,
    'l2_leaf_reg': 1.3e-6,
    'random_strength': 6e-7,
    'bagging_temperature': 0.38,
    'iterations': 15000,
    'verbose': False,
    'random_state': 42,
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# Final: 0.6 * cat_preds + 0.4 * lgb_preds
```

## Cross-Validation Strategy
```python
# Standard (10+ competitions)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Time series with gap
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=168)
```

## Quick Baseline (Top 30% in <2h)
```python
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

features = [c for c in train.columns if c not in ['ID', 'id', 'target']]

# Simple label encoding
from sklearn.preprocessing import LabelEncoder
for col in train.select_dtypes(include='object').columns:
    if col in features:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

model = LGBMRegressor(learning_rate=0.05, num_leaves=150, max_depth=9,
                      n_estimators=1000, random_state=42, verbose=-1)
cv = cross_val_score(model, train[features], train['target'],
                     cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"CV MAE: {-cv.mean():.4f} (+/- {cv.std():.4f})")

model.fit(train[features], train['target'])
test['target'] = model.predict(test[features])
test[['ID', 'target']].to_csv('quick_baseline.csv', index=False)
print("✅ quick_baseline.csv ready — expected: Top 30%")
```

## Common Pitfalls
1. Using RandomForest (never appears in winning solutions)
2. Default hyperparameters (always tune from proven starting point)
3. Single model submissions (ensemble adds 0.01-0.02)
4. Features without validation (always check OOF score)
5. Leakage in target encoding (always fold-aware)
6. Ignoring target distribution (always check skewness → sqrt if >1)
""",
            metadata={
                "source": "core_winning_patterns",
                "competitions": "15+",
                "success_rate": "top_10%_average",
            }
        )

    def _load_notebooks(self) -> List[Document]:
        """Load all available notebooks from the knowledge directory."""
        documents = []
        for filename, domain, comp_name in self._NOTEBOOK_REGISTRY:
            nb_path = self.knowledge_dir / filename
            if not nb_path.exists():
                print(f"   ⏭  Skipping (not found): {filename}")
                continue
            try:
                with open(nb_path, "r", encoding="utf-8", errors="replace") as f:
                    nb = json.load(f)

                code_cells = [
                    cell["source"] if isinstance(cell["source"], str)
                    else "".join(cell["source"])
                    for cell in nb.get("cells", [])
                    if cell.get("cell_type") == "code"
                ]
                markdown_cells = [
                    cell["source"] if isinstance(cell["source"], str)
                    else "".join(cell["source"])
                    for cell in nb.get("cells", [])
                    if cell.get("cell_type") == "markdown"
                ]

                combined = (
                    f"# Competition Notebook: {comp_name}\n\n"
                    f"## Notes\n{''.join(markdown_cells[:5])}\n\n"
                    f"## Code\n{''.join(code_cells)}"
                )

                doc = Document(
                    text=combined,
                    metadata={
                        "source": filename,
                        "domain": domain,
                        "competition": comp_name,
                    }
                )
                documents.append(doc)
                print(f"   ✅ Loaded: {filename}")
            except Exception as e:
                print(f"   ⚠️  Could not load {filename}: {e}")
        return documents

    def _load_pdf(self) -> List[Document]:
        """Load the Traffic Forecasting PDF report."""
        pdf_path = self.knowledge_dir / "Kolesh_Final_Report.pdf"
        documents = []
        if not pdf_path.exists():
            print("   ⏭  PDF report not found — skipping.")
            return documents
        try:
            pdf_docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
            for doc in pdf_docs:
                doc.metadata["source"] = "traffic_forecasting_1st_place_report"
                doc.metadata["competition"] = "Traffic Forecasting"
                doc.metadata["rank"] = "1st"
            documents.extend(pdf_docs)
            print(f"   ✅ Loaded: Kolesh_Final_Report.pdf ({len(pdf_docs)} pages)")
        except Exception as e:
            print(f"   ⚠️  Could not load PDF: {e}")
        return documents

    def _collect_documents(self) -> List[Document]:
        """Collect all documents for indexing."""
        docs: List[Document] = []

        print("\n📚 Loading knowledge base documents...")
        print(f"   Knowledge dir: {self.knowledge_dir}\n")

        # 1. Core patterns (always available)
        docs.append(self._build_core_patterns_doc())
        print("   ✅ Core winning patterns loaded")

        # 2. PDF report
        docs.extend(self._load_pdf())

        # 3. Notebooks
        docs.extend(self._load_notebooks())

        print(f"\n📊 Total documents loaded: {len(docs)}")
        return docs

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """Build or load the RAG index from disk."""
        if self.persist_dir.exists() and not force_rebuild:
            print("📖 Loading existing knowledge base from disk...")
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            self.index = load_index_from_storage(storage_context)
            print("✅ Knowledge base loaded!")
        else:
            print("🔨 Building knowledge base from your competition files...")
            documents = self._collect_documents()
            self.index = VectorStoreIndex.from_documents(documents, show_progress=True)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(self.persist_dir))
            print(f"✅ Knowledge base built and saved → {self.persist_dir}")

        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )
        return self.index

    def query(self, question: str, verbose: bool = True) -> str:
        """
        Query your Zindi knowledge base.

        Example questions:
          kb.query("How do I do target encoding without leakage?")
          kb.query("What LightGBM hyperparameters for Traffic Forecasting?")
          kb.query("Best ensemble weights for CatBoost + LightGBM?")
          kb.query("Rolling features for time series?")
        """
        if self.query_engine is None:
            raise ValueError("Index not built. Call build_index() first.")

        if verbose:
            print(f"\n🔍 Query: '{question}'")
            print("-" * 70)

        enhanced = (
            f"Based on Koleshjr's proven Zindi winning solutions, answer:\n{question}\n\n"
            "Provide:\n"
            "1. The exact technique used (with code if applicable)\n"
            "2. Which competition(s) this worked in\n"
            "3. Why it worked (domain reasoning)\n"
            "4. Any warnings or pitfalls to avoid"
        )

        response = self.query_engine.query(enhanced)

        if verbose:
            print(f"\n✅ Answer:\n{response}\n")
            print("-" * 70)

        return str(response)

    def get_competition_summary(self, competition_name: str) -> str:
        """Get the full approach summary for a specific competition."""
        return self.query(
            f"Summarize the approach for {competition_name} competition: "
            "feature engineering, models, CV strategy, and final ensemble."
        )

    def get_technique_examples(self, technique: str) -> str:
        """
        Get code examples for a specific technique.
        E.g. 'target encoding', 'rolling features', 'CatBoost hyperparameters'
        """
        return self.query(
            f"Show me code examples and best practices for {technique} "
            "from past winning solutions."
        )

    def add_competition(self, notebook_path: str, metadata: Dict) -> None:
        """
        Add a new competition notebook to the knowledge base.

        Args:
            notebook_path: Absolute path to the .ipynb file
            metadata: Dict with keys: competition, domain, rank (optional)

        Example:
            kb.add_competition(
                "c:/Users/user/Downloads/files/my_new_win.ipynb",
                {"competition": "New Competition", "domain": "tabular", "rank": "1st"}
            )
        """
        if self.index is None:
            raise ValueError("Build the index first with build_index()")

        nb_path = Path(notebook_path)
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb_path}")

        with open(nb_path, "r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)

        code_cells = [
            cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        combined = f"# Competition: {metadata.get('competition', nb_path.stem)}\n\n" + "\n".join(code_cells)

        doc = Document(text=combined, metadata=metadata)
        self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=str(self.persist_dir))
        print(f"✅ Added '{metadata.get('competition')}' to knowledge base")

    def list_competitions(self) -> None:
        """Print all competitions that would be indexed from the knowledge dir."""
        print(f"\n📋 Competition notebooks in: {self.knowledge_dir}\n")
        found, missing = [], []
        for filename, domain, comp in self._NOTEBOOK_REGISTRY:
            path = self.knowledge_dir / filename
            if path.exists():
                found.append((comp, domain, filename))
            else:
                missing.append((comp, domain, filename))

        print("✅ Found:")
        for comp, domain, fn in found:
            print(f"   [{domain:10s}] {comp} ({fn})")
        if missing:
            print("\n⏭  Missing (will be skipped):")
            for comp, domain, fn in missing:
                print(f"   [{domain:10s}] {comp} ({fn})")
        pdf_path = self.knowledge_dir / "Kolesh_Final_Report.pdf"
        print(f"\n📄 PDF report: {'✅' if pdf_path.exists() else '⏭  Missing'} Kolesh_Final_Report.pdf")


# ─── QUICK START EXAMPLES ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Zindi Knowledge Base — Your Personal Winning Library")
    print("=" * 70)

    kb = ZindiKnowledgeBase()  # auto-detects knowledge dir from script location

    # Show what's available
    kb.list_competitions()

    # Build or load index
    print("\n")
    kb.build_index(force_rebuild=False)

    # Example queries
    print("\n" + "=" * 70)
    print("EXAMPLE QUERIES")
    print("=" * 70)

    kb.query("How do I implement target encoding without data leakage? Show exact code.")
    kb.query("What are the proven LightGBM hyperparameters from Traffic Forecasting?")
    kb.query("What rolling and expanding features should I use for time series?")
    kb.query("How do I ensemble CatBoost + LightGBM? Best weights?")

    print("\n✅ Done! Use kb.query('your question') to access all winning patterns.")
