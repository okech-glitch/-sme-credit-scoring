"""
QUICK START — Zindi Grandmaster System
Smoke-test your installation without spending any API credits.

Run:
    python quick_start.py

Checks:
  ✅ Core imports (crewai, llama_index, lightgbm, catboost, dotenv)
  ✅ .env file exists and key format looks correct
  ✅ Embedding model loads (no API key needed)
  ✅ Core knowledge patterns index builds in-memory
  ✅ Sample query runs and returns an answer
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
ENV_FILE = HERE / ".env"

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

# ─── Step 1: Core imports ──────────────────────────────────────────────────────
print("\n🔍 Zindi Grandmaster — Installation Check")
print("=" * 60)

print("\n[1/5] Checking required packages...")
missing = []
packages = {
    "crewai":             "crewai",
    "llama_index.core":   "llama-index-core",
    "lightgbm":           "lightgbm",
    "catboost":           "catboost",
    "dotenv":             "python-dotenv",
    "sklearn":            "scikit-learn",
    "sentence_transformers": "sentence-transformers",
}
for module, pip_name in packages.items():
    try:
        __import__(module)
        print(f"   {PASS} {pip_name}")
    except ImportError:
        print(f"   {FAIL} {pip_name}  →  pip install {pip_name}")
        missing.append(pip_name)

if missing:
    print(f"\n{FAIL} Missing packages. Run:")
    print(f"   pip install {' '.join(missing)}")
    results.append(False)
else:
    results.append(True)

# ─── Step 2: .env check ────────────────────────────────────────────────────────
print("\n[2/5] Checking .env configuration...")
if not ENV_FILE.exists():
    print(f"   {WARN} .env not found — copy .env.template → .env and add your API key")
    print(f"         Using offline mode (Ollama) — make sure Ollama is running")
    results.append(None)  # warning, not failure
else:
    from dotenv import load_dotenv
    import os
    load_dotenv(ENV_FILE)
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key and key != "your_anthropic_key_here":
        print(f"   {PASS} ANTHROPIC_API_KEY found ({key[:8]}...)")
        results.append(True)
    elif os.getenv("OPENAI_API_KEY", "") not in ("", "your_openai_key_here"):
        print(f"   {PASS} OPENAI_API_KEY found")
        results.append(True)
    elif os.getenv("USE_LOCAL_LLM", "false").lower() == "true":
        print(f"   {PASS} USE_LOCAL_LLM=true (Ollama mode)")
        results.append(True)
    else:
        print(f"   {WARN} No API key configured in .env")
        print(f"         Edit .env and add your ANTHROPIC_API_KEY")
        results.append(None)

# ─── Step 3: Embedding model ───────────────────────────────────────────────────
print("\n[3/5] Loading embedding model (BAAI/bge-small-en-v1.5)...")
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    test_embed  = embed_model.get_text_embedding("target encoding leakage")
    print(f"   {PASS} Embedding model loaded (dim={len(test_embed)})")
    results.append(True)
except Exception as e:
    print(f"   {FAIL} Embedding model failed: {e}")
    print("         Fix: pip install llama-index-embeddings-huggingface sentence-transformers")
    results.append(False)
    embed_model = None

# ─── Step 4: In-memory RAG index ──────────────────────────────────────────────
print("\n[4/5] Building in-memory knowledge index (no API key needed)...")
try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.core.node_parser import SentenceSplitter

    if embed_model:
        Settings.embed_model = embed_model
        Settings.node_parser  = SentenceSplitter(chunk_size=512, chunk_overlap=64)

        sample_doc = Document(
            text="Target encoding fold-aware prevents leakage. LightGBM learning_rate 0.02. CatBoost depth 9.",
            metadata={"source": "quick_start_test"},
        )
        index = VectorStoreIndex.from_documents([sample_doc])
        print(f"   {PASS} In-memory index built successfully")
        results.append(True)
    else:
        print(f"   {WARN} Skipped (embed model not loaded)")
        results.append(None)
        index = None
except Exception as e:
    print(f"   {FAIL} Index build failed: {e}")
    results.append(False)
    index = None

# ─── Step 5: Sample retrieval (no LLM) ────────────────────────────────────────
print("\n[5/5] Running sample retrieval (no LLM API call)...")
try:
    if index:
        # Use retriever directly — no LLM generation, no API cost
        retriever = index.as_retriever(similarity_top_k=1)
        nodes = retriever.retrieve("What is the recommended LightGBM learning rate?")
        if nodes:
            preview = nodes[0].text[:120].replace("\n", " ")
            print(f"   {PASS} Retrieval works!")
            print(f"   📄 Top result: '{preview}...'")
            results.append(True)
        else:
            print(f"   {WARN} Retrieval returned no results")
            results.append(None)
    else:
        print(f"   {WARN} Skipped (index not built)")
        results.append(None)
except Exception as e:
    print(f"   {FAIL} Retrieval failed: {e}")
    results.append(False)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed  = sum(1 for r in results if r is True)
warned  = sum(1 for r in results if r is None)
failed  = sum(1 for r in results if r is False)

if failed == 0:
    if warned == 0:
        print(f"{PASS} All checks passed! System is ready.")
        print("\n🚀 Next steps:")
        print("   1. Run: python zindi_rag_knowledge_base.py")
        print("   2. Run: python zindi_competition_workflow.py --dry-run \\")
        print("             --train train.csv --test test.csv --target your_target")
    else:
        print(f"{WARN} {passed}/5 passed, {warned} warnings (configure .env to unlock full features)")
        print("\n📝 To enable LLM features:")
        print("   1. Edit .env with your ANTHROPIC_API_KEY")
        print("   2. OR set USE_LOCAL_LLM=true and install Ollama")
else:
    print(f"{FAIL} {failed} check(s) failed. Fix errors above before proceeding.")
    sys.exit(1)
