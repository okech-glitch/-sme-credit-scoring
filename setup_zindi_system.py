"""
SETUP SCRIPT - Windows/Mac/Linux compatible
Replaces setup_zindi_system.sh for cross-platform use

Run:
    python setup_zindi_system.py
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

HERE = Path(__file__).parent

PACKAGES = [
    # Core ML
    "pandas>=2.1",
    "numpy>=1.26",
    "scikit-learn>=1.3",
    "lightgbm>=4.1",
    "catboost>=1.2",
    "optuna>=3.5",
    "shap>=0.44",
    # CrewAI
    "crewai>=0.28",
    "crewai-tools>=0.1",
    # LlamaIndex RAG
    "llama-index-core",
    "llama-index-llms-anthropic",
    "llama-index-llms-ollama",
    "llama-index-llms-openai",
    "llama-index-llms-openai-like",
    "llama-index-embeddings-huggingface",
    "llama-index-readers-file",  # PDF support
    # LLM clients
    "anthropic>=0.18",
    "langchain-anthropic>=0.1",
    "langchain-openai>=0.0.8",
    # Embeddings
    "sentence-transformers>=2.3",
    # Utilities
    "python-dotenv>=1.0",
    "tqdm>=4.66",
]


def run(cmd: list, check: bool = True):
    print(f"   $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def main():
    print("=" * 60)
    print("🚀 Zindi Grandmaster System — Setup")
    print("=" * 60)

    # ── Python version check ────────────────────────────────────────────────
    major, minor = sys.version_info[:2]
    print(f"\n✅ Python {major}.{minor} detected")
    if major < 3 or (major == 3 and minor < 9):
        print("❌ Python 3.9+ required. Please upgrade.")
        sys.exit(1)

    # ── Upgrade pip ─────────────────────────────────────────────────────────
    print("\n[1/4] Upgrading pip...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    print("   ✅ pip up-to-date")

    # ── Install packages ────────────────────────────────────────────────────
    print(f"\n[2/4] Installing {len(PACKAGES)} packages...")
    run([sys.executable, "-m", "pip", "install"] + PACKAGES + ["-q"])
    print("   ✅ All packages installed")

    # ── Download embedding model ────────────────────────────────────────────
    print("\n[3/4] Pre-downloading embedding model (BAAI/bge-small-en-v1.5, ~90 MB)...")
    print("      This is a one-time download; future runs use the local cache.")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("   ✅ Embedding model cached!")
    except Exception as e:
        print(f"   ⚠️  Could not pre-download model: {e}")
        print("      It will download automatically on first use.")

    # ── Copy .env template ──────────────────────────────────────────────────
    print("\n[4/4] Setting up .env configuration...")
    env_template = HERE / ".env.template"
    env_file     = HERE / ".env"

    if env_file.exists():
        print("   ⏭  .env already exists — skipping (not overwriting)")
    elif env_template.exists():
        shutil.copy(env_template, env_file)
        print("   ✅ .env created from .env.template")
        print()
        print("   👉 ACTION REQUIRED: Edit .env and add your ANTHROPIC_API_KEY")
        print(f"      File location: {env_file}")
    else:
        print("   ⚠️  .env.template not found — create .env manually")

    # ── Done ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print()
    print("🎯 Next steps:")
    print("   1. Edit .env → add ANTHROPIC_API_KEY (or USE_LOCAL_LLM=true)")
    print("   2. python quick_start.py         ← verify everything works")
    print("   3. python zindi_rag_knowledge_base.py  ← build knowledge base")
    print("   4. python zindi_competition_workflow.py --help")
    print()
    print("🦙 Offline mode (no API key)?")
    print("   1. Install Ollama: https://ollama.ai")
    print("   2. ollama pull llama3")
    print("   3. Set USE_LOCAL_LLM=true in .env")
    print()
    print("🔥 You're ready to dominate Zindi!")


if __name__ == "__main__":
    main()
