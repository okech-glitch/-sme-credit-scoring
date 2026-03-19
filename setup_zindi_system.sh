#!/bin/bash
# Zindi Grandmaster Setup - Install CrewAI + RAG System
# Works with minimal internet (downloads cached to ~/.cache)

echo "🚀 Installing Zindi Grandmaster System..."
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv zindi_env
source zindi_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core packages..."
pip install --quiet \
    pandas==2.1.4 \
    numpy==1.26.2 \
    scikit-learn==1.3.2 \
    lightgbm==4.1.0 \
    catboost==1.2.2 \
    optuna==3.5.0 \
    shap==0.44.0

# Install CrewAI
echo "🤖 Installing CrewAI..."
pip install --quiet crewai==0.28.8 crewai-tools==0.1.6

# Install RAG dependencies
echo "🔍 Installing RAG system..."
pip install --quiet \
    llama-index==0.10.12 \
    llama-index-llms-anthropic \
    llama-index-llms-ollama \
    llama-index-embeddings-huggingface \
    sentence-transformers==2.3.1

# Install API clients (optional)
echo "🔌 Installing LLM clients..."
pip install --quiet \
    anthropic==0.18.1 \
    langchain-anthropic==0.1.9 \
    langchain-openai==0.0.8

# Download embedding model (works offline after first download)
echo "⬇️  Downloading embedding model (one-time, ~90MB)..."
python3 << EOF
from sentence_transformers import SentenceTransformer
print("Downloading BAAI/bge-small-en-v1.5...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print("✅ Embedding model cached!")
EOF

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Quick Start:"
echo "1. Activate environment: source zindi_env/bin/activate"
echo "2. Set API key (optional): export ANTHROPIC_API_KEY='your-key'"
echo "3. Run RAG: python zindi_rag_knowledge_base.py"
echo "4. Run CrewAI: python zindi_crew_system.py"
echo ""
echo "💡 For offline use with Ollama:"
echo "   - Install Ollama: https://ollama.ai"
echo "   - Pull model: ollama pull llama3"
echo "   - Set use_local_llm=True in scripts"
echo ""
echo "🔥 You're ready to dominate Zindi!"
