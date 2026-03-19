# 🚀 ZINDI GRANDMASTER - QUICK START CHEAT SHEET

## ⚡ 5-Minute Setup

```bash
# 1. Make setup script executable
chmod +x setup_zindi_system.sh

# 2. Run installation
./setup_zindi_system.sh

# 3. Activate environment
source zindi_env/bin/activate

# 4. Set API key (optional - can use Ollama offline)
export ANTHROPIC_API_KEY='sk-ant-...'
```

## 🎯 Competition Day 1 (Get Top 30% in 2 Hours)

### Method 1: Full Automated Workflow
```python
from zindi_competition_workflow import ZindiCompetitionWorkflow

workflow = ZindiCompetitionWorkflow(
    train_path="train.csv",
    test_path="test.csv",
    target_col="target",
    problem_type="regression"  # or "classification"
)

# This runs everything: RAG → CrewAI → Code → Baseline
workflow.run_full_workflow()
```

**Output:**
- ✅ `baseline_submission.csv` - Submit immediately for top 30%
- ✅ `competition_YYYYMMDD_HHMMSS/` - All generated code
- ✅ `quick_baseline.py` - Copy-paste ready baseline

---

### Method 2: Just the Baseline (Fastest)
```python
# Edit quick_baseline.py with your file paths, then:
python quick_baseline.py

# Submits in <5 minutes → Top 30%
```

---

## 🔍 Need a Specific Technique? Query RAG

```python
from zindi_rag_knowledge_base import ZindiKnowledgeBase

kb = ZindiKnowledgeBase()
kb.build_index()

# Ask anything
answer = kb.query("How do I create rolling features without leakage?")
print(answer)
```

**Common Queries:**
```python
# Feature engineering
kb.query("Show me target encoding code that prevents leakage")
kb.query("Best interaction features for customer-product data")
kb.query("How to handle skewed targets in regression")

# Model tuning  
kb.query("Proven LightGBM hyperparameters for regression")
kb.query("Best CatBoost settings for classification")
kb.query("How to ensemble CatBoost and LightGBM")

# Time series
kb.query("Rolling window sizes for hourly data")
kb.query("Expanding features with proper shifts")
kb.query("How to prevent temporal leakage")

# Vision
kb.query("FastAI settings for image classification")
kb.query("Multi-scale inference for object detection")
```

---

## 🤖 Just Run the Agents

```python
from zindi_crew_system import ZindiGrandmasterCrew

crew = ZindiGrandmasterCrew(
    dataset_path="train.csv",
    target_column="target",
    problem_type="regression"
)

result = crew.run()
# Saves to crew_output.txt
```

**What the agents do:**
1. **EDA Agent**: Analyzes data, finds patterns, recommends transform
2. **Feature Agent**: Generates target encoding, rolling, interactions
3. **Model Agent**: Trains CatBoost + LightGBM with proven params
4. **Ensemble Agent**: Optimizes weights, validates, creates submission

---

## 📊 Competition Timeline Strategy

### Day 1 (First 24h) - Target: Top 30%
```bash
python quick_baseline.py  # <5 min
# Submit baseline_submission.csv
```

### Day 2-3 - Target: Top 10%  
```python
# Run full workflow to get feature engineering code
workflow.run_full_workflow()

# Use generated code from competition_YYYYMMDD/step_2_generated_code.py
# Add target encoding features
# Re-train and submit
```

### Day 4-7 - Target: Top 5%
```python
# Query RAG for domain-specific features
kb.query("Best features for [your domain] problems")

# Optimize ensemble weights
# Test: 0.5/0.5, 0.6/0.4, 0.7/0.3
# Pick best by CV score
```

### Final Week - Target: Top 3%
```python
# Add deep learning if applicable
kb.query("When should I blend TabNet with GBDT?")

# Stack models
kb.query("Show me stacking code for Level 2 meta-model")

# Final ensemble optimization
```

---

## 🛠️ Customization Quick Ref

### Use Offline (No API Keys)
```python
# Install Ollama first: https://ollama.ai
# Then: ollama pull llama3

kb = ZindiKnowledgeBase(use_local_llm=True)
```

### Add Your Own Competition to RAG
```python
# 1. Put your .ipynb in /mnt/user-data/uploads/
# 2. Rebuild index
kb = ZindiKnowledgeBase()
kb.build_index(force_rebuild=True)

# 3. Query it
kb.query("Show me the approach from my latest competition")
```

### Change CrewAI LLM
Edit `zindi_crew_system.py` line ~15:
```python
# For Claude (best)
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# For OpenAI
llm = ChatOpenAI(model="gpt-4o")

# For offline
llm = Ollama(model="llama3")
```

---

## 🔥 Pro Tips from 20+ Competitions

### Always Do First:
1. ✅ Check target distribution → apply sqrt/log if skewed
2. ✅ Identify categorical features → use target encoding
3. ✅ Look for time columns → add rolling/expanding features
4. ✅ Stratify CV properly → prevents misleading scores

### Never Do:
1. ❌ Use RandomForest (never wins on Zindi)
2. ❌ Use default hyperparameters (always tune)
3. ❌ Submit single model (always ensemble)
4. ❌ Encode test with test statistics (leakage!)
5. ❌ Trust public LB too much (focus on CV)

### Winning Formula:
```
Quick Baseline (Day 1) 
→ Add Target Encoding (Day 2)
→ Add Domain Features (Day 3-4)
→ Optimize Ensemble (Day 5-7)
→ Final Tuning (Last week)
= Top 5%
```

---

## 🆘 Troubleshooting

**Error: Module not found**
```bash
source zindi_env/bin/activate
pip install -r requirements.txt
```

**Error: ANTHROPIC_API_KEY not set**
```bash
export ANTHROPIC_API_KEY='your-key'
# Or use offline: use_local_llm=True
```

**Error: Out of memory during RAG**
```python
Settings.chunk_size = 512  # Reduce from 1024
```

**CrewAI timeout**
```python
llm = Ollama(model="llama3", request_timeout=300.0)
```

---

## 📞 Quick Reference

| Need | File | Command |
|------|------|---------|
| Full workflow | `zindi_competition_workflow.py` | `python zindi_competition_workflow.py` |
| Quick baseline | `quick_baseline.py` | `python quick_baseline.py` |
| Query RAG | `zindi_rag_knowledge_base.py` | `python zindi_rag_knowledge_base.py` |
| Run agents | `zindi_crew_system.py` | `python zindi_crew_system.py` |

---

**Remember: Speed matters. Get baseline up in <2 hours, iterate from there. 🚀**
