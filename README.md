# Credit Risk Assessment Engine
### Alternative Data Scoring for SME Lending in Sub-Saharan Africa

An ML-based credit scoring engine that replaces traditional bureau scores with alternative behavioral signals: mobile money transaction frequency, utility payment patterns, and business operating history. 

This repository includes the core inference engine (88% precision), a secured FastAPI inference endpoint, and a Model Context Protocol (MCP) server that empowers stakeholders to query the portfolio data using natural language AI agents.

---

## Getting Started

### Prerequisites
- Python 3.10+
- `pip`

### 1. Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/okech-christopher/-sme-credit-scoring.git
cd -sme-credit-scoring
pip install -r requirements.txt
```

*(Note: Ensure you have `fastapi`, `uvicorn`, `lightgbm`, `pandas`, `scikit-learn`, `shap`, and `mcp` installed in your environment).*

### 2. Running the FastAPI Inference Engine
The core scoring engine is served via a FastAPI REST interface, secured by a CAPIE-style defensive perimeter.

Start the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Once running, you can:
- View the interactive API docs at `http://localhost:8000/docs`
- Send POST requests to `/score` with applicant JSON payloads to receive back a default probability and risk tier.

### 3. Running the MCP Server (Conversational Analytics)
The system includes an MCP (Model Context Protocol) server that exposes the credit portfolio to LLM agents. This allows developers and stakeholders to ask plain English questions about the risk distribution without writing SQL or Python.

Start the MCP server on a different port:
```bash
python mcp_credit.py --port 8200
```
This exposes 6 tools to AI agents: `score_applicant`, `explain_decision`, `risk_distribution`, `list_applicants`, `compare_applicants`, and `default_by_segment`.

## Live Demo & Stakeholder Report
- **Live Inference App:** [Hugging Face Space](https://huggingface.co/spaces/okechobonyo/sme-credit-scoring)
- **Stakeholder Report:** [View the interactive report](https://okech-christopher.github.io/credit_report.html)
