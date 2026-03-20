"""
Credit Risk Assessment Engine — MCP Server
Model Context Protocol server exposing credit scoring tools.

Connects any LLM-based agent to the Credit Risk Engine's data layer,
enabling conversational queries like:
  "What is the risk score for a 3-year-old business with mobile money?"
  "Show me the feature importance breakdown for this applicant."

Usage:
  python mcp_credit.py                # Start the MCP server
  python mcp_credit.py --port 8200    # Custom port

Architecture:
  Agent (LLM)  →  MCP Client  →  This Server  →  Model / Data Layer
"""

import json
import argparse
import math
from http.server import HTTPServer, BaseHTTPRequestHandler

# ---------------------------------------------------------------------------
# Simulated data layer — replace with live model calls in production
# ---------------------------------------------------------------------------

# Feature importance (from SHAP analysis of production LightGBM model)
FEATURE_IMPORTANCE = [
    {"feature": "mobile_money_txn_frequency",  "shap_value": 0.31, "description": "Monthly mobile money transaction count"},
    {"feature": "repayment_history_proxy",     "shap_value": 0.27, "description": "Inferred repayment regularity from utility and mobile data"},
    {"feature": "business_operating_age",      "shap_value": 0.18, "description": "Months since business registration or first transaction"},
    {"feature": "utility_payment_regularity",  "shap_value": 0.12, "description": "Consistency of monthly utility bill payments"},
    {"feature": "loan_to_income_ratio",        "shap_value": 0.08, "description": "Requested loan amount divided by estimated monthly income"},
    {"feature": "sector_risk_index",           "shap_value": 0.04, "description": "Industry-level default rate baseline"},
]

# Simulated applicant pool for portfolio-level queries
APPLICANT_POOL = [
    {"id": "A001", "name": "Wanjiku Groceries",    "sector": "Retail",         "age_months": 36, "mobile_txns": 85,  "has_bank": False, "risk_tier": "Low",    "default_prob": 0.08},
    {"id": "A002", "name": "Ochieng Transport",    "sector": "Logistics",      "age_months": 18, "mobile_txns": 42,  "has_bank": True,  "risk_tier": "Medium", "default_prob": 0.22},
    {"id": "A003", "name": "Nyambura Tailoring",   "sector": "Manufacturing",  "age_months": 48, "mobile_txns": 110, "has_bank": True,  "risk_tier": "Low",    "default_prob": 0.05},
    {"id": "A004", "name": "Kamau Hardware",        "sector": "Retail",         "age_months": 8,  "mobile_txns": 15,  "has_bank": False, "risk_tier": "High",   "default_prob": 0.41},
    {"id": "A005", "name": "Akinyi Beauty Salon",  "sector": "Services",       "age_months": 24, "mobile_txns": 60,  "has_bank": True,  "risk_tier": "Low",    "default_prob": 0.11},
    {"id": "A006", "name": "Mwangi Agri Supplies", "sector": "Agriculture",    "age_months": 12, "mobile_txns": 30,  "has_bank": False, "risk_tier": "Medium", "default_prob": 0.28},
    {"id": "A007", "name": "Otieno Welding",       "sector": "Manufacturing",  "age_months": 60, "mobile_txns": 95,  "has_bank": True,  "risk_tier": "Low",    "default_prob": 0.06},
    {"id": "A008", "name": "Wairimu Catering",     "sector": "Food & Bev",     "age_months": 4,  "mobile_txns": 8,   "has_bank": False, "risk_tier": "High",   "default_prob": 0.52},
    {"id": "A009", "name": "Njeri Pharmacy",        "sector": "Healthcare",     "age_months": 30, "mobile_txns": 78,  "has_bank": True,  "risk_tier": "Low",    "default_prob": 0.09},
    {"id": "A010", "name": "Kipchoge Boda Boda",   "sector": "Transport",      "age_months": 6,  "mobile_txns": 22,  "has_bank": False, "risk_tier": "High",   "default_prob": 0.45},
]


def _score_profile(profile):
    """Simple credit scoring heuristic mirroring the production LightGBM logic."""
    mobile_score = min(profile.get("mobile_txns", 0) / 100, 1.0) * 0.31
    age_score = min(profile.get("age_months", 0) / 48, 1.0) * 0.18
    bank_score = 0.12 if profile.get("has_bank", False) else 0.0
    utility_score = min(profile.get("utility_regularity", 0.5), 1.0) * 0.12
    income_ratio = profile.get("loan_to_income", 0.3)
    ratio_score = max(0, (1 - income_ratio)) * 0.08

    raw = mobile_score + age_score + bank_score + utility_score + ratio_score
    default_prob = round(max(0.02, 1 - (raw / 0.81)), 3)

    if default_prob < 0.15:
        tier = "Low"
        decision = "Approve"
    elif default_prob < 0.30:
        tier = "Medium"
        decision = "Manual Review"
    else:
        tier = "High"
        decision = "Decline"

    return {
        "default_probability": default_prob,
        "risk_tier": tier,
        "decision": decision,
        "confidence": round(1 - abs(default_prob - 0.15), 3),
    }


# ---------------------------------------------------------------------------
# MCP Tool definitions
# ---------------------------------------------------------------------------

TOOLS = {
    "score_applicant": {
        "description": "Score a new applicant's credit risk using alternative data signals. "
                       "Returns default probability, risk tier, and recommended decision.",
        "parameters": {
            "mobile_txns": {"type": "integer", "description": "Monthly mobile money transactions"},
            "age_months": {"type": "integer", "description": "Business operating age in months"},
            "has_bank": {"type": "boolean", "description": "Whether applicant has a bank account"},
            "utility_regularity": {"type": "number", "description": "0-1 score for utility payment consistency"},
            "loan_to_income": {"type": "number", "description": "Requested loan / monthly income ratio"},
        },
    },
    "explain_decision": {
        "description": "Return the SHAP feature importance breakdown explaining how the model "
                       "weighs each factor in its credit decisions.",
        "parameters": {},
    },
    "risk_distribution": {
        "description": "Show the portfolio-level risk tier distribution across all applicants.",
        "parameters": {},
    },
    "list_applicants": {
        "description": "List all applicants in the current portfolio with their risk tiers.",
        "parameters": {
            "tier": {"type": "string", "description": "Filter by risk tier: Low, Medium, High, or 'all'"},
        },
    },
    "compare_applicants": {
        "description": "Side-by-side comparison of two applicants.",
        "parameters": {
            "id_1": {"type": "string", "description": "First applicant ID (e.g. A001)"},
            "id_2": {"type": "string", "description": "Second applicant ID (e.g. A004)"},
        },
    },
    "default_by_segment": {
        "description": "Show average default probability broken down by a segment: "
                       "sector, bank_status, or age_bracket.",
        "parameters": {
            "segment": {"type": "string", "description": "One of: sector, bank_status, age_bracket"},
        },
    },
}


def execute_tool(name, params):
    if name == "score_applicant":
        return _score_profile(params)

    elif name == "explain_decision":
        return FEATURE_IMPORTANCE

    elif name == "risk_distribution":
        tiers = {"Low": 0, "Medium": 0, "High": 0}
        for a in APPLICANT_POOL:
            tiers[a["risk_tier"]] += 1
        total = len(APPLICANT_POOL)
        return {
            tier: {"count": c, "percentage": round(c / total * 100, 1)}
            for tier, c in tiers.items()
        }

    elif name == "list_applicants":
        tier_filter = params.get("tier", "all").lower()
        results = []
        for a in APPLICANT_POOL:
            if tier_filter == "all" or a["risk_tier"].lower() == tier_filter:
                results.append(a)
        return results

    elif name == "compare_applicants":
        id1 = params.get("id_1", "")
        id2 = params.get("id_2", "")
        a1 = next((a for a in APPLICANT_POOL if a["id"] == id1), None)
        a2 = next((a for a in APPLICANT_POOL if a["id"] == id2), None)
        if not a1 or not a2:
            return {"error": f"Applicant(s) not found: {id1}, {id2}"}
        return {"applicant_1": a1, "applicant_2": a2}

    elif name == "default_by_segment":
        seg = params.get("segment", "sector")
        groups = {}

        for a in APPLICANT_POOL:
            if seg == "sector":
                key = a["sector"]
            elif seg == "bank_status":
                key = "Has Bank Account" if a["has_bank"] else "No Bank Account"
            elif seg == "age_bracket":
                months = a["age_months"]
                if months < 12:
                    key = "< 1 year"
                elif months < 24:
                    key = "1-2 years"
                elif months < 48:
                    key = "2-4 years"
                else:
                    key = "4+ years"
            else:
                return {"error": f"Unknown segment: {seg}. Use: sector, bank_status, age_bracket"}

            if key not in groups:
                groups[key] = []
            groups[key].append(a["default_prob"])

        return {
            k: {"avg_default_prob": round(sum(v) / len(v), 3), "count": len(v)}
            for k, v in sorted(groups.items())
        }

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/tools":
            self._respond({"tools": TOOLS})
        elif self.path == "/execute":
            tool_name = body.get("tool")
            tool_params = body.get("parameters", {})
            result = execute_tool(tool_name, tool_params)
            self._respond({"result": result})
        else:
            self._respond({"error": "Unknown endpoint"}, 404)

    def do_GET(self):
        if self.path == "/health":
            self._respond({"status": "ok", "server": "credit-risk-mcp", "tools": list(TOOLS.keys())})
        else:
            self._respond({"error": "Use POST /tools or POST /execute"}, 405)

    def _respond(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, format, *args):
        print(f"[MCP-Credit] {args[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk MCP Server")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), MCPHandler)
    print(f"[MCP-Credit] Credit Risk MCP Server running on http://localhost:{args.port}")
    print(f"[MCP-Credit] Tools: {list(TOOLS.keys())}")
    server.serve_forever()
