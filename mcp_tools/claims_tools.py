"""
MCP tools for the Claims Agent.

1. claim_lookup_tool — fetches claim record from a (mock) claims DB
2. fraud_check_tool  — runs a fraud-risk score on a claim

In a real system: claim_lookup hits Snowflake/Oracle; fraud_check calls
a microservice running an XGBoost model + business rules.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_tools.base import MCPTool


# --- Mock DB (replace with real DB calls in production) -------------------
_MOCK_CLAIMS: dict[str, dict[str, Any]] = {
    "CLM-1001": {
        "claim_id": "CLM-1001",
        "member_id": "MEM-7788",
        "policy_id": "POL-555",
        "service_date": "2025-09-15",
        "provider": "Lakeshore Medical Center",
        "diagnosis_codes": ["J45.909"],   # Asthma, unspecified
        "procedure_codes": ["99213"],      # Office visit
        "billed_amount_usd": 320.00,
        "submitted_at": "2025-09-20T10:14:00Z",
        "status": "submitted",
    },
    "CLM-1002": {
        "claim_id": "CLM-1002",
        "member_id": "MEM-7788",
        "policy_id": "POL-555",
        "service_date": "2025-09-22",
        "provider": "Out-of-State Urgent Care",
        "diagnosis_codes": ["S93.401A"],   # Sprained ankle
        "procedure_codes": ["99284", "73600"],
        "billed_amount_usd": 4250.00,
        "submitted_at": "2025-09-25T18:02:00Z",
        "status": "submitted",
    },
    "CLM-1003": {
        "claim_id": "CLM-1003",
        "member_id": "MEM-9912",
        "policy_id": "POL-201",
        "service_date": "2025-10-01",
        "provider": "Smile Dental",
        "diagnosis_codes": ["K02.9"],      # Dental caries
        "procedure_codes": ["D2330"],      # Filling
        "billed_amount_usd": 285.00,
        "submitted_at": "2025-10-03T09:00:00Z",
        "status": "submitted",
    },
}


class ClaimLookupTool(MCPTool):
    name = "claim_lookup"
    description = (
        "Retrieve full claim record by claim_id. Returns provider, "
        "diagnosis/procedure codes, billed amount, dates, and current status. "
        "Use this FIRST before any other claim analysis."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "Claim identifier, format CLM-NNNN",
            },
        },
        "required": ["claim_id"],
    }

    def execute(self, *, claim_id: str) -> dict[str, Any]:
        record = _MOCK_CLAIMS.get(claim_id)
        if not record:
            return {"found": False, "claim_id": claim_id}
        return {"found": True, **record}


class FraudCheckTool(MCPTool):
    name = "fraud_check"
    description = (
        "Run a fraud-risk score on a claim. Returns a risk_score (0.0-1.0), "
        "risk_band (LOW/MEDIUM/HIGH), and a list of triggered fraud signals. "
        "Use this AFTER claim_lookup when the claim looks unusual or amount is high."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string"},
            "billed_amount_usd": {"type": "number"},
            "provider": {"type": "string"},
            "member_id": {"type": "string"},
        },
        "required": ["claim_id", "billed_amount_usd"],
    }

    def execute(
        self, *, claim_id: str, billed_amount_usd: float,
        provider: str = "", member_id: str = "",
    ) -> dict[str, Any]:
        signals: list[str] = []
        score = 0.05  # base rate

        # Heuristic rules — production would be an ML model
        if billed_amount_usd > 3000:
            signals.append("high_amount")
            score += 0.35
        if "out-of-state" in provider.lower() or "out of state" in provider.lower():
            signals.append("out_of_network_provider")
            score += 0.25
        if billed_amount_usd > 10000:
            signals.append("very_high_amount")
            score += 0.20

        # Tiny bit of jitter so demo isn't deterministic
        score = min(0.99, score + random.uniform(-0.03, 0.03))

        if score >= 0.6:
            band = "HIGH"
        elif score >= 0.3:
            band = "MEDIUM"
        else:
            band = "LOW"

        return {
            "claim_id": claim_id,
            "risk_score": round(score, 3),
            "risk_band": band,
            "signals": signals,
            "model_version": "fraud-v1.4-mock",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
