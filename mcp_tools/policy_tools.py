"""
MCP tools for the Policy Agent.

1. policy_lookup_tool   — fetch policy contract details
2. coverage_check_tool  — given a procedure/diagnosis, what's covered?

In a real system: these hit a policy-master-data service backed by
a structured contract repository (often a graph or rules engine).
"""
from __future__ import annotations

from typing import Any

from mcp_tools.base import MCPTool


_MOCK_POLICIES: dict[str, dict[str, Any]] = {
    "POL-555": {
        "policy_id": "POL-555",
        "member_id": "MEM-7788",
        "plan_type": "PPO Gold",
        "effective_date": "2025-01-01",
        "termination_date": None,
        "deductible_remaining_usd": 250.00,
        "out_of_pocket_max_usd": 5000.00,
        "out_of_pocket_used_usd": 1820.00,
        "in_network_coinsurance_pct": 20,
        "out_of_network_coinsurance_pct": 50,
        "covers_dental": False,
    },
    "POL-201": {
        "policy_id": "POL-201",
        "member_id": "MEM-9912",
        "plan_type": "Dental Basic",
        "effective_date": "2025-01-01",
        "termination_date": None,
        "deductible_remaining_usd": 50.00,
        "out_of_pocket_max_usd": 1500.00,
        "out_of_pocket_used_usd": 200.00,
        "in_network_coinsurance_pct": 20,
        "out_of_network_coinsurance_pct": 50,
        "covers_dental": True,
    },
}

# Coverage rules: which CPT/ICD codes the plan covers
_COVERAGE_RULES = {
    "PPO Gold": {
        "covered_cpt_prefixes": ["992", "736", "001", "100"],   # E/M, X-ray, etc.
        "excluded_cpt_prefixes": ["D"],                          # No dental
        "preauth_required_cpts": ["27447", "27130"],             # Joint replacements
    },
    "Dental Basic": {
        "covered_cpt_prefixes": ["D"],
        "excluded_cpt_prefixes": ["992", "736"],
        "preauth_required_cpts": ["D7240"],                      # Surgical extraction
    },
}


class PolicyLookupTool(MCPTool):
    name = "policy_lookup"
    description = (
        "Retrieve a member's policy details: plan type, deductibles, "
        "coinsurance percentages, and out-of-pocket status. "
        "Use this FIRST for any coverage question."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "policy_id": {"type": "string", "description": "Policy ID, format POL-NNN"},
        },
        "required": ["policy_id"],
    }

    def execute(self, *, policy_id: str) -> dict[str, Any]:
        record = _MOCK_POLICIES.get(policy_id)
        if not record:
            return {"found": False, "policy_id": policy_id}
        return {"found": True, **record}


class CoverageCheckTool(MCPTool):
    name = "coverage_check"
    description = (
        "Check whether a specific procedure code (CPT or D-code) is covered "
        "under the member's plan. Returns covered (bool), preauth_required (bool), "
        "and the in-network/out-of-network coinsurance the member would owe."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "policy_id": {"type": "string"},
            "procedure_code": {"type": "string", "description": "CPT or HCPCS code"},
            "in_network": {"type": "boolean", "default": True},
        },
        "required": ["policy_id", "procedure_code"],
    }

    def execute(
        self, *, policy_id: str, procedure_code: str, in_network: bool = True,
    ) -> dict[str, Any]:
        policy = _MOCK_POLICIES.get(policy_id)
        if not policy:
            return {"covered": False, "reason": "policy_not_found"}

        rules = _COVERAGE_RULES.get(policy["plan_type"], {})
        covered = any(
            procedure_code.startswith(p)
            for p in rules.get("covered_cpt_prefixes", [])
        )
        excluded = any(
            procedure_code.startswith(p)
            for p in rules.get("excluded_cpt_prefixes", [])
        )
        if excluded:
            return {
                "covered": False,
                "reason": f"Procedure code {procedure_code} is excluded under plan {policy['plan_type']}",
                "preauth_required": False,
            }
        if not covered:
            return {
                "covered": False,
                "reason": f"Procedure code {procedure_code} not in covered list for plan {policy['plan_type']}",
                "preauth_required": False,
            }

        preauth = procedure_code in rules.get("preauth_required_cpts", [])
        coinsurance = (
            policy["in_network_coinsurance_pct"]
            if in_network
            else policy["out_of_network_coinsurance_pct"]
        )
        return {
            "covered": True,
            "preauth_required": preauth,
            "member_coinsurance_pct": coinsurance,
            "deductible_remaining_usd": policy["deductible_remaining_usd"],
            "plan_type": policy["plan_type"],
        }
