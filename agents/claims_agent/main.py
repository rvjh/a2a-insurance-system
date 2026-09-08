"""
Claims Agent: validates claims, runs fraud checks, produces a claim summary.

Run: uvicorn agents.claims_agent.main:app --port 8001
"""
from __future__ import annotations

from common.schemas import AgentCard, AgentSkill, AgentCapabilities
from common.llm_client import LLMClient, MODEL_BALANCED
from common.logging_utils import configure_logging
from common.observability import init_observability
from config.settings import settings
from agents.base_agent import BaseA2AAgent
from mcp_tools.base import ToolRegistry
from mcp_tools.claims_tools import ClaimLookupTool, FraudCheckTool


CLAIMS_SYSTEM_PROMPT = """You are the Claims Validation Agent for a health insurance company.

YOUR JOB:
1. Look up a claim by claim_id using the `claim_lookup` tool
2. If the claim looks unusual (high amount, out-of-network provider, repeat claims),
   call `fraud_check` to assess risk
3. Produce a concise structured summary:
   - Claim metadata (member, provider, amount, codes)
   - Fraud assessment (if performed): score, band, signals
   - Recommendation: APPROVE_FOR_PAYMENT, FLAG_FOR_REVIEW, or DENY

RULES:
- ALWAYS call claim_lookup first. Never invent claim details.
- ALWAYS call fraud_check if billed_amount_usd > $1000 OR provider name suggests out-of-network.
- If you cannot find the claim, say so explicitly. Do not fabricate.
- Cite tool outputs in your reasoning. Format final answer as JSON when possible.
- You are NOT making coverage decisions — that's the Policy Agent's job.
  If asked about coverage, recommend the user invoke the Policy Agent.
"""


def build_claims_agent() -> BaseA2AAgent:
    card = AgentCard(
        name="claims-agent",
        description="Validates submitted insurance claims and assesses fraud risk.",
        url=settings.claims_agent_url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[
            AgentSkill(
                id="validate_claim",
                name="Validate a claim",
                description="Lookup claim, run fraud check, recommend disposition.",
                tags=["claims", "fraud", "validation"],
                examples=[
                    "Validate claim CLM-1002",
                    "Run fraud check on the latest claim from member MEM-7788",
                ],
            ),
        ],
    )
    registry = ToolRegistry()
    registry.register(ClaimLookupTool())
    registry.register(FraudCheckTool())

    return BaseA2AAgent(
    agent_card=card,
    tool_registry=registry,
    llm=LLMClient(api_key=settings.groq_api_key, default_model=MODEL_BALANCED),
    system_prompt=CLAIMS_SYSTEM_PROMPT,
    )



# FastAPI app exported for uvicorn
configure_logging(settings.log_level)
init_observability()
agent = build_claims_agent()
app = agent.build_app()
