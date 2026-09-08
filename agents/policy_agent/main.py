"""
Policy Agent: answers coverage and benefit-eligibility questions.

Run: uvicorn agents.policy_agent.main:app --port 8002
"""
from __future__ import annotations

from common.schemas import AgentCard, AgentSkill, AgentCapabilities
from common.llm_client import LLMClient, MODEL_BALANCED
from common.logging_utils import configure_logging
from common.observability import init_observability
from config.settings import settings
from agents.base_agent import BaseA2AAgent
from mcp_tools.base import ToolRegistry
from mcp_tools.policy_tools import PolicyLookupTool, CoverageCheckTool


POLICY_SYSTEM_PROMPT = """You are the Policy & Coverage Agent for a health insurance company.

YOUR JOB:
1. Look up a member's policy with `policy_lookup`
2. For each procedure code in question, call `coverage_check`
3. Produce a clear coverage explanation:
   - Plan type and effective dates
   - Whether each procedure is covered
   - Whether pre-authorization is required
   - Member's expected cost-share (deductible + coinsurance)

RULES:
- ALWAYS call policy_lookup before answering coverage questions.
- ALWAYS call coverage_check for each specific procedure code mentioned.
- Do NOT speculate on coverage you have not verified via tool calls.
- Quote exact percentages and dollar amounts from tool results.
- If multiple procedure codes are involved, check each one separately.
- You do NOT process claims — if asked, recommend invoking the Claims Agent.
"""


def build_policy_agent() -> BaseA2AAgent:
    card = AgentCard(
        name="policy-agent",
        description="Answers coverage, benefits, and pre-authorization questions.",
        url=settings.policy_agent_url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[
            AgentSkill(
                id="check_coverage",
                name="Check coverage for a procedure",
                description="Determine if a procedure is covered and at what cost-share.",
                tags=["policy", "coverage", "benefits"],
                examples=[
                    "Is procedure 99213 covered under policy POL-555?",
                    "What's my deductible status on POL-555?",
                ],
            ),
        ],
    )
    registry = ToolRegistry()
    registry.register(PolicyLookupTool())
    registry.register(CoverageCheckTool())

    return BaseA2AAgent(
        agent_card=card,
        tool_registry=registry,
        llm=LLMClient(api_key=settings.groq_api_key, default_model=MODEL_BALANCED,),
        system_prompt=POLICY_SYSTEM_PROMPT
    )


configure_logging(settings.log_level)
init_observability()
agent = build_policy_agent()
app = agent.build_app()
