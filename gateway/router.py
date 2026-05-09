"""
Agent Router — picks the right downstream agent for a request.

THREE STRATEGIES (use in order):
  1. Explicit `agent` field in params      — direct addressing
  2. Skill-tag match on user message       — keyword routing
  3. LLM-based router (Claude Haiku)       — semantic routing fallback

Why three? Direct addressing is fastest (free), keyword is cheap & deterministic,
LLM routing handles ambiguous cases. In production we log which path was used
so we can tune the cheaper layers over time.
"""
from __future__ import annotations

from typing import Optional
import re

import httpx

from common.schemas import AgentCard
from common.llm_client import LLMClient, MODEL_FAST
from common.logging_utils import get_logger
from common.observability import trace_span

logger = get_logger(__name__)


class AgentRegistry:
    """Holds known agents and their cards. Refreshes via /.well-known/agent-card.json."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentCard] = {}  # name -> card

    async def register_from_url(self, url: str, http: httpx.AsyncClient) -> AgentCard:
        resp = await http.get(f"{url.rstrip('/')}/.well-known/agent-card.json")
        resp.raise_for_status()
        card = AgentCard(**resp.json())
        self._agents[card.name] = card
        logger.info(f"Registered agent: {card.name} at {card.url}")
        return card

    def all(self) -> list[AgentCard]:
        return list(self._agents.values())

    def get(self, name: str) -> Optional[AgentCard]:
        return self._agents.get(name)


class AgentRouter:
    def __init__(self, registry: AgentRegistry, llm: Optional[LLMClient] = None) -> None:
        self.registry = registry
        self.llm = llm

    async def route(
        self,
        *,
        explicit_agent: Optional[str],
        user_text: str,
    ) -> AgentCard:
        with trace_span("router.route", input={"text_preview": user_text[:120]}) as span:
            # 1. Explicit
            if explicit_agent:
                card = self.registry.get(explicit_agent)
                if card:
                    span["output"] = {"strategy": "explicit", "agent": card.name}
                    return card
                logger.warning(f"Explicit agent '{explicit_agent}' not found, falling back")

            # 2. Keyword on tags + skill names
            text_lc = user_text.lower()
            scored: list[tuple[int, AgentCard]] = []
            for card in self.registry.all():
                score = 0
                for skill in card.skills:
                    for tag in skill.tags:
                        if tag.lower() in text_lc:
                            score += 2
                    if any(w in text_lc for w in skill.name.lower().split()):
                        score += 1
                # Heuristic boosts for our domain
                if any(w in text_lc for w in ("claim", "fraud", "submitted")):
                    if card.name == "claims-agent":
                        score += 3
                if any(w in text_lc for w in ("coverage", "policy", "deductible", "covered", "preauth")):
                    if card.name == "policy-agent":
                        score += 3
                if score > 0:
                    scored.append((score, card))

            if scored:
                scored.sort(key=lambda x: -x[0])
                top = scored[0][1]
                span["output"] = {"strategy": "keyword", "agent": top.name, "score": scored[0][0]}
                return top

            # 3. LLM router
            if self.llm:
                chosen = await self._llm_route(user_text)
                if chosen:
                    span["output"] = {"strategy": "llm", "agent": chosen.name}
                    return chosen

            # 4. Default to first registered agent
            agents = self.registry.all()
            if not agents:
                raise RuntimeError("No agents registered")
            span["output"] = {"strategy": "default", "agent": agents[0].name}
            return agents[0]

    async def _llm_route(self, user_text: str) -> Optional[AgentCard]:
        agents = self.registry.all()
        if not agents:
            return None
        catalog = "\n".join(
            f"- {c.name}: {c.description}" for c in agents
        )
        prompt = (
            f"Given these agents:\n{catalog}\n\n"
            f"Which ONE is best for this request: '{user_text}'?\n"
            f"Reply with ONLY the agent name, nothing else."
        )
        try:
            resp = self.llm.complete(
                system="You are a routing classifier. Output only the agent name.",
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_FAST,
                max_tokens=20,
                temperature=0,
            )
            picked = resp["text"].strip()
            for c in agents:
                if c.name == picked:
                    return c
        except Exception as e:
            logger.warning(f"LLM routing failed: {e}")
        return None
