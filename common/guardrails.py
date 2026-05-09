"""
Guardrails: input + output safety checks.

WHY THIS MATTERS IN INSURANCE:
  - PII leakage (SSN, member IDs in logs/responses) = compliance violation
  - Prompt injection from claim notes = agent manipulation risk
  - Hallucinated coverage decisions = legal/financial liability

DESIGN: a small composable checker. We don't pull in the full
`guardrails-ai` framework here because (a) it's heavy, (b) you'd want
to evaluate it against this minimal version. I'll show how to plug
guardrails-ai in below as an optional layer.

Two enforcement points:
  1. INPUT guardrails — run on every incoming user message at the gateway
  2. OUTPUT guardrails — run on every agent response before returning to user
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from common.logging_utils import get_logger

logger = get_logger(__name__)


class GuardrailSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GuardrailViolation:
    rule: str
    severity: GuardrailSeverity
    message: str
    detail: Optional[str] = None


@dataclass
class GuardrailResult:
    passed: bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    sanitized_text: Optional[str] = None  # set if we redacted PII

    def has_blocking(self) -> bool:
        return any(v.severity == GuardrailSeverity.BLOCK for v in self.violations)


# --- Detectors -------------------------------------------------------------

# US SSN pattern. In production, use Microsoft Presidio for multi-locale PII.
SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Heuristic prompt-injection markers. Real-world: use a classifier model
# (e.g., Lakera, Prompt Guard, or a small fine-tuned BERT).
INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?previous\s+instructions?\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+", re.I),
    re.compile(r"\bsystem\s*:\s*", re.I),
    re.compile(r"<\s*\|\s*(im_start|system|endoftext)\s*\|\s*>", re.I),
    re.compile(r"\bdisregard\s+(the\s+)?(above|prior)\b", re.I),
]


def check_pii(text: str, *, redact: bool = True) -> tuple[list[GuardrailViolation], str]:
    """Detect (and optionally redact) PII. Returns (violations, possibly_redacted_text)."""
    violations: list[GuardrailViolation] = []
    redacted = text

    if SSN_RE.search(text):
        violations.append(GuardrailViolation(
            "pii.ssn", GuardrailSeverity.BLOCK,
            "Possible SSN detected. Cannot process raw SSNs; use member IDs.",
        ))
        if redact:
            redacted = SSN_RE.sub("[SSN-REDACTED]", redacted)

    if CREDIT_CARD_RE.search(text):
        violations.append(GuardrailViolation(
            "pii.credit_card", GuardrailSeverity.BLOCK,
            "Credit card-like number detected.",
        ))
        if redact:
            redacted = CREDIT_CARD_RE.sub("[CC-REDACTED]", redacted)

    # Emails: warn (often legitimate in insurance) but redact in logs
    if EMAIL_RE.search(text):
        violations.append(GuardrailViolation(
            "pii.email", GuardrailSeverity.INFO, "Email present in payload.",
        ))

    return violations, redacted


def check_prompt_injection(text: str) -> list[GuardrailViolation]:
    violations: list[GuardrailViolation] = []
    for pat in INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            violations.append(GuardrailViolation(
                "prompt_injection", GuardrailSeverity.BLOCK,
                "Suspected prompt-injection pattern.",
                detail=f"Matched: {m.group(0)[:60]}",
            ))
            break  # one is enough to flag
    return violations


def check_topic_scope(text: str, allowed_topics: list[str]) -> list[GuardrailViolation]:
    """
    Very simple topic gate. Real-world: a small classifier (BART-MNLI zero-shot)
    or a Claude-Haiku classification call. Here we just keep it lightweight.
    """
    if not allowed_topics:
        return []
    text_lc = text.lower()
    if any(topic.lower() in text_lc for topic in allowed_topics):
        return []
    return [GuardrailViolation(
        "topic.out_of_scope", GuardrailSeverity.WARN,
        "Message may be off-topic for this agent.",
    )]


# --- Public entrypoints ----------------------------------------------------

def run_input_guardrails(
    text: str,
    *,
    allowed_topics: Optional[list[str]] = None,
    redact_pii: bool = True,
) -> GuardrailResult:
    """Apply on incoming user messages — gateway enforces this BEFORE routing."""
    all_violations: list[GuardrailViolation] = []

    pii_v, redacted = check_pii(text, redact=redact_pii)
    all_violations.extend(pii_v)
    all_violations.extend(check_prompt_injection(text))
    if allowed_topics:
        all_violations.extend(check_topic_scope(text, allowed_topics))

    blocking = any(v.severity == GuardrailSeverity.BLOCK for v in all_violations)
    return GuardrailResult(
        passed=not blocking,
        violations=all_violations,
        sanitized_text=redacted if redact_pii else None,
    )


def run_output_guardrails(text: str) -> GuardrailResult:
    """Apply on agent responses BEFORE returning to user."""
    violations: list[GuardrailViolation] = []
    pii_v, redacted = check_pii(text, redact=True)
    violations.extend(pii_v)

    # Hallucination heuristic: refuse responses with absolute coverage statements
    # not backed by tool calls. (In a real impl, track tool-call evidence and
    # require citations.)
    if re.search(r"\b(definitely|guaranteed|100%)\s+(covered|approved|denied)\b", text, re.I):
        violations.append(GuardrailViolation(
            "hallucination.absolute_claim", GuardrailSeverity.WARN,
            "Absolute coverage claim without explicit policy citation.",
        ))

    blocking = any(v.severity == GuardrailSeverity.BLOCK for v in violations)
    return GuardrailResult(
        passed=not blocking, violations=violations, sanitized_text=redacted,
    )
