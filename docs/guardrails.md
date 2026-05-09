# Guardrails — Beyond the Built-In Layer

The scaffold ships with a minimal `common/guardrails.py` (regex + heuristics). Below is how to plug in production-grade systems.

## 1. Microsoft Presidio (PII detection)

For multi-locale, ML-based PII detection (names, addresses, IBAN, passport numbers).

```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

Replace our `check_pii` with:

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

def check_pii_presidio(text: str, redact: bool = True):
    results = _analyzer.analyze(text=text, language="en")
    violations = [
        GuardrailViolation(
            f"pii.{r.entity_type.lower()}",
            GuardrailSeverity.BLOCK if r.entity_type in {"US_SSN", "CREDIT_CARD"} else GuardrailSeverity.WARN,
            f"{r.entity_type} detected (confidence {r.score:.2f})",
        )
        for r in results
    ]
    redacted = text
    if redact and results:
        redacted = _anonymizer.anonymize(text=text, analyzer_results=results).text
    return violations, redacted
```

## 2. Guardrails-AI (structured output validation)

For ensuring LLM output matches a schema, contains no toxic language, etc.

```bash
pip install guardrails-ai
guardrails hub install hub://guardrails/toxic_language
guardrails hub install hub://guardrails/competitor_check
```

Wrap the agent's final response:

```python
from guardrails import Guard
from guardrails.hub import ToxicLanguage

guard = Guard().use(ToxicLanguage(threshold=0.5, on_fail="exception"))

# In BaseA2AAgent.handle_message_send, after final_text is set:
try:
    validated = guard.validate(final_text)
    final_text = validated.validated_output
except Exception as e:
    # Block this output
    ...
```

## 3. Lakera Guard / Prompt Guard (prompt-injection)

Our regex-based injection detection misses sophisticated attacks (Unicode tricks, multi-step injection, encoded payloads).

**Lakera Guard** (commercial) or **PromptGuard** (Meta, open source) provide ML-based detection.

```python
# Example with Lakera (requires API key)
import requests

def check_injection_lakera(text: str) -> bool:
    resp = requests.post(
        "https://api.lakera.ai/v1/prompt_injection",
        headers={"Authorization": f"Bearer {os.getenv('LAKERA_API_KEY')}"},
        json={"input": text},
    )
    return resp.json().get("results", [{}])[0].get("flagged", False)
```

## 4. NeMo Guardrails (rail-based control)

NVIDIA's NeMo Guardrails lets you write *colang* rules:

```colang
define user ask about competitors
  "what about company X"
  "is X better"

define bot decline competitor question
  "I can only discuss our policies and claims."

define flow
  user ask about competitors
  bot decline competitor question
```

Heavier, but powerful when you have many topical rules. Worth evaluating if guardrails become a major part of your spec.

## 5. Hallucination / grounding checks

Real challenge: the LLM produces a coverage decision, but did it cite tool results?

Approach:
1. Track which tool calls fired in a task (`state["tool_results"]`).
2. On final output, check that each factual claim has a corresponding tool result.
3. Use a small cheap LLM (Haiku) as a **critic**:
   ```
   Given this output and these tool results, are all facts in the output supported by the tool results?
   Output: {final_text}
   Tools: {tool_results}
   Reply: GROUNDED | UNGROUNDED | UNCERTAIN
   ```
4. If UNGROUNDED → block + retry with stricter prompt.

This is straightforward to add as a new node in the LangGraph after `call_llm` and before END.

## 6. Where to enforce — the layering rule

| Layer | What to check | Why |
|-------|---------------|-----|
| **Gateway (input)** | PII redaction, prompt-injection, auth, rate limit | Outermost; cheap & uniform |
| **Agent (input)** | Domain-specific scope (e.g., "is this a claim/coverage question?") | Per-agent expertise |
| **Tool (input)** | Schema validation, business rules (e.g., claim ID exists) | Closest to data |
| **Agent (output)** | Hallucination, schema, toxicity, format | Right before user sees result |
| **Gateway (output)** | Final PII pass, audit log | Last line of defense |

The minimum viable set: **gateway-input PII** + **agent-output hallucination** + **gateway-output audit log**.
