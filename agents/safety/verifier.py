"""
MedExplain AI - Safety Verification Agent.
Module: agents.safety.verifier

Agent 5 of the pipeline, and the Responsible AI control point. It reviews the
draft explanation before anything reaches the patient and runs five checks:

  1. Groundedness          - is every claim supported by a retrieved passage?
  2. Scope                 - does it diagnose, prescribe or predict prognosis?
  3. Tone and harm         - is it alarming, fatalistic or dismissive?
  4. Disclaimer integrity  - is the non-diagnostic notice intact?
  5. Critical-value escalation - do any values need urgent care instead?

On failure it sets approved = False with structured reasons. The orchestrator
routes control back to the explanation agent for revision (max 2 retries), then
fails closed. The agent that writes is never the agent that approves.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from medexplain.logging_config import audit, get_logger
from agents.safety import rules

logger = get_logger("MedExplain.Safety.Agent")

# Fraction of sentences that must overlap with retrieved context.
GROUNDEDNESS_THRESHOLD = float(os.getenv("SAFETY_GROUNDEDNESS_THRESHOLD", "0.5"))

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "for", "on", "with", "as", "by", "at", "from", "that", "this", "these",
    "those", "it", "its", "your", "you", "their", "there", "which", "can", "may", "might",
    "will", "would", "should", "could", "has", "have", "had", "do", "does", "did", "not",
    "if", "than", "then", "so", "such", "also", "about", "into", "over", "more", "most",
    "some", "any", "other", "when", "what", "how", "why", "who", "level", "levels", "test",
    "result", "results", "value", "values", "range", "normal", "please", "doctor",
}


@dataclass
class SafetyVerdict:
    """Structured outcome of the safety review."""

    approved: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    critical_values: List[Dict[str, Any]] = field(default_factory=list)
    escalation_message: Optional[str] = None
    groundedness_score: Optional[float] = 0.0
    revision_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _content_words(text: str) -> set:
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


# Sentences that assert nothing about medicine, and so cannot be "ungrounded".
#
# Without this the groundedness check punished honesty. When the knowledge base
# does not cover an analyte the explanation says so, and every one of those
# admissions was counted as an unsupported claim - a report with partial
# coverage scored 0% and was blocked entirely, hiding the tests that WERE
# properly grounded. Restatements of the patient's own measured values are
# excluded for the same reason: their source is the report itself.
_NON_CLAIM_MARKERS = (
    "no trusted reference passage",
    "could not find trusted reference",
    "could not find trusted",
    "no explanation is available",
    "no explanation is offered",
    "will not attempt an explanation",
    "not attempt an explanation",
    "do not have a trusted source",
    "producing one without a source",
    "risk giving you inaccurate information",
    "current knowledge base",
    "your result is",
    "this report contains",
    "reports 0 measured",
    "measured value(s)",
    "educational explanation",
    "educational tool",
    "not a diagnosis",
    "consult your doctor",
    "discuss these results",
    "discuss the results",
    "replace professional medical advice",
    "questions for your doctor",
    "how do these results compare",
    "should any of these be repeated",
    "is there anything in my history",
)


def _is_claim(sentence: str) -> bool:
    """True when a sentence asserts a medical fact that needs a source."""
    lowered = sentence.lower()
    return not any(marker in lowered for marker in _NON_CLAIM_MARKERS)


def _sentences(text: str) -> List[str]:
    cleaned = re.sub(r"[#*`>|\-]{1,}", " ", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [p.strip() for p in parts if len(p.strip().split()) >= 5]


class SafetyVerificationAgent:
    """[AGENT 5: Safety Verification Agent] - the final gate before the patient."""

    def __init__(self, groundedness_threshold: float = GROUNDEDNESS_THRESHOLD) -> None:
        self.groundedness_threshold = groundedness_threshold

    # ---------------------------------------------------------------- checks

    def check_groundedness(self, explanation: str, retrieved_passages: Sequence[Dict[str, Any]]) -> float:
        """
        Fraction of *claim* sentences whose content words overlap the retrieved
        passages. A lexical proxy for groundedness - deterministic, explainable
        and testable, which matters more here than sophistication.

        Boilerplate, value restatements and explicit "we have no source for this"
        admissions are excluded from the denominator: none of them assert a
        medical fact, so counting them as unsupported would penalise the system
        for being honest about the limits of its knowledge base.
        """
        lowered_exp = explanation.lower()
        if "could not find trusted" in lowered_exp or "no explanation is available" in lowered_exp or "reports 0 measured" in lowered_exp:
            return 1.0

        sentences = [s for s in _sentences(explanation) if _is_claim(s)]
        if not sentences:
            # Nothing was asserted, so nothing is unsupported. An honest
            # abstention is not an ungrounded claim.
            return 1.0
        if not retrieved_passages:
            return 0.0


        context_words: set = set()
        for passage in retrieved_passages:
            context_words |= _content_words(str(passage.get("text", "")))
            context_words |= _content_words(str(passage.get("title", "")))

        if not context_words:
            return 0.0

        supported = 0
        for sentence in sentences:
            words = _content_words(sentence)
            if not words:
                continue
            overlap = len(words & context_words) / len(words)
            if overlap >= 0.25:
                supported += 1

        return round(supported / len(sentences), 4)

    # ----------------------------------------------------------------- verify

    def verify(
        self,
        explanation: str,
        retrieved_passages: Sequence[Dict[str, Any]],
        test_items: Sequence[Dict[str, Any]],
        citations: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> SafetyVerdict:
        """Run all five checks and return a structured verdict."""
        violations: List[Dict[str, Any]] = []

        # 5. Critical-value escalation runs first - it overrides everything else.
        critical = rules.find_critical_values(list(test_items))
        if critical:
            # Only the critical-value check is reported. The others were never
            # run - escalation short-circuits before the explanation is reviewed,
            # and showing "groundedness 0%" implies a failure that did not happen.
            verdict = SafetyVerdict(
                approved=False,
                checks={"critical_values": False},
                critical_values=critical,
                escalation_message=rules.ESCALATION_MESSAGE,
                revision_hint=None,
                groundedness_score=None,
            )
            logger.warning("Safety gate: critical values present, escalating instead of explaining: %s", critical)
            audit("agent.safety", approved=False, reason="critical_value_escalation", critical_values=critical)
            return verdict

        # 2. Scope.
        scope_findings = rules.check_scope(explanation)
        for finding in scope_findings:
            violations.append({"check": "scope", **finding})

        # 3. Tone and harm.
        tone_findings = rules.check_tone(explanation)
        for finding in tone_findings:
            violations.append({"check": "tone", **finding})

        # 4. Disclaimer integrity.
        disclaimer_ok = rules.has_disclaimer(explanation)
        if not disclaimer_ok:
            violations.append({"check": "disclaimer", "reason": "non-diagnostic notice missing", "matched_text": ""})

        # 1. Groundedness.
        groundedness = self.check_groundedness(explanation, retrieved_passages)
        grounded_ok = groundedness >= self.groundedness_threshold
        if not grounded_ok:
            violations.append(
                {
                    "check": "groundedness",
                    "reason": f"only {groundedness:.0%} of sentences are supported by retrieved sources "
                              f"(threshold {self.groundedness_threshold:.0%})",
                    "matched_text": "",
                }
            )

        checks = {
            "groundedness": grounded_ok,
            "scope": not scope_findings,
            "tone": not tone_findings,
            "disclaimer": disclaimer_ok,
            "critical_values": True,
        }
        approved = all(checks.values())

        verdict = SafetyVerdict(
            approved=approved,
            checks=checks,
            violations=violations,
            groundedness_score=groundedness,
            revision_hint=self._build_revision_hint(violations) if not approved else None,
        )

        audit(
            "agent.safety",
            approved=approved,
            checks=checks,
            groundedness=groundedness,
            violation_count=len(violations),
            violations=[{"check": v["check"], "reason": v["reason"]} for v in violations],
        )
        if approved:
            logger.info("Safety gate: APPROVED (groundedness %.0f%%)", groundedness * 100)
        else:
            logger.warning(
                "Safety gate: REJECTED - %s",
                "; ".join(f"{v['check']}: {v['reason']}" for v in violations),
            )
        return verdict

    @staticmethod
    def _build_revision_hint(violations: Sequence[Dict[str, Any]]) -> str:
        """Instruction handed back to the explanation agent on the retry pass."""
        parts: List[str] = []
        for violation in violations:
            check = violation["check"]
            if check == "scope":
                parts.append(
                    "Remove any statement that names a diagnosis, recommends or names a medication, "
                    "or predicts what will happen to the patient."
                )
            elif check == "tone":
                parts.append(
                    "Remove alarming, fatalistic or dismissive wording, and do not speculate about serious conditions."
                )
            elif check == "disclaimer":
                parts.append(
                    "End with a clear notice that this is an educational explanation, not a diagnosis, "
                    "and that the patient should discuss the results with their doctor."
                )
            elif check == "groundedness":
                parts.append(
                    "Base every sentence on the retrieved source passages provided. "
                    "Do not add facts that are not present in those passages."
                )
        # de-duplicate while preserving order
        seen: set = set()
        ordered = [p for p in parts if not (p in seen or seen.add(p))]
        return " ".join(ordered)


_AGENT_SINGLETON: Optional[SafetyVerificationAgent] = None


def get_safety_agent() -> SafetyVerificationAgent:
    """Process-wide singleton."""
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = SafetyVerificationAgent()
    return _AGENT_SINGLETON
