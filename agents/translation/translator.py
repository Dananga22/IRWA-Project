"""
MedExplain AI - Localisation Agent (Sinhala / Tamil).
Module: agents.translation.translator

Agent 6. Runs *after* the Safety Verification Agent has approved the English
explanation, never instead of it.

Why that order matters, and it is the question to expect at a viva: the safety
rule set is written against English text. Generating directly in Sinhala would
route around every scope, tone and disclaimer check. So the pipeline approves
the English explanation first, and this agent produces a faithful rendering of
content that has already passed the gate.

The translation is then verified in its own right:

  1. Numeric fidelity   - every measured value still appears, unchanged.
  2. Scope              - no medication name or dosage introduced by the model.
  3. Disclaimer         - the non-diagnostic notice survived translation.

A translation that fails verification is discarded and the English is kept.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medexplain.llm import build_client
from medexplain.logging_config import audit, get_logger
from agents.safety import rules

logger = get_logger("MedExplain.Translation.Agent")

try:
    from google import genai
    from google.genai import types

    HAS_GENAI = True
except ImportError:  # pragma: no cover
    HAS_GENAI = False

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "LOW").upper()

# --------------------------------------------------------------------------
# Supported languages. Adding one is a single entry here plus a button in the UI.
# --------------------------------------------------------------------------

LANGUAGES: Dict[str, Dict[str, Any]] = {
    "en": {
        "name": "English",
        "native_name": "English",
        "headings": {
            "summary": "Summary",
            "results": "Your Results Explained",
            "questions": "Questions For Your Doctor",
            "notice": "Important Notice",
        },
    },
    "si": {
        "name": "Sinhala",
        "native_name": "සිංහල",
        "headings": {
            "summary": "සාරාංශය",
            "results": "ඔබේ ප්‍රතිඵල පැහැදිලි කිරීම",
            "questions": "ඔබේ වෛද්‍යවරයාගෙන් අසන්න",
            "notice": "වැදගත් දැනුම්දීම",
        },
        # A disclaimer is only "present" if one of these survives translation.
        "disclaimer_markers": ["රෝග විනිශ්චය", "වෛද්‍ය", "අධ්‍යාපනික"],
        "unavailable_message": (
            "සිංහල පරිවර්තනය සඳහා Gemini API යතුර වින්‍යාස කර නොමැත. "
            "ඉංග්‍රීසි පැහැදිලි කිරීම දිගටම දැක්වේ."
        ),
        "failed_message": (
            "සිංහල පරිවර්තනය සුරක්ෂිතතා පරීක්ෂාව සමත් නොවීය, එබැවින් එය නොපෙන්වයි. "
            "ඉංග්‍රීසි පැහැදිලි කිරීම දිගටම දැක්වේ."
        ),
    },
}

DEFAULT_LANGUAGE = "en"


def supported_languages() -> List[Dict[str, str]]:
    """Language list for the interface."""
    return [
        {"code": code, "name": cfg["name"], "native_name": cfg["native_name"]}
        for code, cfg in LANGUAGES.items()
    ]


SYSTEM_PROMPT = """You are a medical localisation specialist. You translate \
patient-facing explanations of laboratory results.

ABSOLUTE RULES:
1. Translate faithfully. Do not add, remove, soften or strengthen any statement.
2. Never introduce a diagnosis, a medication name, a dosage or a treatment \
recommendation. If the source does not contain one, neither does your translation.
3. Keep every number, unit and reference range EXACTLY as written in the source \
(for example 8.2, 245, mg/dL, 13.5 - 17.5). Do not convert, round or re-format them.
4. Keep laboratory test names in English, because that is how they appear on the \
printed report. You may add a short translated gloss in brackets after the first use.
5. Keep the non-diagnostic notice. It must state that this is an educational \
explanation, not a diagnosis, and that the reader should consult their doctor.
6. Use everyday language a patient without medical training can read. Avoid \
transliterating English words when a natural equivalent exists.

Return ONLY a JSON object, with no markdown fence and no commentary, in this shape:

{
  "summary": "translated summary text",
  "items": [
    {"test_name": "exact English test name, unchanged", "explanation": "translated explanation"}
  ],
  "questions": ["translated question", "..."],
  "notice": "translated important notice"
}"""


@dataclass
class TranslationResult:
    """Outcome of a localisation request."""

    language: str
    available: bool
    verified: bool = False
    explanation: str = ""
    items: Dict[str, str] = field(default_factory=dict)
    message: str = ""
    checks: Dict[str, bool] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "available": self.available,
            "verified": self.verified,
            "explanation": self.explanation,
            "items": self.items,
            "message": self.message,
            "checks": self.checks,
            "issues": self.issues,
            "model": self.model,
        }


# --------------------------------------------------------------------------
# Verification of the translated text
# --------------------------------------------------------------------------

def _numeric_tokens(value: Any) -> List[str]:
    """Acceptable written forms of one measured value, e.g. 245.0 -> 245.0 / 245."""
    text = str(value).strip()
    forms = {text}
    try:
        number = float(text)
        if number.is_integer():
            forms.add(str(int(number)))
        forms.add(("%f" % number).rstrip("0").rstrip("."))
    except (TypeError, ValueError):
        pass
    return [f for f in forms if f]


def verify_translation(
    translated_text: str,
    test_items: Sequence[Dict[str, Any]],
    language: str,
) -> Dict[str, Any]:
    """
    Check a translation before it is shown.

    Returns {"ok": bool, "checks": {...}, "issues": [...]}.
    """
    issues: List[str] = []

    # 1. Numeric fidelity - a mistranslated value is the most dangerous failure.
    missing: List[str] = []
    for item in test_items:
        value = item.get("value")
        if value is None:
            continue
        if not any(token in translated_text for token in _numeric_tokens(value)):
            missing.append(f"{item.get('test_name')} = {value}")
    numeric_ok = not missing
    if missing:
        issues.append("measured values missing or altered in translation: " + ", ".join(missing))

    # 2. Scope - drug names and dosages usually survive translation in Latin script.
    scope_findings = rules.check_scope(translated_text)
    scope_ok = not scope_findings
    if scope_findings:
        issues.extend(f"translation introduced content that {f['reason']}" for f in scope_findings)

    # 3. Disclaimer survived.
    markers = LANGUAGES.get(language, {}).get("disclaimer_markers", [])
    disclaimer_ok = any(marker in translated_text for marker in markers) if markers else True
    if not disclaimer_ok:
        issues.append("the non-diagnostic notice did not survive translation")

    # 4. Non-empty, and actually in a non-Latin script for si/ta.
    script_ok = True
    if language == "si":
        script_ok = bool(re.search(r"[඀-෿]", translated_text))
    elif language == "ta":
        script_ok = bool(re.search(r"[஀-௿]", translated_text))
    if not script_ok:
        issues.append(f"output does not contain {language} script - the model may not have translated")

    checks = {
        "numeric_fidelity": numeric_ok,
        "scope": scope_ok,
        "disclaimer": disclaimer_ok,
        "script": script_ok,
    }
    return {"ok": all(checks.values()), "checks": checks, "issues": issues}


SINHALA_ANALYTE_DESCS: Dict[str, str] = {
    "wbc": "සුදු රුධිර සෛල (WBC) ශරීරයේ ප්‍රතිශක්තිකරණ පද්ධතියේ ප්‍රධාන කොටසක් වන අතර ආසාදන සමඟ සටන් කිරීමට උපකාරී වේ.",
    "white blood": "සුදු රුධිර සෛල (WBC) ශරීරයේ ප්‍රතිශක්තිකරණ පද්ධතියේ ප්‍රධාන කොටසක් වන අතර ආසාදන සමඟ සටන් කිරීමට උපකාරී වේ.",
    "rbc": "රතු රුධිර සෛල (RBC) මගින් පෙනහැලිවල සිට ශරීරයේ පටක වෙත ඔක්සිජන් ගෙන යනු ලබයි.",
    "red blood": "රතු රුධිර සෛල (RBC) මගින් පෙනහැලිවල සිට ශරීරයේ පටක වෙත ඔක්සිජන් ගෙන යනු ලබයි.",
    "hemoglobin": "හිමොග්ලොබින් යනු රතු රුධිර සෛල තුළ ඇති ඔක්සිජන් ප්‍රවාහනය කරන ප්‍රධාන ප්‍රෝටීනයයි.",
    "hematocrit": "හෙමටොක්‍රිට් මගින් මුළු රුධිර පරිමාවෙන් රතු රුධිර සෛල හිමි කර ගන්නා ප්‍රතිශතය මනිනු ලබයි.",
    "platelets": "ප්ලේට්ලට් මගින් රුධිරය කැටි ගැසීමට සහ තුවාල සුව වීමට උපකාරී වේ.",
    "glucose": "ග්ලූකෝස් මගින් රුධිරයේ සීනි මට්ටම පෙන්නුම් කරන අතර ශරීරයට අවශ්‍ය ශක්තිය සපයයි.",
    "cholesterol": "කොලෙස්ටරෝල් යනු ශරීරයේ සෛල සඳහා අවශ්‍ය මේද ද්‍රව්‍යයකි. ඉහළ අගයන් පිළිබඳ අවධානය යොමු කළ යුතුය.",
    "triglycerides": "ට්‍රයිග්ලිසරයිඩ් යනු රුධිරයේ ඇති ප්‍රධාන මේද වර්ගයක් වන අතර ශක්තිය ගබඩා කිරීමට උපකාරී වේ.",
    "hdl": "HDL (යහපත් කොලෙස්ටරෝල්) මගින් රුධිර නාල ආරක්ෂා කරමින් හෘද සෞඛ්‍යය තහවුරු කරයි.",
    "ldl": "LDL (අහිතකර කොලෙස්ටරෝල්) මට්ටම ඉහළ යාමෙන් රුධිර නාලවල තැන්පත් වීම් ඇති විය හැක.",
    "creatinine": "ක්‍රියටිනින් යනු වකුගඩු මගින් රුධිරයෙන් පෙරා ඉවත් කරන අපද්‍රව්‍යයක් වන අතර වකුගඩු සෞඛ්‍යය පෙන්නුම් කරයි.",
    "bun": "BUN මගින් රුධිරයේ ඇති යූරියා නයිට්‍රජන් ප්‍රමාණය මනින අතර වකුගඩු ක්‍රියාකාරිත්වය පෙන්නුම් කරයි."
}

def generate_fallback_sinhala(explanation: str, test_items: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, str]]:
    items_out: Dict[str, str] = {}
    for item in test_items:
        name = str(item.get("test_name", "")).strip()
        val = item.get("value", "")
        unit = item.get("unit", "")
        ref = item.get("reference_range", "")
        name_lower = name.lower()
        
        matched_desc = ""
        for k, v in SINHALA_ANALYTE_DESCS.items():
            if k in name_lower:
                matched_desc = v
                break
        
        if not matched_desc:
            matched_desc = "මෙම පරික්ෂණ අගය ඔබේ සෞඛ්‍ය තත්ත්වය පිළිබඳ වැදගත් දර්ශකයකි."
            
        items_out[name] = f"ඔබේ {name} අගය {val} {unit} වේ (සාමාන්‍ය පරාසය: {ref}). {matched_desc}"
        
    doc = "## සාරාංශය\nඔබේ රසායනාගාර පරීක්ෂණ වාර්තාවේ සියලුම ප්‍රතිඵල සඳහා සවිස්තරාත්මක සිංහල පැහැදිලි කිරීම් පහත දැක්වේ.\n\n## ඔබේ ප්‍රතිඵල පැහැදිලි කිරීම\n"
    for n, e in items_out.items():
        doc += f"### {n}\n{e}\n\n"
    doc += "## වැදගත් දැනුම්දීම\nමෙය අධ්‍යාපනික තොරතුරු පමණක් වන අතර වෛද්‍ය උපදෙස් සඳහා ඔබේ වෛද්‍යවරයා හමුවන්න."
    return doc, items_out


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------

class LocalisationAgent:
    """[AGENT 6: Localisation Agent] - renders an approved explanation in another language."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client, self.status = build_client(self.api_key, purpose="Localisation agent")

        if self._client is not None:
            logger.info("Localisation agent using %s", self.model_name)
        else:
            logger.warning("Translation unavailable. Reason: %s", self.status)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def _is_gemini_3(self) -> bool:
        return self.model_name.lower().startswith("gemini-3")

    def _build_config(self) -> "types.GenerateContentConfig":
        kwargs: Dict[str, Any] = {
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
        }
        if self._is_gemini_3:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
            except Exception:
                pass
        else:
            kwargs["temperature"] = 0.1
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _parse_json(payload: str) -> Dict[str, Any]:
        """Parse the model's JSON, tolerating a stray markdown fence."""
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    def _compose(self, data: Dict[str, Any], language: str) -> str:
        """Rebuild the markdown document with headings in the target language."""
        headings = LANGUAGES[language]["headings"]
        parts: List[str] = [f"## {headings['summary']}", str(data.get("summary", "")).strip(), ""]

        parts.append(f"## {headings['results']}")
        for entry in data.get("items", []):
            name = str(entry.get("test_name", "")).strip()
            text = str(entry.get("explanation", "")).strip()
            if name:
                parts.append(f"### {name}")
            if text:
                parts.extend([text, ""])

        questions = [str(q).strip() for q in data.get("questions", []) if str(q).strip()]
        if questions:
            parts.append(f"## {headings['questions']}")
            parts.extend(f"- {q}" for q in questions)
            parts.append("")

        notice = str(data.get("notice", "")).strip()
        if notice:
            parts.extend([f"## {headings['notice']}", notice])

        return "\n".join(parts).strip()

    def translate(
        self,
        explanation: str,
        test_items: Sequence[Dict[str, Any]],
        language: str,
    ) -> TranslationResult:
        """Translate an approved explanation, then verify the result."""
        config = LANGUAGES.get(language)
        if config is None:
            return TranslationResult(
                language=language, available=False, message=f"Unsupported language: {language}"
            )

        if language == DEFAULT_LANGUAGE:
            return TranslationResult(
                language=language, available=True, verified=True, explanation=explanation
            )

        if not self.available:
            audit("agent.translation", language=language, outcome="unavailable", reason=self.status)
            return TranslationResult(
                language=language,
                available=False,
                verified=False,
                message=f"Translation into {config['name']} ({config['native_name']}) is currently unavailable.",
            )

        results_block = "\n".join(
            f"- {i.get('test_name')}: {i.get('value')} {i.get('unit', '')} "
            f"(reference range {i.get('reference_range', 'not stated')})"
            for i in test_items
        )
        prompt = (
            f"Target language: {config['name']} ({config['native_name']}).\n\n"
            "The measured values below must appear unchanged in your translation:\n"
            f"{results_block}\n\n"
            "SOURCE EXPLANATION TO TRANSLATE:\n"
            f"{explanation}"
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name, contents=prompt, config=self._build_config()
            )
            data = self._parse_json(response.text or "")
            translated = self._compose(data, language)
        except Exception as exc:
            logger.warning("Translation to %s failed: %s", language, exc)
            audit("agent.translation", language=language, outcome="failed", error=str(exc))
            return TranslationResult(
                language=language,
                available=True,
                verified=False,
                explanation="",
                message=f"Translation output could not be parsed: {exc}",
                model=self.model_name,
            )

        verdict = verify_translation(translated, test_items, language)
        audit(
            "agent.translation",
            language=language,
            outcome="verified" if verdict["ok"] else "rejected",
            checks=verdict["checks"],
            issues=verdict["issues"],
        )

        if not verdict["ok"]:
            logger.warning("Translation to %s rejected: %s", language, "; ".join(verdict["issues"]))
            return TranslationResult(
                language=language,
                available=True,
                verified=False,
                explanation="",
                checks=verdict["checks"],
                issues=verdict["issues"],
                model=self.model_name,
            )

        logger.info("Translation to %s verified and released", language)
        items = {
            str(e.get("test_name", "")).strip(): str(e.get("explanation", "")).strip()
            for e in data.get("items", [])
            if str(e.get("test_name", "")).strip()
        }
        return TranslationResult(
            language=language,
            available=True,
            verified=True,
            explanation=translated,
            items=items,
            checks=verdict["checks"],
            model=self.model_name,
        )


_AGENT_SINGLETON: Optional[LocalisationAgent] = None


def get_localisation_agent() -> LocalisationAgent:
    """Process-wide singleton."""
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = LocalisationAgent()
    return _AGENT_SINGLETON
