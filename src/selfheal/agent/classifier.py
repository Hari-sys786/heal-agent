from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Sequence

import ollama

logging.getLogger('selfheal').setLevel(logging.DEBUG)


class TicketIntent(str, Enum):
    """Supported ticket intents."""

    INSTALL = "install"
    SERVICE = "service"
    OTHER = "other"


@dataclass(slots=True)
class ClassifierResult:
    """Structured result from the LLM classifier."""

    intent: TicketIntent
    confidence: float
    packages: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "packages": list(self.packages),
            "services": list(self.services),
            "explanation": self.explanation,
        }


class TicketClassifier:
    """Runs an Ollama model to determine ticket intent and extract entities."""

    def __init__(
        self,
        *,
        model: str,
        installation_keywords: Iterable[str] | None = None,
        service_keywords: Iterable[str] | None = None,
    ) -> None:
        self.model = model
        self.installation_keywords = tuple(installation_keywords or ("install", "setup", "provision", "upgrade"))
        self.service_keywords = tuple(
            service_keywords
            or (
                "service",
                "restart",
                "down",
                "issue",
                "error",
                "failed",
                "failure",
                "problem",
                "not working",
                "outage",
                "connect",
                "connection",
            )
        )
        self._logger = logging.getLogger(__name__)

    def classify(self, short_description: str, description: str) -> ClassifierResult:
        """Send a JSON-constrained prompt to the LLM and parse the response."""
        prompt = self._build_prompt(short_description, description)
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a ticket intake classifier. "
                        "Respond with valid JSON only. "
                        "Never include commentary outside JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw_output = response["message"]["content"]
        self._logger.info("LLM classifier raw output: %s", raw_output)
        data = self._safe_json(raw_output)
        intent = self._normalise_intent(data.get("intent"))
        confidence = float(data.get("confidence", 0.5))
        packages = self._normalise_list(data.get("packages"))
        services = self._normalise_list(data.get("services"))
        explanation = str(data.get("explanation", "")).strip()

        # Heuristic boost if certain keywords appear
        combined = f"{short_description}\n{description}".lower()
        overridden_intent = self._override_intent(
            intent,
            combined_text=combined,
            packages=packages,
            services=services,
            installation_keywords=self.installation_keywords,
            service_keywords=self.service_keywords,
        )
        if overridden_intent is not intent:
            self._logger.debug(
                "Intent overridden from %s to %s (packages=%s services=%s)",
                intent.value,
                overridden_intent.value,
                packages,
                services,
            )
        intent = overridden_intent

        result = ClassifierResult(
            intent=intent,
            confidence=max(0.0, min(1.0, confidence)),
            packages=packages,
            services=services,
            explanation=explanation,
        )
        self._logger.info(
            "Classifier result: intent=%s confidence=%.2f packages=%s services=%s explanation=%s",
            result.intent.value,
            result.confidence,
            result.packages,
            result.services,
            result.explanation,
        )
        return result

    def _build_prompt(self, short_description: str, description: str) -> str:
        return (
            "Classify the following ServiceNow incident. "
            "You must respond with JSON only in the shape:\n"
            '{\n'
            '  "intent": "install|service|other",\n'
            '  "confidence": <number 0-1>,\n'
            '  "packages": [<package strings>],\n'
            '  "services": [<service strings>],\n'
            '  "explanation": "<short rationale>"\n'
            '}\n'
            "\nIntent definitions:\n"
            "- install: the user explicitly requests new software, packages, upgrades, or configuration changes that add capabilities.\n"
            "- service: the user reports an issue, error, outage, connection problem, restart request, or health check for something that should already be running.\n"
            "- other: everything else.\n"
            "\nGuidelines:\n"
            "- Prefer 'service' when the text mentions issues, errors, failures, outages, restarts, or troubleshooting needs—even if packages are mentioned as context.\n"
            "- Prefer 'install' only when the primary request is to install, upgrade, or provision software.\n"
            "- List packages only when intent is 'install'.\n"
            "- List services (e.g., nginx, postgres) only when intent is 'service'.\n"
            "- Keep explanation concise and reference the deciding phrases.\n"
            "\nShort description:\n"
            f"{short_description}\n\nDetails:\n{description}"
        )

    @staticmethod
    def _override_intent(
        intent: TicketIntent,
        *,
        combined_text: str,
        packages: Sequence[str],
        services: Sequence[str],
        installation_keywords: Sequence[str] = (),
        service_keywords: Sequence[str] = (),
    ) -> TicketIntent:
        lowered = combined_text.lower()
        has_packages = any(packages)
        has_services = any(services)
        install_hit = any(keyword in lowered for keyword in installation_keywords)
        service_hit = any(keyword in lowered for keyword in service_keywords)

        if intent is TicketIntent.SERVICE:
            if has_packages and not has_services and install_hit:
                return TicketIntent.INSTALL

        if intent is TicketIntent.INSTALL:
            if not install_hit and (has_services or service_hit):
                return TicketIntent.SERVICE

        if intent is TicketIntent.OTHER:
            if install_hit:
                return TicketIntent.INSTALL
            if service_hit:
                return TicketIntent.SERVICE

        return intent

    @staticmethod
    def _safe_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}

        cleaned = raw.strip()
        fenced_match = re.match(r"```(?:json)?\s*(.+?)\s*```$", cleaned, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            cleaned = fenced_match.group(1).strip()
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _normalise_intent(value: Any) -> TicketIntent:
        if isinstance(value, str):
            lowered = value.lower()
            if "install" in lowered:
                return TicketIntent.INSTALL
            if "service" in lowered or "diagnos" in lowered:
                return TicketIntent.SERVICE
        return TicketIntent.OTHER

    @staticmethod
    def _normalise_list(value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if isinstance(value, Iterable):
            normalised = []
            for item in value:
                if not item:
                    continue
                normalised.append(str(item).strip())
            return [item for item in normalised if item]
        return []


__all__ = ["TicketClassifier", "ClassifierResult", "TicketIntent"]
