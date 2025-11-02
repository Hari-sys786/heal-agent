from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from dotenv import load_dotenv
load_dotenv()
import ollama

logging.getLogger('selfheal').setLevel(logging.DEBUG)


class TicketIntent(str, Enum):
    """Supported ticket intents."""

    INSTALL = "install"
    SERVICE = "service"
    OTHER = "other"

@dataclass(slots=True)
class PackageRequest:
    """Normalized representation of a requested software install."""

    name: str
    version: str | None = None
    manager: str | None = None
    install_command: str | None = None
    install_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "manager": self.manager,
            "install_command": self.install_command,
            "install_steps": list(self.install_steps),
        }


@dataclass(slots=True)
class ClassifierResult:
    """Structured result from the LLM classifier."""

    intent: TicketIntent
    confidence: float
    packages: List[PackageRequest] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "packages": [package.to_dict() for package in self.packages],
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
        supported_services: Iterable[str] | None = None,
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
        self.supported_services = tuple(supported_services or ())
        self._logger = logging.getLogger(__name__)
        self._latest_version_cache: Dict[str, str] = {}
        self._install_strategy_cache: Dict[str, PackageRequest] = {}

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
        packages = self._normalise_packages(data.get("packages"))
        packages = self._enrich_packages(packages)
        services = self._normalise_services(data.get("services"))
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
        services_line = ", ".join(self.supported_services) if self.supported_services else "(no restrictions)"
        return (
            "Classify the following ServiceNow incident. "
            "You must respond with JSON only in the shape:\n"
            '{\n'
            '  "intent": "install|service|other",\n'
            '  "confidence": <number 0-1>,\n'
            '  "packages": [\n'
            '    {\n'
            '      "name": "<software name>",\n'
            '      "version": "<specific version or \"latest\">" | null,\n'
            '      "manager": "apt-get|apt|yum|dnf|custom" | null,\n'
            '      "install_command": "<single shell command when manager is custom>" | null,\n'
            '      "install_steps": ["<ordered shell commands>", ...] | []\n'
            '    }\n'
            '  ],\n'
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
            "- For packages, capture the precise name. If the user says 'install version 2.1.0', set 'version' to '2.1.0'. If they say 'latest version', determine the latest stable release number and set 'version' to that actual string—never respond with the literal word 'latest'.\n"
            "- When the software needs a manual installer (for example, ngrok), set 'manager' to 'custom' and provide either a single 'install_command' or an ordered list of 'install_steps' that an automation can execute without prompts (download, extract, chmod, run installer, etc.).\n"
            "- When the package manager is usable, set 'manager' accordingly and include any necessary shell commands in 'install_steps' (for example 'sudo apt-get install -y <package>').\n"
            "- List services only when intent is 'service'. If a list of supported services is provided, choose only from that list: "
            f"{services_line}.\n"
            "- Keep explanation concise and reference the deciding phrases.\n"
            "\nShort description:\n"
            f"{short_description}\n\nDetails:\n{description}"
        )

    @staticmethod
    def _override_intent(
        intent: TicketIntent,
        *,
        combined_text: str,
        packages: Sequence[PackageRequest],
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

    def _normalise_services(self, value: Any) -> List[str]:
        services = self._normalise_list(value)
        if not self.supported_services:
            return services
        allowed = {service.lower() for service in self.supported_services}
        return [service for service in services if service.lower() in allowed]

    @staticmethod
    def _normalize_package_mapping(data: Mapping[str, Any]) -> PackageRequest | None:
        name = str(data.get("name", "")).strip()
        if not name:
            return None
        version_raw = data.get("version")
        version = str(version_raw).strip() or None if version_raw is not None else None
        manager_raw = data.get("manager")
        manager = str(manager_raw).strip() or None if manager_raw is not None else None
        command_raw = data.get("install_command")
        install_command = str(command_raw).strip() or None if command_raw is not None else None
        steps_raw = data.get("install_steps")
        steps: List[str] = []
        if isinstance(steps_raw, str):
            candidate = steps_raw.strip()
            if candidate:
                steps = [candidate]
        elif isinstance(steps_raw, Iterable) and not isinstance(steps_raw, (str, bytes)):
            for item in steps_raw:
                if isinstance(item, str):
                    candidate = item.strip()
                    if candidate:
                        steps.append(candidate)
        return PackageRequest(name=name, version=version, manager=manager, install_command=install_command, install_steps=steps)

    @classmethod
    def _normalise_packages(cls, value: Any) -> List[PackageRequest]:
        packages: List[PackageRequest] = []
        if value is None:
            return packages
        if isinstance(value, Mapping):
            maybe = cls._normalize_package_mapping(value)
            if maybe:
                packages.append(maybe)
            return packages
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, Mapping):
                    maybe = cls._normalize_package_mapping(item)
                    if maybe:
                        packages.append(maybe)
                elif isinstance(item, str):
                    name = item.strip()
                    if name:
                        packages.append(PackageRequest(name=name))
        return packages

    def _enrich_packages(self, packages: Sequence[PackageRequest]) -> List[PackageRequest]:
        enriched: List[PackageRequest] = []
        for package in packages:
            updated = package

            if package.version and package.version.strip().lower() == "latest":
                resolved_version = self._resolve_latest_version(package.name)
                if resolved_version:
                    updated = PackageRequest(
                        name=package.name,
                        version=resolved_version,
                        manager=package.manager,
                        install_command=package.install_command,
                        install_steps=list(package.install_steps),
                    )
                else:
                    updated = PackageRequest(
                        name=package.name,
                        version=None,
                        manager=package.manager,
                        install_command=package.install_command,
                        install_steps=list(package.install_steps),
                    )

            updated = self._get_install_strategy(updated)
            enriched.append(updated)
        return enriched

    def _get_install_strategy(self, package: PackageRequest) -> PackageRequest:
        key = package.name.lower()
        cached = self._install_strategy_cache.get(key)
        if cached:
            return self._merge_package_metadata(package, cached)

        strategy = self._query_install_strategy(package)
        merged = self._merge_package_metadata(package, strategy) if strategy else package

        if merged.manager and merged.manager.lower() == "apt-get" and not merged.install_steps:
            merged = self._merge_package_metadata(merged, PackageRequest(name=merged.name, manager="apt-get", install_steps=self._default_apt_steps(merged)))

        if (not merged.install_command and not merged.install_steps) or (
            merged.manager and merged.manager.lower() == "custom" and not merged.install_command and not merged.install_steps
        ):
            fallback = self._default_custom_command(package.name)
            if fallback:
                merged = self._merge_package_metadata(merged, fallback)

        self._install_strategy_cache[key] = merged
        return merged

    @staticmethod
    def _merge_package_metadata(base: PackageRequest, override: PackageRequest | None) -> PackageRequest:
        if override is None:
            return base
        manager = override.manager or base.manager
        version = base.version or override.version
        install_command = override.install_command or base.install_command

        steps = list(base.install_steps)
        for step in override.install_steps:
            if step not in steps:
                steps.append(step)

        return PackageRequest(
            name=base.name,
            version=version,
            manager=manager,
            install_command=install_command,
            install_steps=steps,
        )

    def _query_install_strategy(self, package: PackageRequest) -> PackageRequest | None:
        prompt = (
            "For a Debian/Ubuntu system, provide the unattended shell commands required to install the software '"
            f"{package.name}'. Respond with JSON only in the shape:\n"
            '{\n'
            '  "manager": "apt-get" | "custom",\n'
            '  "install_command": "<single shell command>" | null,\n'
            '  "install_steps": ["<shell command>", ...]\n'
            '}\n'
            "Rules:\n"
            "- Always include the ordered 'install_steps' array with the exact commands to run.\n"
            "- Use 'apt-get' only when the package is available in apt repositories; otherwise set 'manager' to 'custom'.\n"
            "- When 'manager' is 'apt-get', leave 'install_command' null and include the necessary apt commands in 'install_steps' (for example 'sudo apt-get install -y package').\n"
            "- When 'manager' is 'custom', provide either a single 'install_command' or detail each command in 'install_steps' (download, extract, chmod, run installer, etc.).\n"
            "- Do not include explanations or commentary outside the JSON response.\n"
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover - LLM failure path
            self._logger.warning("Install strategy lookup failed for %s: %s", package.name, exc)
            return None

        payload = response.get("message", {}).get("content", "")
        data = self._safe_json(payload)

        manager_raw = data.get("manager")
        manager = str(manager_raw).strip().lower() if isinstance(manager_raw, str) else None
        if manager not in {"apt-get", "apt", "custom"}:
            manager = None
        if manager == "apt":
            manager = "apt-get"

        command_raw = data.get("install_command")
        install_command = str(command_raw).strip() if isinstance(command_raw, str) else None

        steps: List[str] = []
        steps_raw = data.get("install_steps")
        if isinstance(steps_raw, str) and steps_raw.strip():
            steps = [steps_raw.strip()]
        elif isinstance(steps_raw, Iterable) and not isinstance(steps_raw, (str, bytes)):
            for item in steps_raw:
                if isinstance(item, str) and item.strip():
                    steps.append(item.strip())

        if manager == "apt-get" and not steps:
            steps = self._default_apt_steps(package)

        if manager == "custom" and not install_command and not steps:
            fallback = self._default_custom_command(package.name)
            if fallback:
                install_command = fallback.install_command
                steps = list(fallback.install_steps)

        return PackageRequest(
            name=package.name,
            manager=manager,
            install_command=install_command,
            install_steps=steps,
        )

    def _resolve_latest_version(self, package_name: str) -> str | None:
        cached = self._latest_version_cache.get(package_name.lower())
        if cached:
            return cached

        prompt = (
            "Provide the latest stable version number for the software named '"
            f"{package_name}' as JSON {{\"version\": \"<number>\"}}. If unknown, return null."
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:  # pragma: no cover - LLM failure path
            self._logger.warning("Latest version lookup failed for %s: %s", package_name, exc)
            return None

        payload = response.get("message", {}).get("content", "")
        data = self._safe_json(payload)
        version = data.get("version")
        if isinstance(version, str):
            sanitized = self._sanitize_version_string(version)
            if sanitized:
                self._latest_version_cache[package_name.lower()] = sanitized
                return sanitized
        return None

    @staticmethod
    def _sanitize_version_string(value: str) -> str | None:
        cleaned = value.strip()
        if not cleaned:
            return None
        if re.match(r"^[0-9]+(?:\.[0-9A-Za-z-]+)*$", cleaned):
            return cleaned
        return None

    @staticmethod
    def _default_apt_steps(package: PackageRequest) -> List[str]:
        commands: List[str] = ["sudo apt-get update"]
        if package.version and package.version.strip() and package.version.strip().lower() != "latest":
            version_suffix = f"={package.version.strip()}"
        else:
            version_suffix = ""
        commands.append(f"sudo apt-get install -y {package.name}{version_suffix}")
        return commands

    @staticmethod
    def _default_custom_command(name: str) -> PackageRequest | None:
        table: Dict[str, PackageRequest] = {
            "ngrok": PackageRequest(
                name="ngrok",
                manager="custom",
                install_steps=[
                    "curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null",
                    "echo 'deb https://ngrok-agent.s3.amazonaws.com/ buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list",
                    "sudo apt-get update",
                    "sudo apt-get install -y ngrok",
                ],
            )
        }
        return table.get(name.lower())


__all__ = ["TicketClassifier", "ClassifierResult", "TicketIntent", "PackageRequest"]
