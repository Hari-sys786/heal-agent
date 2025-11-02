from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Sequence

from selfheal.utils.servicenow_client import ServiceNowClient
from selfheal.utils.shell import CommandResult, run_command


@dataclass(slots=True)
class DiagnosticRoutine:
    """A diagnostic routine containing a command sequence."""

    name: str
    commands: Sequence[Sequence[str]]
    summary: str


@dataclass(slots=True)
class DiagnosticsConfig:
    """Configuration for the diagnostics suite."""

    dry_run: bool = True
    default_routine: str = "generic-service-check"
    enabled_routines: Sequence[str] = ()


class DiagnosticsSuite:
    """Executes service diagnostic routines and records the results."""

    def __init__(self, sn_client: ServiceNowClient, config: DiagnosticsConfig | None = None) -> None:
        self.sn_client = sn_client
        self.config = config or DiagnosticsConfig()
        self._routine_catalog: Dict[str, DiagnosticRoutine] = self._build_catalog()
        self._logger = logging.getLogger(__name__)

    def run(self, sys_id: str, ticket_text: str, *, services: Sequence[str] | None = None) -> list[CommandResult]:
        """Run the most appropriate diagnostic routine for the ticket."""
        routine = self._select_routine(ticket_text, services=services)
        self._logger.info(
            "Running diagnostic routine '%s' for ticket %s (services=%s dry_run=%s)",
            routine.name,
            sys_id,
            services,
            self.config.dry_run,
        )
        self.sn_client.append_work_note(sys_id, f"Running diagnostic routine: {routine.name}")

        service_tokens = self._guess_service_tokens(ticket_text, services=services)

        results: list[CommandResult] = []
        for command_template in routine.commands:
            command = tuple(token.format(**service_tokens) for token in command_template)
            self._logger.debug("Executing diagnostic command for %s: %s", routine.name, command)
            result = run_command(command, dry_run=self.config.dry_run)
            results.append(result)
            status = "ok" if result.succeeded() else f"failed ({result.returncode})"
            self.sn_client.append_work_note(sys_id, f"`{' '.join(command)}` -> {status}")
            if result.succeeded():
                self._logger.debug("Command succeeded for routine %s: %s", routine.name, command)
            else:
                self._logger.warning(
                    "Command failed during routine %s: %s (rc=%s stderr=%s)",
                    routine.name,
                    command,
                    result.returncode,
                    result.stderr.strip(),
                )

        self.sn_client.append_work_note(sys_id, routine.summary)
        self._logger.info("Diagnostic routine '%s' completed for ticket %s", routine.name, sys_id)
        return results

    def supported_services(self) -> tuple[str, ...]:
        """Return the canonical service names the diagnostics suite understands."""
        if self.config.enabled_routines:
            normalized = {name.lower(): name for name in self.config.enabled_routines}
            matches: list[str] = []
            for key in self._routine_catalog.keys():
                original = normalized.get(key.lower())
                if original:
                    matches.append(original)
            if matches:
                return tuple(matches)
        return tuple(self._routine_catalog.keys())

    def _select_routine(self, ticket_text: str, *, services: Sequence[str] | None = None) -> DiagnosticRoutine:
        text = ticket_text.lower()
        enabled = set(name.lower() for name in self.config.enabled_routines) if self.config.enabled_routines else None

        if services:
            for service in services:
                key = service.lower()
                if enabled and key not in enabled:
                    continue
                if key in self._routine_catalog:
                    self._logger.debug("Selected routine %s based on explicit service %s", key, service)
                    return self._routine_catalog[key]

        for name, routine in self._routine_catalog.items():
            if enabled and name.lower() not in enabled:
                continue
            if name.lower() in text:
                self._logger.debug("Selected routine %s based on ticket text match", name)
                return routine

        self._logger.debug("Falling back to default diagnostic routine %s", self.config.default_routine)
        return self._routine_catalog[self.config.default_routine]

    @staticmethod
    def _guess_service_tokens(text: str, *, services: Sequence[str] | None = None) -> Dict[str, str]:
        tokens = {"service": "target"}
        if services:
            for service in services:
                if service:
                    tokens["service"] = service
                    return tokens
        words = text.lower().split()
        for word in words:
            if word.endswith(".service"):
                tokens["service"] = word
                break
            if word in {"nginx", "postgres", "postgresql", "redis"}:
                tokens["service"] = word
                break
        return tokens

    @staticmethod
    def _build_catalog() -> Dict[str, DiagnosticRoutine]:
        return {
            "generic-service-check": DiagnosticRoutine(
                name="generic-service-check",
                commands=(
                    ("systemctl", "status", "{service}.service"),
                    ("journalctl", "-u", "{service}.service", "--no-pager", "-n", "100"),
                ),
                summary="Collected service status and latest logs for review.",
            ),
            "nginx": DiagnosticRoutine(
                name="nginx",
                commands=(
                    ("systemctl", "status", "nginx"),
                    ("journalctl", "-u", "nginx", "--no-pager", "-n", "100"),
                    ("nginx", "-t"),
                ),
                summary="Captured Nginx status, logs, and configuration validation output.",
            ),
            "postgres": DiagnosticRoutine(
                name="postgres",
                commands=(
                    ("systemctl", "status", "postgresql"),
                    ("journalctl", "-u", "postgresql", "--no-pager", "-n", "100"),
                ),
                summary="Gathered PostgreSQL service status and recent logs.",
            ),
        }


__all__ = ["DiagnosticsSuite", "DiagnosticsConfig", "DiagnosticRoutine"]
