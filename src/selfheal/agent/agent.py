from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from selfheal.agent.classifier import ClassifierResult, TicketClassifier, TicketIntent
from selfheal.agent.diagnostics import DiagnosticsConfig, DiagnosticsSuite
from selfheal.agent.installers import InstallerConfig, PackageInstaller
from selfheal.agent.ticket_updater import TicketUpdater, TicketUpdaterConfig
from selfheal.utils.servicenow_client import ServiceNowClient

logger = logging.getLogger(__name__)


class TicketState(TypedDict, total=False):
    """Graph state shared across nodes."""

    ticket: Dict[str, Any]
    classification: Dict[str, Any]
    outcome: Dict[str, Any]
    logs: List[str]


@dataclass(slots=True)
class AgentConfig:
    """Runtime configuration for the automation agent."""

    ollama_model: str = "phi3:latest"
    dry_run_installs: bool = True
    package_manager: str = "apt-get"
    auto_resolve: bool = False
    review_assignment_group: str = "Auto-Bot Review"
    reassignment_group: str = "Service Desk"
    enabled_diagnostics: tuple[str, ...] = ()
    resolved_state: str = "3"
    review_state: str = "2"
    reassigned_state: str = "2"


class TicketAutomationAgent:
    """Wrapper around the compiled LangGraph workflow."""

    def __init__(
        self,
        *,
        graph,
    ) -> None:
        self._graph = graph

    def invoke(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the automation graph with the provided ticket payload."""
        initial_state: TicketState = {"ticket": ticket}
        return self._graph.invoke(initial_state)  # type: ignore[return-value]


def build_agent(sn_client: ServiceNowClient, config: Optional[AgentConfig] = None) -> TicketAutomationAgent:
    """Construct the LangGraph workflow and expose a simple interface."""
    cfg = config or AgentConfig()
    logger.debug("Building automation agent with config: %s", cfg)

    classifier = TicketClassifier(model=cfg.ollama_model)
    installer = PackageInstaller(
        sn_client,
        InstallerConfig(package_manager=cfg.package_manager, dry_run=cfg.dry_run_installs),
    )
    diagnostics = DiagnosticsSuite(
        sn_client,
        DiagnosticsConfig(dry_run=cfg.dry_run_installs, enabled_routines=cfg.enabled_diagnostics),
    )
    updater = TicketUpdater(
        sn_client,
        TicketUpdaterConfig(
            auto_resolve=cfg.auto_resolve,
            review_assignment_group=cfg.review_assignment_group,
            resolved_state=cfg.resolved_state,
            review_state=cfg.review_state,
            reassigned_state=cfg.reassigned_state,
        ),
    )

    graph = StateGraph(TicketState)

    def classify_node(state: TicketState) -> Dict[str, Any]:
        ticket = state["ticket"]
        short_description = ticket.get("short_description", "")
        description = ticket.get("description", "")
        logger.info(
            "Classifying ticket %s with short_description=%s",
            ticket.get("sys_id"),
            short_description,
        )
        result = classifier.classify(short_description, description)
        logs = state.get("logs", [])
        logs.append(f"Classifier intent: {result.intent.value} (confidence={result.confidence:.2f})")
        logger.debug(
            "Classification result for %s: intent=%s confidence=%.2f packages=%s services=%s",
            ticket.get("sys_id"),
            result.intent.value,
            result.confidence,
            result.packages,
            result.services,
        )
        return {
            "classification": result.to_dict(),
            "logs": logs,
        }

    def route_from_classification(state: TicketState) -> str:
        intent_value = state["classification"]["intent"]
        logger.info(
            "Routing ticket %s based on intent %s",
            state["ticket"].get("sys_id"),
            intent_value,
        )
        if intent_value == TicketIntent.INSTALL.value:
            return "install"
        if intent_value == TicketIntent.SERVICE.value:
            return "diagnostics"
        return "fallback"

    def installation_node(state: TicketState) -> Dict[str, Any]:
        ticket = state["ticket"]
        classification = state["classification"]
        classifier_packages = list(classification.get("packages") or [])
        sys_id = ticket["sys_id"]
        full_text = " ".join(
            filter(
                None,
                (ticket.get("short_description"), ticket.get("description")),
            )
        )

        candidate_packages = ticket.get("packages") or classifier_packages or installer.extract_packages(full_text)
        packages = _dedupe_sequence(candidate_packages)
        logger.debug(
            "Resolved package list for %s: %s (classifier=%s ticket_packages=%s)",
            sys_id,
            packages,
            classifier_packages,
            ticket.get("packages"),
        )

        if not packages:
            note = "No packages identified; escalating for manual review."
            updater.add_work_note(sys_id, note)
            logger.warning("No packages found for ticket %s; escalating for review", sys_id)
            return {
                "outcome": {
                    "type": "install",
                    "status": "needs_review",
                    "reason": "missing_packages",
                    "packages": [],
                    "commands": [],
                    "note": note,
                }
            }

        logger.info("Starting package installation for %s: %s", sys_id, packages)
        results = installer.install(sys_id, packages)
        commands = [result.to_dict() for result in results]
        finalize_note = "Install commands executed; awaiting verification."

        if cfg.auto_resolve and not cfg.dry_run_installs:
            updater.mark_resolved(sys_id, "Automation installed requested software.")
            status = "resolved"
        else:
            updater.mark_for_review(sys_id, finalize_note)
            status = "pending_review"
        logger.info("Installation outcome for %s: status=%s dry_run=%s", sys_id, status, cfg.dry_run_installs)

        return {
            "outcome": {
                "type": "install",
                "status": status,
                "classifier_packages": classifier_packages,
                "packages": packages,
                "commands": commands,
                "note": finalize_note,
            }
        }

    def diagnostics_node(state: TicketState) -> Dict[str, Any]:
        ticket = state["ticket"]
        classification = state["classification"]
        services = list(classification.get("services") or [])
        sys_id = ticket["sys_id"]
        description = ticket.get("description", "")
        short_description = ticket.get("short_description", "")
        text = f"{short_description}\n{description}"

        logger.info("Running diagnostics for %s (services=%s)", sys_id, services)
        results = diagnostics.run(sys_id, text, services=services)
        command_dicts = [result.to_dict() for result in results]
        finalize_note = "Diagnostics complete; review findings in work notes."

        if cfg.auto_resolve and not cfg.dry_run_installs:
            updater.mark_resolved(sys_id, "Diagnostics executed automatically.")
            status = "resolved"
        else:
            updater.mark_for_review(sys_id, finalize_note)
            status = "pending_review"
        logger.info("Diagnostics outcome for %s: status=%s dry_run=%s", sys_id, status, cfg.dry_run_installs)

        return {
            "outcome": {
                "type": "diagnostics",
                "status": status,
                "services": services,
                "commands": command_dicts,
                "note": finalize_note,
            }
        }

    def fallback_node(state: TicketState) -> Dict[str, Any]:
        ticket = state["ticket"]
        sys_id = ticket["sys_id"]
        note = "Ticket marked out-of-scope; reassigned to service desk."
        logger.info("Ticket %s routed to fallback path", sys_id)
        updater.reassign_out_of_scope(sys_id, cfg.reassignment_group, note)
        return {
            "outcome": {
                "type": "fallback",
                "status": "reassigned",
                "note": note,
            }
        }

    def finalize_node(state: TicketState) -> Dict[str, Any]:
        # Final node simply returns accumulated state; updates executed earlier.
        logger.debug(
            "Finalizing ticket %s with outcome=%s",
            state["ticket"].get("sys_id"),
            state.get("outcome"),
        )
        return {
            "ticket": state["ticket"],
            "classification": state.get("classification"),
            "outcome": state.get("outcome"),
            "logs": state.get("logs", []),
        }

    graph.add_node("classify", classify_node)
    graph.add_node("install", installation_node)
    graph.add_node("diagnostics", diagnostics_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_from_classification,
        {
            "install": "install",
            "diagnostics": "diagnostics",
            "fallback": "fallback",
        },
    )
    graph.add_edge("install", "finalize")
    graph.add_edge("diagnostics", "finalize")
    graph.add_edge("fallback", "finalize")
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    return TicketAutomationAgent(graph=compiled)


def _dedupe_sequence(values: Iterable[Any] | None) -> list[Any]:
    if not values:
        return []
    seen = set()
    result: list[Any] = []
    for value in values:
        if value is None:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["AgentConfig", "TicketAutomationAgent", "TicketState", "build_agent"]
