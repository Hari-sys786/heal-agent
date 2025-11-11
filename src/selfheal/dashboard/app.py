from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from dotenv import find_dotenv, load_dotenv

# Allow running as `python src/selfheal/dashboard/app.py` without installing the package.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except ImportError:  # pragma: no cover - dashboard only runs when Streamlit is installed
    st = None  # type: ignore[assignment]

    def add_script_run_ctx(_: threading.Thread) -> None:  # type: ignore[misc]
        """Fallback no-op when Streamlit runtime context is unavailable."""
        return None

from selfheal.agent import AgentConfig, TicketAutomationAgent, build_agent
from selfheal.utils.servicenow_client import ServiceNowClient, ServiceNowConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DashboardConfig:
    """Configuration for the Streamlit dashboard."""

    refresh_interval_seconds: int = 5
    ticket_limit: int = 20


def load_servicenow_config() -> ServiceNowConfig:
    """Load ServiceNow credentials from environment variables."""
    assignment_group = os.getenv("SERVICENOW_ASSIGNMENT_GROUP", "").strip() or None
    token = os.getenv("SERVICENOW_TOKEN", "").strip() or None
    username: str | None
    password: str | None
    if token:
        username = None
        password = None
    else:
        username = _require_env("SERVICENOW_USERNAME")
        password = _require_env("SERVICENOW_PASSWORD")
    config = ServiceNowConfig(
        instance_url=_require_env("SERVICENOW_INSTANCE_URL"),
        username=username,
        password=password,
        token=token,
        table=os.getenv("SERVICENOW_TABLE", "u_heal_agent"),
        assignment_group=assignment_group,
        include_display_values=_env_bool("SERVICENOW_DISPLAY_VALUES", default=False),
    )
    _apply_default_field_overrides(config)
    return config


def load_agent_config() -> AgentConfig:
    """Build agent configuration from environment variables."""
    enabled_services = _split_csv(os.getenv("AGENT_ENABLED_SERVICES", ""))

    return AgentConfig(
        ollama_model=os.getenv("OLLAMA_MODEL", "phi3:latest"),
        dry_run_installs=_env_bool("AGENT_DRY_RUN", default=True),
        package_manager=os.getenv("AGENT_PACKAGE_MANAGER", "apt-get"),
        auto_resolve=_env_bool("AGENT_AUTO_RESOLVE", default=False),
        review_assignment_group=os.getenv("AGENT_REVIEW_GROUP", "Auto-Bot Review"),
        reassignment_group=os.getenv("AGENT_REASSIGN_GROUP", "Service Desk"),
        enabled_diagnostics=tuple(enabled_services),
        resolved_state=os.getenv("AGENT_RESOLVED_STATE", "3"),
        review_state=os.getenv("AGENT_REVIEW_STATE", "2"),
        reassigned_state=os.getenv("AGENT_REASSIGN_STATE", os.getenv("AGENT_REVIEW_STATE", "2")),
        sudo_password=os.getenv("AGENT_SUDO_PASSWORD", "").strip() or None,
    )


def load_dashboard_config() -> DashboardConfig:
    """Load dashboard configuration from environment variables."""
    return DashboardConfig(
        refresh_interval_seconds=int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "5")),
        ticket_limit=int(os.getenv("DASHBOARD_TICKET_LIMIT", "20")),
    )


def _apply_default_field_overrides(config: ServiceNowConfig) -> None:
    """Allow optional overrides of default ticket fields via environment variables."""
    overrides = {
        "SERVICENOW_DEFAULT_CALLER_ID": "caller_id",
        "SERVICENOW_DEFAULT_CATEGORY": "category",
        "SERVICENOW_DEFAULT_SUBCATEGORY": "subcategory",
        "SERVICENOW_DEFAULT_IMPACT": "impact",
        "SERVICENOW_DEFAULT_URGENCY": "urgency",
        "SERVICENOW_DEFAULT_STATE": "state",
    }
    for env_var, field_key in overrides.items():
        value = os.getenv(env_var)
        if value is not None and value.strip():
            config.default_fields[field_key] = value.strip()


def _load_environment() -> None:
    """Load environment variables from the nearest .env file."""
    dotenv_path = find_dotenv(filename=".env", usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        logger.debug("Loaded environment from %s", dotenv_path)
        return

    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            logger.debug("Loaded environment from %s", candidate)
            return


def run() -> None:
    """Entrypoint for the Streamlit application."""
    _load_environment()
    if st is None:
        raise RuntimeError("Streamlit is not installed. Run `pip install streamlit` to launch the dashboard.")

    logger.info("Using OLLAMA_HOST=%s", os.getenv("OLLAMA_HOST") or "(default)")

    st.set_page_config(page_title="Self-Heal Automation", layout="wide")
    st.title("Self-Heal Automation Control Center")

    dashboard_cfg = load_dashboard_config()
    sn_client = ServiceNowClient(load_servicenow_config())
    agent = build_agent(sn_client, load_agent_config())

    _render_ticket_form(sn_client, agent)
    _render_live_tickets(sn_client, dashboard_cfg)


def _render_ticket_form(sn_client: ServiceNowClient, agent: TicketAutomationAgent) -> None:
    st.subheader("Create ServiceNow Ticket")
    with st.form("incident_form"):
        short_description = st.text_input("Short Description", max_chars=160)
        details = st.text_area("Details", height=200)
        submitted = st.form_submit_button("Submit")

    if not submitted:
        return

    if not short_description.strip():
        st.error("Short description is required.")
        return

    try:
        ticket = sn_client.create_incident(short_description.strip(), details.strip())
    except RuntimeError as exc:
        st.error(f"Failed to create incident: {exc}")
        return

    ticket_label = ticket.get("number") or ticket.get("short_description") or ticket.get("sys_id", "(unknown)")
    st.success(f"Ticket {ticket_label} created. Automation has been triggered.")
    _launch_agent_thread(agent, ticket)


def _render_live_tickets(sn_client: ServiceNowClient, config: DashboardConfig) -> None:
    st.subheader("My Tickets")
    placeholder = st.empty()

    try:
        tickets = list(sn_client.list_incidents(limit=config.ticket_limit))
    except RuntimeError as exc:
        placeholder.error(f"Unable to fetch tickets from ServiceNow: {exc}")
        st.stop()
        return

    if not tickets:
        placeholder.info("No tickets assigned to Auto-Bot.")
        return

    table_data = {
        "Ticket": [],
        "Short Description": [],
        "Details": [],
        "Category": [],
        "Subcategory": [],
        "Impact": [],
        "Urgency": [],
        "State": [],
        "AI Done": [],
        "Updated": [],
    }

    for ticket in tickets:
        table_data["Ticket"].append(ticket.get("number") or ticket.get("sys_id"))
        table_data["Short Description"].append(ticket.get("short_description"))
        table_data["Details"].append(ticket.get("description"))
        table_data["Category"].append(ticket.get("category"))
        table_data["Subcategory"].append(ticket.get("subcategory"))
        table_data["Impact"].append(ticket.get("impact"))
        table_data["Urgency"].append(ticket.get("urgency"))
        table_data["State"].append(ticket.get("state"))
        table_data["AI Done"].append("Yes" if ticket.get("ai_done") else "No")
        table_data["Updated"].append(ticket.get("sys_updated_on") or ticket.get("updated_on"))

    placeholder.table(table_data)
    if st.button("Refresh tickets"):
        st.rerun()
    st.caption(f"Recommended refresh cadence: every {config.refresh_interval_seconds}s.")


def _launch_agent_thread(agent: TicketAutomationAgent, ticket: Dict[str, Any]) -> None:
    def _worker() -> None:
        try:
            agent.invoke(ticket)
        except Exception:  # pragma: no cover - background thread logging
            logger.exception("Agent invocation failed for %s", ticket.get("sys_id"))

    thread = threading.Thread(target=_worker, name=f"agent-{ticket.get('sys_id', 'ticket')}", daemon=True)
    add_script_run_ctx(thread)
    thread.start()


def _split_csv(value: str) -> Iterable[str]:
    return tuple(filter(None, (item.strip() for item in value.split(",") if item)))


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required for the dashboard.")
    return value


if __name__ == "__main__":
    run()
