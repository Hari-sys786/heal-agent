from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

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


def load_servicenow_config(settings: Mapping[str, Any]) -> ServiceNowConfig:
    """Load ServiceNow credentials from environment variables."""
    assignment_group = _get_str(settings, "SERVICENOW_ASSIGNMENT_GROUP", "").strip() or None
    token = _get_str(settings, "SERVICENOW_TOKEN", "").strip() or None
    username: str | None
    password: str | None
    if token:
        username = None
        password = None
    else:
        username = _require_value(settings, "SERVICENOW_USERNAME")
        password = _require_value(settings, "SERVICENOW_PASSWORD")
    config = ServiceNowConfig(
        instance_url=_require_value(settings, "SERVICENOW_INSTANCE_URL"),
        username=username,
        password=password,
        token=token,
        table=_get_str(settings, "SERVICENOW_TABLE", "u_heal_agent"),
        assignment_group=assignment_group,
        include_display_values=_get_bool(settings, "SERVICENOW_DISPLAY_VALUES", default=False),
    )
    _apply_default_field_overrides(config, settings)
    return config


def load_agent_config(settings: Mapping[str, Any]) -> AgentConfig:
    """Build agent configuration from environment variables."""
    enabled_services = _split_csv(_get_str(settings, "AGENT_ENABLED_SERVICES", ""))
    llm_provider, llm_model = _resolve_llm_settings(settings)

    return AgentConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        azure_openai_endpoint=_get_str(settings, "AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=_get_str(settings, "AZURE_OPENAI_API_KEY"),
        azure_openai_api_version=_get_str(settings, "AZURE_OPENAI_API_VERSION"),
        azure_openai_deployment=_get_str(settings, "AZURE_OPENAI_DEPLOYMENT") if llm_provider == "azure_openai" else None,
        openai_api_key=_get_str(settings, "OPENAI_API_KEY"),
        gemini_api_key=_get_str(settings, "GEMINI_API_KEY"),
        dry_run_installs=_get_bool(settings, "AGENT_DRY_RUN", default=True),
        package_manager=_get_str(settings, "AGENT_PACKAGE_MANAGER", "apt-get"),
        auto_resolve=_get_bool(settings, "AGENT_AUTO_RESOLVE", default=False),
        review_assignment_group=_get_str(settings, "AGENT_REVIEW_GROUP", "Auto-Bot Review"),
        reassignment_group=_get_str(settings, "AGENT_REASSIGN_GROUP", "Service Desk"),
        enabled_diagnostics=tuple(enabled_services),
        resolved_state=_get_str(settings, "AGENT_RESOLVED_STATE", "3"),
        review_state=_get_str(settings, "AGENT_REVIEW_STATE", "2"),
        reassigned_state=_get_str(settings, "AGENT_REASSIGN_STATE", _get_str(settings, "AGENT_REVIEW_STATE", "2")),
        sudo_password=_get_str(settings, "AGENT_SUDO_PASSWORD", "").strip() or None,
    )


def load_dashboard_config(settings: Mapping[str, Any]) -> DashboardConfig:
    """Load dashboard configuration from environment variables."""
    return DashboardConfig(
        refresh_interval_seconds=_get_int(settings, "DASHBOARD_REFRESH_INTERVAL", 5),
        ticket_limit=_get_int(settings, "DASHBOARD_TICKET_LIMIT", 20),
    )


def _apply_default_field_overrides(config: ServiceNowConfig, settings: Mapping[str, Any]) -> None:
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
        value = _get_str(settings, env_var)
        if value is not None and value.strip():
            config.default_fields[field_key] = value.strip()


def _load_environment() -> None:
    """Load environment variables from config.yml if present."""
    config_path = _find_config_file()
    if not config_path:
        logger.debug("No config.yml found; relying on existing environment variables.")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover - config read failure
        logger.warning("Unable to read config.yml: %s", exc)
        return

    mapping = _extract_env_mapping(data)
    for key, value in mapping.items():
        if value is None:
            continue
        os.environ.setdefault(key, _stringify(value))
    logger.debug("Loaded environment from %s", config_path)


def run() -> None:
    """Entrypoint for the Streamlit application."""
    settings = _load_settings()
    if st is None:
        raise RuntimeError("Streamlit is not installed. Run `pip install streamlit` to launch the dashboard.")

    st.set_page_config(page_title="Self-Heal Automation", layout="wide")
    st.title("Self-Heal Automation Control Center")

    dashboard_cfg = load_dashboard_config(settings)
    agent_cfg = load_agent_config(settings)
    if agent_cfg.llm_provider == "ollama":
        logger.info(
            "Using OLLAMA_HOST=%s model=%s",
            (_get_str(settings, "OLLAMA_HOST") or "(default)"),
            agent_cfg.llm_model,
        )
    else:
        logger.info("Using LLM provider=%s model=%s", agent_cfg.llm_provider, agent_cfg.llm_model)

    sn_client = ServiceNowClient(load_servicenow_config(settings))
    agent = build_agent(sn_client, agent_cfg)

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


def _resolve_llm_settings(settings: Mapping[str, Any]) -> tuple[str, str]:
    provider = _normalise_provider(_get_str(settings, "LLM_PROVIDER", "ollama"))
    model = _get_str(settings, "LLM_MODEL")
    if not model:
        if provider == "azure_openai":
            model = _get_str(settings, "AZURE_OPENAI_DEPLOYMENT") or _get_str(settings, "AZURE_OPENAI_MODEL")
        elif provider == "gemini":
            model = _get_str(settings, "GEMINI_MODEL") or "gemini-1.5-flash"
        elif provider == "openai":
            model = _get_str(settings, "OPENAI_MODEL") or "gpt-4o"
        else:
            model = _get_str(settings, "OLLAMA_MODEL", "phi3:latest")
    return provider, model


def _normalise_provider(provider: str | None) -> str:
    if not provider:
        return "ollama"
    normalized = provider.strip().lower().replace("-", "_")
    if normalized in {"azure", "azureopenai"}:
        return "azure_openai"
    if normalized in {"google", "google_ai"}:
        return "gemini"
    return normalized


def _split_csv(value: str) -> Iterable[str]:
    return tuple(filter(None, (item.strip() for item in value.split(",") if item)))


def _get_bool(settings: Mapping[str, Any], key: str, *, default: bool) -> bool:
    if settings and key in settings:
        val = settings[key]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "on"}
    env_val = os.getenv(key)
    if env_val is None:
        return default
    return env_val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(settings: Mapping[str, Any], key: str, default: int) -> int:
    if settings and key in settings and settings[key] is not None:
        try:
            return int(settings[key])
        except (TypeError, ValueError):
            return default
    env_val = os.getenv(key)
    if env_val is None:
        return default
    try:
        return int(env_val)
    except ValueError:
        return default


def _get_str(settings: Mapping[str, Any], key: str, default: str | None = None) -> str | None:
    if settings and key in settings and settings[key] is not None:
        return str(settings[key])
    env_val = os.getenv(key)
    if env_val is None:
        return default
    return env_val


def _require_value(settings: Mapping[str, Any], key: str) -> str:
    value = _get_str(settings, key)
    if not value:
        raise RuntimeError(f"Configuration value {key} is required for the dashboard.")
    return value


def _find_config_file() -> Path | None:
    """Locate config.yml starting from CWD and walking up parents."""
    start = Path.cwd()
    candidates = [start / "config.yml", start / "config.yaml"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for parent in Path(__file__).resolve().parents:
        for name in ("config.yml", "config.yaml"):
            candidate = parent / name
            if candidate.exists():
                return candidate
    return None


def _extract_env_mapping(data: Any) -> Mapping[str, Any]:
    if isinstance(data, Mapping):
        env_block = data.get("env")
        if isinstance(env_block, Mapping):
            return env_block
        return data
    return {}


def _load_settings() -> Dict[str, Any]:
    config_path = _find_config_file()
    if not config_path:
        logger.debug("No config.yml found; relying on existing environment variables.")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to read config.yml: %s", exc)
        return {}
    return dict(_extract_env_mapping(data))


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


if __name__ == "__main__":
    run()
