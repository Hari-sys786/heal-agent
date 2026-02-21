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
except ImportError:
    st = None

    def add_script_run_ctx(_: threading.Thread) -> None:
        return None

from selfheal.agent import AgentConfig, TicketAutomationAgent, build_agent
from selfheal.utils.servicenow_client import ServiceNowClient, ServiceNowConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DashboardConfig:
    refresh_interval_seconds: int = 5
    ticket_limit: int = 20


# ─── Config Loaders (unchanged logic) ────────────────────────────────────────

def load_servicenow_config(settings: Mapping[str, Any]) -> ServiceNowConfig:
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
    enabled_services = _split_csv(_get_str(settings, "AGENT_ENABLED_SERVICES", ""))
    llm_provider, llm_model = _resolve_llm_settings(settings)
    return AgentConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        ollama_host=_get_str(settings, "OLLAMA_HOST"),
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
    return DashboardConfig(
        refresh_interval_seconds=_get_int(settings, "DASHBOARD_REFRESH_INTERVAL", 5),
        ticket_limit=_get_int(settings, "DASHBOARD_TICKET_LIMIT", 20),
    )


def _apply_default_field_overrides(config: ServiceNowConfig, settings: Mapping[str, Any]) -> None:
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


# ─── Custom Theme CSS ────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dark theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .hero-header h1 {
        color: #f1f5f9;
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-connected {
        background: rgba(34,197,94,0.15);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .status-error {
        background: rgba(239,68,68,0.15);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.3);
    }

    /* Metric cards */
    .metric-card {
        background: rgba(30,41,59,0.8);
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.2s;
    }
    .metric-card:hover {
        border-color: rgba(59,130,246,0.3);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-icon {
        font-size: 24px;
        margin-bottom: 8px;
    }

    /* Section headers */
    .section-header {
        color: #f1f5f9;
        font-size: 18px;
        font-weight: 600;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(148,163,184,0.1);
    }

    /* Form styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background: rgba(30,41,59,0.6) !important;
        border: 1px solid rgba(148,163,184,0.2) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(59,130,246,0.5) !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(59,130,246,0.3) !important;
    }

    /* Table styling */
    .ticket-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.1);
    }
    .ticket-table th {
        background: rgba(30,41,59,0.9);
        color: #94a3b8;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid rgba(148,163,184,0.1);
    }
    .ticket-table td {
        background: rgba(15,23,42,0.6);
        color: #e2e8f0;
        padding: 12px 16px;
        font-size: 13px;
        border-bottom: 1px solid rgba(148,163,184,0.05);
    }
    .ticket-table tr:hover td {
        background: rgba(30,41,59,0.8);
    }

    /* State badges */
    .state-new { color: #60a5fa; background: rgba(96,165,250,0.1); padding: 3px 10px; border-radius: 12px; font-size: 12px; }
    .state-progress { color: #fbbf24; background: rgba(251,191,36,0.1); padding: 3px 10px; border-radius: 12px; font-size: 12px; }
    .state-resolved { color: #4ade80; background: rgba(74,222,128,0.1); padding: 3px 10px; border-radius: 12px; font-size: 12px; }
    .state-closed { color: #94a3b8; background: rgba(148,163,184,0.1); padding: 3px 10px; border-radius: 12px; font-size: 12px; }

    /* AI badge */
    .ai-yes { color: #4ade80; font-weight: 600; }
    .ai-no { color: #94a3b8; }

    /* Impact/Urgency badges */
    .priority-high { color: #f87171; }
    .priority-medium { color: #fbbf24; }
    .priority-low { color: #4ade80; }

    /* Log area */
    .log-container {
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 12px;
        padding: 16px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 12px;
        color: #94a3b8;
        max-height: 300px;
        overflow-y: auto;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15,23,42,0.95) !important;
        border-right: 1px solid rgba(148,163,184,0.1) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8 !important;
    }
</style>
"""


def _state_badge(state: str | None) -> str:
    state = str(state or "").strip()
    mapping = {"1": ("New", "new"), "2": ("In Progress", "progress"), "3": ("Resolved", "resolved"), "6": ("Closed", "closed")}
    label, cls = mapping.get(state, (state or "Unknown", "new"))
    return f'<span class="state-{cls}">{label}</span>'


def _ai_badge(done: Any) -> str:
    if str(done).strip().lower() in {"true", "yes", "1"}:
        return '<span class="ai-yes">✅ Yes</span>'
    return '<span class="ai-no">—</span>'


def _priority_class(value: Any) -> str:
    v = str(value or "3").strip()
    if v == "1":
        return "priority-high"
    elif v == "2":
        return "priority-medium"
    return "priority-low"


# ─── Main App ────────────────────────────────────────────────────────────────

def run() -> None:
    settings = _load_settings()
    if st is None:
        raise RuntimeError("Streamlit is not installed.")

    st.set_page_config(
        page_title="Self-Heal Automation",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    dashboard_cfg = load_dashboard_config(settings)
    agent_cfg = load_agent_config(settings)
    sn_config = load_servicenow_config(settings)
    sn_client = ServiceNowClient(sn_config)
    agent = build_agent(sn_client, agent_cfg)

    # ─── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤖 Self-Heal Agent")
        st.markdown("---")
        st.markdown(f"**Instance:** `{sn_config.instance_url}`")
        st.markdown(f"**Table:** `{sn_config.table}`")
        st.markdown(f"**LLM:** `{agent_cfg.llm_provider}` / `{agent_cfg.llm_model}`")
        st.markdown(f"**Dry Run:** `{agent_cfg.dry_run_installs}`")
        st.markdown(f"**Auto Resolve:** `{agent_cfg.auto_resolve}`")
        st.markdown("---")
        st.markdown(f"**Package Manager:** `{agent_cfg.package_manager}`")
        diagnostics = ", ".join(agent_cfg.enabled_diagnostics) if agent_cfg.enabled_diagnostics else "None"
        st.markdown(f"**Diagnostics:** `{diagnostics}`")

    # ─── Hero Header ──────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>🛡️ Self-Heal Automation Control Center</h1>
        <p>AI-powered incident automation • ServiceNow integration • Real-time remediation</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Connection Test ──────────────────────────────────
    try:
        tickets_list = list(sn_client.list_incidents(limit=dashboard_cfg.ticket_limit))
        st.markdown('<span class="status-badge status-connected">● Connected to ServiceNow</span>', unsafe_allow_html=True)
    except RuntimeError as exc:
        st.markdown(f'<span class="status-badge status-error">● Connection Failed</span>', unsafe_allow_html=True)
        st.error(f"ServiceNow error: {exc}")
        tickets_list = []

    # ─── Metrics Row ──────────────────────────────────────
    total_tickets = len(tickets_list)
    ai_done_count = sum(1 for t in tickets_list if str(t.get("ai_done", "")).strip().lower() in {"true", "yes", "1"})
    new_count = sum(1 for t in tickets_list if str(t.get("state", "")).strip() == "1")
    resolved_count = sum(1 for t in tickets_list if str(t.get("state", "")).strip() == "3")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🎫</div>
            <p class="metric-value">{total_tickets}</p>
            <p class="metric-label">Total Tickets</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🆕</div>
            <p class="metric-value">{new_count}</p>
            <p class="metric-label">New / Open</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">✅</div>
            <p class="metric-value">{resolved_count}</p>
            <p class="metric-label">Resolved</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🤖</div>
            <p class="metric-value">{ai_done_count}</p>
            <p class="metric-label">AI Processed</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Two Column Layout ────────────────────────────────
    left_col, right_col = st.columns([1, 2])

    # ─── Create Ticket Form ───────────────────────────────
    with left_col:
        st.markdown('<div class="section-header">📝 Create New Ticket</div>', unsafe_allow_html=True)
        with st.form("incident_form", clear_on_submit=True):
            short_description = st.text_input("Short Description", placeholder="e.g., Install nginx on web-02")
            details = st.text_area("Details", height=120, placeholder="Describe the issue or request...")

            form_col1, form_col2 = st.columns(2)
            with form_col1:
                category = st.selectbox("Category", ["request", "software", "hardware", "network", "inquiry"])
            with form_col2:
                urgency = st.selectbox("Urgency", ["3 - Low", "2 - Medium", "1 - High"])

            submitted = st.form_submit_button("🚀 Submit & Auto-Heal", use_container_width=True)

        if submitted:
            if not short_description.strip():
                st.error("Short description is required.")
            else:
                try:
                    urgency_val = urgency.split(" - ")[0]
                    ticket = sn_client.create_incident(
                        short_description.strip(),
                        details.strip(),
                        category=category,
                        urgency=urgency_val,
                    )
                    ticket_label = ticket.get("number") or ticket.get("sys_id", "(unknown)")
                    st.success(f"✅ Ticket **{ticket_label}** created. Automation triggered!")
                    _launch_agent_thread(agent, ticket)
                except RuntimeError as exc:
                    st.error(f"Failed to create incident: {exc}")

    # ─── Live Tickets Table ───────────────────────────────
    with right_col:
        st.markdown('<div class="section-header">📋 Live Tickets</div>', unsafe_allow_html=True)

        if not tickets_list:
            st.info("No tickets found. Create one to get started!")
        else:
            # Build HTML table
            rows_html = ""
            for ticket in tickets_list:
                number = ticket.get("number") or ticket.get("sys_id", "—")
                short_desc = ticket.get("short_description", "—")
                cat = ticket.get("category", "—")
                impact = ticket.get("impact", "3")
                urg = ticket.get("urgency", "3")
                state = ticket.get("state", "1")
                ai_done = ticket.get("ai_done", False)
                updated = ticket.get("sys_updated_on") or ticket.get("updated_on", "—")

                rows_html += f"""
                <tr>
                    <td><strong>{number}</strong></td>
                    <td>{short_desc}</td>
                    <td>{cat}</td>
                    <td><span class="{_priority_class(impact)}">{impact}</span></td>
                    <td><span class="{_priority_class(urg)}">{urg}</span></td>
                    <td>{_state_badge(state)}</td>
                    <td>{_ai_badge(ai_done)}</td>
                    <td style="font-size:11px;color:#64748b;">{updated}</td>
                </tr>
                """

            table_html = f"""
            <table class="ticket-table">
                <thead>
                    <tr>
                        <th>Ticket</th>
                        <th>Description</th>
                        <th>Category</th>
                        <th>Impact</th>
                        <th>Urgency</th>
                        <th>State</th>
                        <th>AI Done</th>
                        <th>Updated</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Tickets", use_container_width=True):
            st.rerun()


def _launch_agent_thread(agent: TicketAutomationAgent, ticket: Dict[str, Any]) -> None:
    def _worker() -> None:
        try:
            agent.invoke(ticket)
        except Exception:
            logger.exception("Agent invocation failed for %s", ticket.get("sys_id"))

    thread = threading.Thread(target=_worker, name=f"agent-{ticket.get('sys_id', 'ticket')}", daemon=True)
    add_script_run_ctx(thread)
    thread.start()


# ─── Utility Functions (unchanged) ───────────────────────────────────────────

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
        raise RuntimeError(f"Configuration value {key} is required.")
    return value


def _find_config_file() -> Path | None:
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
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
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
