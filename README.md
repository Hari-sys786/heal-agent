# Self-Heal Automation Platform

This repository contains a Streamlit dashboard and LangGraph-powered automation agent that create ServiceNow incidents and immediately launch remediation or diagnostic routines on Linux hosts.

## Key Components

- `src/selfheal/dashboard/` — Streamlit UI that files tickets and kicks off the agent asynchronously.
- `src/selfheal/agent/` — LangGraph graph, Ollama-backed classifier, installers, diagnostics, and ticket updater.
- `src/selfheal/utils/` — ServiceNow REST client and shell command helpers shared across the project.
- `.env.example` — Sample environment configuration (credentials, table name, default ticket fields, feature flags).

## Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -e .
   pip install streamlit
   ```
2. Copy `.env.example` to `.env` and populate ServiceNow credentials plus agent settings (including your Ollama model name). Supply either `SERVICENOW_USERNAME`/`SERVICENOW_PASSWORD` or `SERVICENOW_TOKEN`. The dashboard automatically loads this file via `python-dotenv`.
   - Optional: set `AGENT_SUDO_PASSWORD` if the automation must supply a sudo password for package installs. The password is piped to `sudo -S`; consider configuring passwordless sudo instead for production use.
   - Package installs now respect explicit versions (e.g. `postgresql=15.2`) and may execute model-provided shell commands or ordered step lists for software that is not available via your configured package manager.
3. Launch the dashboard:
   ```bash
   streamlit run src/selfheal/dashboard/app.py
   ```

When a ticket is submitted, the ServiceNow record is created with default metadata (assignment group, caller, category, etc.) owned by the automation team. The LangGraph agent calls an Ollama LLM to classify the request, runs package install routines (dry-run by default) or diagnostic playbooks, and writes work notes plus state transitions back to ServiceNow for audit and compliance. Ensure an Ollama server is running with the configured model (see `.env.example`).
