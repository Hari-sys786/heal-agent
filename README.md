# Self-Heal Automation Platform

This repository contains a Streamlit dashboard and LangGraph-powered automation agent that create ServiceNow incidents and immediately launch remediation or diagnostic routines on Linux hosts.

## Key Components

- `src/selfheal/dashboard/` - Streamlit UI that files tickets and kicks off the agent asynchronously.
- `src/selfheal/agent/` - LangGraph graph, LLM-backed classifier (Ollama, OpenAI/Azure OpenAI, or Gemini), installers, diagnostics, and ticket updater.
- `src/selfheal/utils/` - ServiceNow REST client and shell command helpers shared across the project.
- `.env.example` - Sample environment configuration (credentials, table name, default ticket fields, feature flags).

## Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -e .
   pip install streamlit
   ```
2. Copy `.env.example` to `.env` and populate ServiceNow credentials plus agent settings (LLM provider/model and keys). Supply either `SERVICENOW_USERNAME`/`SERVICENOW_PASSWORD` or `SERVICENOW_TOKEN`. The dashboard automatically loads this file via `python-dotenv`.
   - Optional: set `AGENT_SUDO_PASSWORD` if the automation must supply a sudo password for package installs. The password is piped to `sudo -S`; consider configuring passwordless sudo instead for production use.
   - Package installs now respect explicit versions (e.g. `postgresql=15.2`) and may execute model-provided shell commands or ordered step lists for software that is not available via your configured package manager.
3. Launch the dashboard:
   ```bash
   streamlit run src/selfheal/dashboard/app.py
   ```

## LLM Configuration

Set `LLM_PROVIDER` to `ollama`, `openai`, `azure_openai`, or `gemini`. Use `LLM_MODEL` or the provider-specific vars:
- Ollama: `OLLAMA_MODEL` (and optional `OLLAMA_HOST`)
- OpenAI: `OPENAI_MODEL` (defaults to `gpt-4o`), `OPENAI_API_KEY`
- Azure OpenAI: `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, optional `AZURE_OPENAI_API_VERSION`
- Gemini: `GEMINI_MODEL` (defaults to `gemini-1.5-flash`), `GEMINI_API_KEY`

When a ticket is submitted, the ServiceNow record is created with default metadata (assignment group, caller, category, etc.) owned by the automation team. The LangGraph agent calls the configured LLM to classify the request, runs package install routines (dry-run by default) or diagnostic playbooks, and writes work notes plus state transitions back to ServiceNow for audit and compliance.
