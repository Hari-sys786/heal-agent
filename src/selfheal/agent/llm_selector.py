from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Protocol, Sequence

import ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI


class ChatModel(Protocol):
    """Minimal interface for chat-style LLM clients."""

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        ...


def _build_openai_model(config: Dict[str, Any]) -> Any:
    settings = config.copy()
    return ChatOpenAI(
        model_name=settings.get("model", settings.get("model_name", "gpt-4o")),
        temperature=settings.get("temperature", 0.3),
        max_tokens=settings.get("max_tokens", 2000),
        timeout=settings.get("timeout", 120),
    )


def _build_azure_openai_model(config: Dict[str, Any]) -> Any:
    settings = config.copy()
    deployment_name = settings.get("deployment_name") or settings.get("deployment")
    if not deployment_name:
        raise ValueError("Azure OpenAI configuration requires 'deployment_name'.")

    azure_key = settings.get("api_key") or settings.get("azure_api_key")
    endpoint = settings.get("endpoint") or settings.get("azure_endpoint")
    api_version = settings.get("api_version") or settings.get("azure_api_version")

    return AzureChatOpenAI(
        deployment_name=deployment_name,
        api_key=azure_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        max_tokens=settings.get("max_tokens", 2000),
        timeout=settings.get("timeout", 120),
    )


def _build_google_model(config: Dict[str, Any]) -> Any:
    settings = config.copy()
    model_name = settings.get("model", "gemini-pro")
    temperature = settings.get("temperature", 0.3)
    max_output_tokens = settings.get("max_output_tokens")
    google_key = settings.get("api_key") or settings.get("google_api_key")
    cred_path = settings.get("credentials_file") or settings.get("google_credentials_file")

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "google_api_key": google_key,
    }
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens
    return ChatGoogleGenerativeAI(**kwargs)


def _build_ollama_model(config: Dict[str, Any]) -> Any:
    model_name = config.get("model", "phi3:latest")
    host = config.get("host") or config.get("endpoint")
    client: Any
    if host and hasattr(ollama, "Client"):
        try:
            client = ollama.Client(host=host)
        except Exception:
            client = ollama
    else:
        client = ollama

    class _OllamaWrapper:
        def __init__(self, backend: Any) -> None:
            self._backend = backend

        def invoke(self, messages: Any) -> str:
            if isinstance(messages, str):
                payload = [{"role": "user", "content": messages}]
            else:
                payload = [{"role": m["role"], "content": m["content"]} for m in messages]
            chat_fn = getattr(self._backend, "chat", None)
            if chat_fn is None:
                chat_fn = ollama.chat
            resp = chat_fn(model=model_name, messages=payload)
            return resp.get("message", {}).get("content", "")

    return _OllamaWrapper(client)


def build_chat_model(config: Dict[str, Any]) -> Any:
    """
    Construct a LangChain chat model based on configuration.

    The configuration is expected to contain either:
      - an `llm` block with a `provider` key and provider-specific settings
      - a legacy `openai` block (provider implicitly OpenAI)
    """
    llm_config: Dict[str, Any] = config.get("llm", {}) if config else {}
    provider = llm_config.get("provider")

    # Backwards compatibility: fall back to legacy 'openai' block
    if not provider and "openai" in config:
        provider = "openai"
        llm_config = config.get("openai", {}).copy()

    provider = _normalise_provider(provider or "openai")

    if provider in {"openai", "chatgpt"}:
        openai_settings = llm_config.get("options", llm_config) if llm_config else {}
        if "provider" in openai_settings:
            openai_settings = {k: v for k, v in openai_settings.items() if k != "provider"}
        if not openai_settings and "openai" in config:
            openai_settings = config.get("openai", {})
        return _build_openai_model(openai_settings)

    if provider in {"azure", "azure-openai", "azure_openai"}:
        azure_settings = llm_config.get("options", llm_config) if llm_config else {}
        if "provider" in azure_settings:
            azure_settings = {k: v for k, v in azure_settings.items() if k != "provider"}
        return _build_azure_openai_model(azure_settings)

    if provider in {"google", "google-genai", "gemini"}:
        google_settings = llm_config.get("options", llm_config)
        if "provider" in google_settings:
            google_settings = {k: v for k, v in google_settings.items() if k != "provider"}
        return _build_google_model(google_settings)

    if provider in {"ollama"}:
        ollama_settings = llm_config.get("options", llm_config)
        if "provider" in ollama_settings:
            ollama_settings = {k: v for k, v in ollama_settings.items() if k != "provider"}
        return _build_ollama_model(ollama_settings)

    raise ValueError(f"Unsupported LLM provider '{provider}'.")


def build_default_llm_client(model: str = "phi3:latest") -> ChatModel:
    """Default Ollama-backed client when none is supplied."""
    cfg: Dict[str, Any] = {"llm": {"provider": "ollama", "options": {"model": model}}}
    return LangChainChatAdapter(build_chat_model(cfg))


def build_llm_client_for_agent(cfg: Any) -> ChatModel:
    """Build an LLM client using AgentConfig-like attributes."""
    provider = _normalise_provider(getattr(cfg, "llm_provider", None) or "ollama")
    model = getattr(cfg, "llm_model", None)
    config: Dict[str, Any] = {"llm": {"provider": provider, "options": {}}}
    opts = config["llm"]["options"]

    if provider == "azure_openai":
        opts["deployment_name"] = getattr(cfg, "azure_openai_deployment", None) or model
        opts["api_key"] = getattr(cfg, "azure_openai_api_key", None)
        opts["endpoint"] = getattr(cfg, "azure_openai_endpoint", None)
        opts["api_version"] = getattr(cfg, "azure_openai_api_version", None)
    elif provider == "openai":
        opts["model"] = model or getattr(cfg, "openai_model", None) or "gpt-4o"
        opts["api_key"] = getattr(cfg, "openai_api_key", None)
    elif provider == "gemini":
        opts["model"] = model or getattr(cfg, "gemini_model", None) or "gemini-1.5-flash"
        opts["api_key"] = getattr(cfg, "gemini_api_key", None)
    else:  # ollama
        opts["model"] = model or "phi3:latest"
        host = getattr(cfg, "ollama_host", None) or os.getenv("OLLAMA_HOST")
        if host:
            opts["host"] = host

    lc_model = build_chat_model(config)
    return LangChainChatAdapter(lc_model)


def _normalise_provider(provider: str | None) -> str:
    if not provider:
        return "ollama"
    normalized = provider.strip().lower().replace("-", "_")
    if normalized in {"azure", "azureopenai"}:
        return "azure_openai"
    if normalized in {"google", "google_ai"}:
        return "gemini"
    return normalized


class LangChainChatAdapter:
    """Adapter to align LangChain/Ollama chat models with our ChatModel protocol."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        prompt = _flatten_messages(messages)
        response = self._llm.invoke(prompt)

        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(part) for part in content)
        if isinstance(response, str):
            return response
        if response is not None and hasattr(response, "get"):
            # Ollama chat payload or dict-like
            maybe = response.get("message", {}).get("content") if isinstance(response, dict) else None
            return maybe or str(response)
        return ""


def _flatten_messages(messages: Sequence[Mapping[str, str]]) -> str:
    chunks = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        chunks.append(f"{role}:\n{content}")
    return "\n\n".join(chunks)


__all__ = [
    "ChatModel",
    "build_chat_model",
    "build_default_llm_client",
    "build_llm_client_for_agent",
]
