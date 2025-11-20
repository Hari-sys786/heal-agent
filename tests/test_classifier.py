import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src" / "selfheal" / "agent" / "classifier.py"

def _fake_chat(**_: object) -> dict:
    return {"message": {"content": ""}}

sys.modules.setdefault("ollama", types.SimpleNamespace(chat=_fake_chat))
langchain_openai = types.ModuleType("langchain_openai")
class _FakeLCChat:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
    def invoke(self, messages: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(content="")
langchain_openai.ChatOpenAI = _FakeLCChat  # type: ignore[attr-defined]
langchain_openai.AzureChatOpenAI = _FakeLCChat  # type: ignore[attr-defined]
sys.modules.setdefault("langchain_openai", langchain_openai)

langchain_google_genai = types.ModuleType("langchain_google_genai")
langchain_google_genai.ChatGoogleGenerativeAI = _FakeLCChat  # type: ignore[attr-defined]
sys.modules.setdefault("langchain_google_genai", langchain_google_genai)

langchain_core = types.ModuleType("langchain_core")
langchain_core_messages = types.ModuleType("langchain_core.messages")
class _BaseMessage:
    def __init__(self, content: str = "") -> None:
        self.content = content
class SystemMessage(_BaseMessage):
    pass
class HumanMessage(_BaseMessage):
    pass
class AIMessage(_BaseMessage):
    pass
langchain_core_messages.SystemMessage = SystemMessage  # type: ignore[attr-defined]
langchain_core_messages.HumanMessage = HumanMessage  # type: ignore[attr-defined]
langchain_core_messages.AIMessage = AIMessage  # type: ignore[attr-defined]
langchain_core_messages.BaseMessage = _BaseMessage  # type: ignore[attr-defined]
langchain_core.messages = langchain_core_messages  # type: ignore[attr-defined]
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.messages", langchain_core_messages)
langgraph_module = types.ModuleType("langgraph")
langgraph_graph = types.ModuleType("langgraph.graph")
langgraph_graph.END = None
langgraph_graph.StateGraph = types.SimpleNamespace  # type: ignore[assignment]
langgraph_module.graph = langgraph_graph
sys.modules.setdefault("langgraph", langgraph_module)
sys.modules.setdefault("langgraph.graph", langgraph_graph)
selfheal_pkg = types.ModuleType("selfheal")
selfheal_pkg.__path__ = [str(ROOT / "src" / "selfheal")]
selfheal_agent_pkg = types.ModuleType("selfheal.agent")
selfheal_agent_pkg.__path__ = [str(ROOT / "src" / "selfheal" / "agent")]
selfheal_pkg.agent = selfheal_agent_pkg  # type: ignore[attr-defined]
sys.modules.setdefault("selfheal", selfheal_pkg)
sys.modules.setdefault("selfheal.agent", selfheal_agent_pkg)

spec = importlib.util.spec_from_file_location("selfheal.agent.classifier", SRC_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load classifier module for testing")

classifier_module = importlib.util.module_from_spec(spec)
sys.modules["selfheal.agent.classifier"] = classifier_module
spec.loader.exec_module(classifier_module)

TicketClassifier = classifier_module.TicketClassifier
TicketIntent = classifier_module.TicketIntent
PackageRequest = classifier_module.PackageRequest


class SafeJsonTests(unittest.TestCase):
    def test_safe_json_parses_code_fence(self) -> None:
        raw = """```json
        {"intent": "install", "confidence": 0.95, "packages": ["python3.13"]}
        ```"""

        data = TicketClassifier._safe_json(raw)

        self.assertEqual(data["intent"], "install")
        self.assertEqual(data["confidence"], 0.95)
        self.assertEqual(data["packages"], ["python3.13"])

    def test_safe_json_extracts_braced_fragment(self) -> None:
        raw = 'noise before {"intent": "service", "packages": []} noise after'

        data = TicketClassifier._safe_json(raw)

        self.assertEqual(data["intent"], "service")
        self.assertEqual(data["packages"], [])


class IntentOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = TicketClassifier(model="dummy")

    def test_service_intent_overridden_when_packages_present(self) -> None:
        intent = self.classifier._override_intent(  # type: ignore[attr-defined]
            TicketIntent.SERVICE,
            combined_text="please install postgresql on the host",
            packages=[PackageRequest(name="postgresql")],
            services=[],
            installation_keywords=self.classifier.installation_keywords,
            service_keywords=self.classifier.service_keywords,
        )
        self.assertEqual(intent, TicketIntent.INSTALL)

    def test_service_intent_not_overridden_without_install_keyword(self) -> None:
        intent = self.classifier._override_intent(  # type: ignore[attr-defined]
            TicketIntent.SERVICE,
            combined_text="postgres service is down",
            packages=[PackageRequest(name="postgresql")],
            services=["postgresql"],
            installation_keywords=self.classifier.installation_keywords,
            service_keywords=self.classifier.service_keywords,
        )
        self.assertEqual(intent, TicketIntent.SERVICE)

    def test_install_intent_overridden_when_issue_reported(self) -> None:
        intent = self.classifier._override_intent(  # type: ignore[attr-defined]
            TicketIntent.INSTALL,
            combined_text="database connection issue on production",
            packages=[PackageRequest(name="mysql-connector-python")],
            services=[],
            installation_keywords=self.classifier.installation_keywords,
            service_keywords=self.classifier.service_keywords,
        )
        self.assertEqual(intent, TicketIntent.SERVICE)


if __name__ == "__main__":
    unittest.main()
