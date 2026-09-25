import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.errors import GraphRecursionError

from agent import build_graph, verify_output
from server import ArticleService, validate_articles


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "articles.json"
        self.output = Path(self.tmp.name) / "out.json"
        self.rows = [{"id": "1", "title": "Тест", "text": "Лимит — 9000 рублей."},
                     {"id": "2", "title": "Срок", "text": "Срок — 7 дней."}]
        self.source.write_text(json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")
        self.service = ArticleService(self.source, self.output, "test-model")

    def test_invalid_data(self):
        for invalid in ([], {}, [{"id": 1, "text": ""}], self.rows + [self.rows[0]],
                        [{"id": True, "text": "a"}], [{"id": "x", "text": "x", "title": 1}]):
            with self.subTest(data=invalid), self.assertRaises(ValueError):
                validate_articles(invalid)

    def test_wrapper_and_integer_id(self):
        self.assertEqual(validate_articles({"articles": [{"id": 7, "text": "текст"}]})[0]["id"], "7")

    async def test_order_and_path_guards(self):
        with self.assertRaises(ValueError):
            await self.service.summarize_article("1")
        with self.assertRaises(ValueError):
            self.service.read_articles(str(self.output))
        self.service.read_articles(str(self.source))
        with self.assertRaises(ValueError):
            self.service.save_results(str(self.output))
        with self.assertRaises(ValueError):
            self.service.save_results(str(self.source))
        with self.assertRaises(ValueError):
            await self.service.summarize_article("missing")
        self.assertFalse(self.output.exists())

    async def test_save_cache_and_preserve_source(self):
        self.service.read_articles(str(self.source))
        self.service.generate_summary = AsyncMock(side_effect=["Лимит — 9000 рублей.", "Срок — 7 дней."])
        await self.service.summarize_article("1")
        await self.service.summarize_article("1")
        self.assertEqual(self.service.generate_summary.await_count, 1)
        self.service.read_articles(str(self.source))
        await self.service.summarize_article("2")
        self.service.save_results(str(self.output))
        self.assertEqual(verify_output(self.source, self.output)["count"], 2)
        self.assertEqual(self.service.save_results(str(self.output))["count"], 2)
        other = ArticleService(self.source, self.output, "test-model")
        other.read_articles(str(self.source))
        other.summaries = self.service.summaries
        with self.assertRaises(FileExistsError):
            other.save_results(str(self.output))

    async def test_empty_summary_rejected(self):
        self.service.read_articles(str(self.source))
        self.service.generate_summary = AsyncMock(return_value="  ")
        with self.assertRaises(ValueError):
            await self.service.summarize_article("1")
        self.assertEqual(self.service.summaries, {})

    async def test_real_mcp_and_graph_without_api(self):
        # Real stdio server, protocol, adapter and graph. Only LLM decisions are scripted.
        # This is an infrastructure test, NOT evidence of a real Groq run.
        env = dict(os.environ, AGENT_INPUT=str(self.source), AGENT_OUTPUT=str(self.output))
        params = StdioServerParameters(command=sys.executable, args=["-m", "server"], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                self.assertEqual({t.name for t in tools}, {"read_articles", "summarize_article", "save_results"})
                sequence = [
                    AIMessage(content="", tool_calls=[{"name": "read_articles", "args": {"path": str(self.source)}, "id": "call1"}]),
                    AIMessage(content="", tool_calls=[{"name": "save_results", "args": {"path": str(self.output)}, "id": "call2"}]),
                    AIMessage(content="Finished infrastructure test; save must fail."),
                ]

                class ScriptedModel:
                    last_messages = []

                    def bind_tools(self, available):
                        return self

                    async def ainvoke(self, messages):
                        self.last_messages = messages
                        return sequence.pop(0) if sequence else AIMessage(content="")

                model = ScriptedModel()
                graph = build_graph(model, tools)
                with self.assertRaises(GraphRecursionError):
                    await graph.ainvoke({"messages": [HumanMessage(content="Test")]},
                                        config={"recursion_limit": 10})
                replies = [m for m in model.last_messages if m.type == "tool"]
                self.assertEqual(len(replies), 2)
                self.assertIn("Summarize these IDs", str(replies[1].content))
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
