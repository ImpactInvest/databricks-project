"""Tool-calling agent implemented as an MLflow ResponsesAgent.

The loop is written out explicitly rather than hidden in a framework:

1. Send the conversation plus two tool definitions to the LLM.
2. If the LLM asks for tools, run them, append the results, and go to 1.
3. Stop when the LLM answers in plain text, or when the tool budget
   (``agent.max_tool_calls``, default 6) is used up. In that case the LLM is
   asked for a final answer with tools disabled, so the user always gets a
   reply rather than an error.

The LLM is a Databricks Foundation Model API endpoint called through the
OpenAI-compatible client built in ``agent/llm.py``.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agent.config import Settings
from agent.llm import make_llm_client
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools import (
    ReviewSearch,
    Text2Sql,
    make_executor,
    result_for_llm,
    summarise_tables,
    tool_specs,
)

CITATION_PATTERN = re.compile(r"review[\s_#:]*(\d+)", re.IGNORECASE)


@dataclass
class TurnState:
    """What happened during one user turn; returned as ``custom_outputs``."""

    tool_calls: int = 0
    sql_queries: list[str] = field(default_factory=list)
    sql_results: list[dict] = field(default_factory=list)
    retrieved_reviews: list[dict] = field(default_factory=list)
    stopped_by_budget: bool = False

    def as_custom_outputs(self, answer: str) -> dict:
        retrieved_ids = {r["review_id"] for r in self.retrieved_reviews}
        cited = sorted({int(m) for m in CITATION_PATTERN.findall(answer or "")} & retrieved_ids)
        return {
            "tool_calls": self.tool_calls,
            "sql_queries": self.sql_queries,
            "sql_results": self.sql_results,
            "retrieved_reviews": self.retrieved_reviews,
            "cited_review_ids": cited,
            "stopped_by_budget": self.stopped_by_budget,
            "trace_id": mlflow.get_active_trace_id(),
        }


class BakehouseAgent(ResponsesAgent):
    def __init__(
        self,
        settings: dict | Settings,
        llm_client=None,
        text2sql=None,
        review_search=None,
        spark=None,
    ):
        self.settings = settings if isinstance(settings, Settings) else Settings(settings)
        self._llm = llm_client
        self._text2sql = text2sql
        self._review_search = review_search
        self._spark = spark

    # Clients are created lazily so the class can be imported, logged and unit
    # tested without a Databricks connection.
    @property
    def llm(self):
        if self._llm is None:
            self._llm = make_llm_client()
        return self._llm

    @property
    def text2sql(self) -> Text2Sql:
        if self._text2sql is None:
            executor = make_executor(self.settings, spark=self._spark)
            self._text2sql = Text2Sql(
                self.settings, self.llm, executor, self.settings["agent"]["schema_context"]
            )
        return self._text2sql

    @property
    def review_search(self) -> ReviewSearch:
        if self._review_search is None:
            self._review_search = ReviewSearch(self.settings)
        return self._review_search

    # ------------------------------------------------------------------

    def _run_tool(self, name: str, raw_args: str, state: TurnState) -> str:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return json.dumps(
                {"error": {"type": "bad_arguments", "message": "Arguments were not valid JSON."}}
            )

        if name == "query_sales_data":
            result = self.text2sql(str(args.get("question", "")))
            state.sql_queries.append(result.get("sql"))
            if "error" not in result:
                state.sql_results.append(
                    {
                        "sql": result["sql"],
                        "columns": result["columns"],
                        "rows": result["rows"][:20],
                        "row_count": result["row_count"],
                    }
                )
            return result_for_llm(result, self.settings["sql"]["rows_shown_to_llm"])

        if name == "search_reviews":
            k = min(max(int(args.get("k") or self.settings["ai_search"]["default_k"]), 1), 10)
            franchise_id = args.get("franchise_id")
            hits = self.review_search(str(args.get("query", "")), franchise_id=franchise_id, k=k)
            state.retrieved_reviews.extend(hits)
            if not hits:
                return json.dumps({"results": [], "note": "No matching reviews."})
            return json.dumps({"results": hits}, default=str)

        return json.dumps({"error": {"type": "unknown_tool", "message": f"No tool named {name}."}})

    def _chat(self, messages: list[dict], tools: list[dict] | None):
        kwargs: dict[str, Any] = {
            "model": self.settings["llm"]["agent_endpoint"],
            "messages": messages,
            "temperature": self.settings["llm"].get("temperature", 0.0),
            "max_tokens": self.settings["llm"].get("max_tokens", 1500),
        }
        if tools:
            kwargs["tools"] = tools
        return self.llm.chat.completions.create(**kwargs).choices[0].message

    def run_turn(self, messages: list[dict]) -> tuple[str, list[dict], TurnState]:
        """Run the tool loop for one user turn.

        Returns the final answer, the Responses API output items (tool calls,
        tool outputs and the answer) and the turn state.
        """
        max_calls = self.settings["agent"]["max_tool_calls"]
        tools = tool_specs(summarise_tables(self.settings["agent"]["schema_context"]))
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, *messages]
        items: list[dict] = []
        state = TurnState()

        while True:
            budget_left = max_calls - state.tool_calls
            msg = self._chat(messages, tools if budget_left > 0 else None)
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls or budget_left <= 0:
                answer = msg.content or ""
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                if state.tool_calls >= max_calls:
                    output = json.dumps({"error": {"type": "budget", "message": "Tool budget used up."}})
                    state.stopped_by_budget = True
                else:
                    state.tool_calls += 1
                    output = self._run_tool(tc.function.name, tc.function.arguments, state)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                items.append(
                    self.create_function_call_item(
                        id=str(uuid.uuid4()),
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )
                items.append(self.create_function_call_output_item(call_id=tc.id, output=output))

            if state.tool_calls >= max_calls:
                state.stopped_by_budget = True
                messages.append(
                    {
                        "role": "user",
                        "content": "The tool budget for this question is used up. Answer now "
                        "from the tool results above, or say you don't know.",
                    }
                )

        items.append(self.create_text_output_item(text=answer, id=str(uuid.uuid4())))
        return answer, items, state

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        messages = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
        answer, items, state = self.run_turn(messages)
        return ResponsesAgentResponse(output=items, custom_outputs=state.as_custom_outputs(answer))

    # Convenience for notebooks and evaluation.
    def answer(self, question: str) -> dict:
        response = self.predict(ResponsesAgentRequest(input=[{"role": "user", "content": question}]))
        text = next(
            (
                c["text"]
                for item in reversed(response.output)
                if item.type == "message"
                for c in item.content
                if c.get("type") == "output_text"
            ),
            "",
        )
        return {"answer": text, **(response.custom_outputs or {})}
