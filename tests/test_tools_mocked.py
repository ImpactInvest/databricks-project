"""Tool and agent behaviour with every Databricks dependency mocked."""

import json
from types import SimpleNamespace as NS

import pytest

from agent.agent import BakehouseAgent
from agent.config import load_settings
from agent.prompts import extract_sql, render_text2sql_messages
from agent.tools import (
    ReviewSearch,
    SqlExecutionError,
    Text2Sql,
    WarehouseExecutor,
    parse_search_response,
    result_for_llm,
    run_sql,
    tool_specs,
)

SCHEMA = "TABLE silver_transactions -- sales\n  revenue DECIMAL(12,2) -- total"


@pytest.fixture
def settings():
    return load_settings(
        overrides={
            "llm": {"agent_endpoint": "test-llm"},
            "agent": {"schema_context": SCHEMA, "max_tool_calls": 3},
            "sql": {"warehouse_id": "wh"},
        }
    )


class FakeExecutor:
    def __init__(self, result=None, error=None):
        self.result = result or (["n"], [[1]])
        self.error = error
        self.calls = []

    def execute(self, sql, timeout_seconds):
        self.calls.append(sql)
        if self.error:
            raise self.error
        return self.result


class ScriptedLLM:
    """Returns pre-baked chat messages in order and records each request."""

    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return NS(choices=[NS(message=self.messages.pop(0))])


def text(content):
    return NS(content=content, tool_calls=None)


def call(name, args, id_="c1"):
    return NS(content=None, tool_calls=[NS(id=id_, function=NS(name=name, arguments=json.dumps(args)))])


# --- run_sql ---------------------------------------------------------------


def test_run_sql_success_shape(settings):
    ex = FakeExecutor((["product", "units"], [["A", 3], ["B", 2]]))
    out = run_sql("SELECT product, units FROM silver_transactions", settings, ex)
    assert set(out) == {"columns", "rows", "row_count", "sql"}
    assert out["row_count"] == 2
    assert ex.calls[0].endswith("LIMIT 200")


def test_run_sql_guard_rejection_does_not_execute(settings):
    ex = FakeExecutor()
    out = run_sql("DROP TABLE silver_transactions", settings, ex)
    assert out["error"]["type"] == "guard_rejected"
    assert ex.calls == []


def test_run_sql_execution_error_is_structured(settings):
    ex = FakeExecutor(error=SqlExecutionError("UNRESOLVED_COLUMN foo"))
    out = run_sql("SELECT foo FROM silver_transactions", settings, ex)
    assert out["error"]["type"] == "execution_error"
    assert "UNRESOLVED_COLUMN" in out["error"]["message"]


def test_result_for_llm_truncates():
    res = {"columns": ["a"], "rows": [[i] for i in range(10)], "row_count": 10, "sql": "x"}
    payload = json.loads(result_for_llm(res, max_rows=3))
    assert len(payload["rows"]) == 3 and "note" in payload


# --- Text2SQL ----------------------------------------------------------------


def test_text2sql_retries_once_with_error_feedback(settings):
    llm = ScriptedLLM(
        [
            text("SELECT nope FROM silver_reviews"),
            text("```sql\nSELECT 1 AS n FROM silver_transactions;\n```"),
        ]
    )
    ex = FakeExecutor()
    out = Text2Sql(settings, llm, ex, SCHEMA)("how many?")
    assert out["retried"] is True and "error" not in out
    retry_msgs = llm.requests[1]["messages"]
    assert "allowlist" in retry_msgs[-1]["content"]


def test_text2sql_gives_up_after_one_retry(settings):
    llm = ScriptedLLM([text("DROP TABLE x"), text("DELETE FROM silver_transactions")])
    out = Text2Sql(settings, llm, FakeExecutor(), SCHEMA)("delete everything")
    assert out["error"]["type"] == "guard_rejected"
    assert len(llm.requests) == 2


def test_prompt_rendering_includes_demos_and_schema():
    msgs = render_text2sql_messages("q?", SCHEMA, "INSTR", [{"question": "d?", "sql": "SELECT 1"}])
    assert msgs[0]["role"] == "system" and "INSTR" in msgs[0]["content"] and "revenue" in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]


def test_extract_sql_strips_fences():
    assert extract_sql("Here:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


# --- AI Search -----------------------------------------------------------------


SEARCH_RESPONSE = {
    "manifest": {
        "columns": [
            {"name": c}
            for c in ["chunk_id", "review_id", "franchise_id", "review_date", "chunk_text", "score"]
        ]
    },
    "result": {"data_array": [["12-0", 12.0, 3000.0, "2024-05-02", "Lovely cookies.", 0.81]]},
}


def test_parse_search_response_retriever_shape():
    docs = parse_search_response(SEARCH_RESPONSE)
    assert docs[0]["page_content"] == "Lovely cookies."
    assert docs[0]["metadata"]["review_id"] == 12
    assert docs[0]["metadata"]["doc_uri"] == "review://12"


def test_review_search_passes_filter_and_returns_tool_shape(settings):
    captured = {}

    class FakeIndex:
        def similarity_search(self, **kw):
            captured.update(kw)
            return SEARCH_RESPONSE

    hits = ReviewSearch(settings, index=FakeIndex())("cookies", franchise_id=3000, k=2)
    assert captured["filters"] == {"franchise_id": 3000} and captured["num_results"] == 2
    assert set(hits[0]) == {"chunk_id", "review_id", "franchise_id", "review_date", "text", "score"}


def test_empty_search_response():
    assert parse_search_response({"manifest": {"columns": []}, "result": {}}) == []


# --- Warehouse executor ------------------------------------------------------------


def test_warehouse_executor_casts_types():
    from databricks.sdk.service.sql import StatementState

    col = lambda n, t: NS(name=n, type_name=NS(value=t))  # noqa: E731
    resp = NS(
        status=NS(state=StatementState.SUCCEEDED, error=None),
        manifest=NS(schema=NS(columns=[col("n", "LONG"), col("r", "DECIMAL"), col("p", "STRING")])),
        result=NS(data_array=[["3", "1.50", "A"]], next_chunk_index=None),
    )
    w = NS(statement_execution=NS(execute_statement=lambda **kw: resp))
    cols, rows = WarehouseExecutor("wh", workspace_client=w).execute("SELECT 1", 30)
    assert cols == ["n", "r", "p"] and rows == [[3, 1.5, "A"]]


def test_warehouse_executor_timeout():
    from databricks.sdk.service.sql import StatementState

    resp = NS(status=NS(state=StatementState.CANCELED, error=None))
    w = NS(statement_execution=NS(execute_statement=lambda **kw: resp))
    with pytest.raises(SqlExecutionError) as e:
        WarehouseExecutor("wh", workspace_client=w).execute("SELECT 1", 30)
    assert e.value.kind == "timeout"


# --- Agent loop ------------------------------------------------------------------


def _agent(settings, llm_messages, sql_result=None, hits=None):
    llm = ScriptedLLM(llm_messages)
    default_sql = {
        "columns": ["franchise_id", "revenue"],
        "rows": [[7, 10.0]],
        "row_count": 1,
        "sql": "SELECT 1",
    }
    default_hits = [
        {
            "chunk_id": "5-0",
            "review_id": 5,
            "franchise_id": 7,
            "review_date": "2024-05-01",
            "text": "long queue",
            "score": 0.9,
        }
    ]

    def t2s(question):
        return sql_result or default_sql

    def rs(query, franchise_id=None, k=5):
        return hits if hits is not None else default_hits

    return BakehouseAgent(settings, llm_client=llm, text2sql=t2s, review_search=rs), llm


def test_agent_hybrid_flow_and_citations(settings):
    agent, _ = _agent(
        settings,
        [
            call("query_sales_data", {"question": "top franchise"}),
            call("search_reviews", {"query": "complaints", "franchise_id": 7}, id_="c2"),
            text("Franchise 7 earned 10. Customers mention long queues [review 5] [review 99]."),
        ],
    )
    out = agent.answer("Top franchise and complaints?")
    assert out["tool_calls"] == 2
    assert out["sql_queries"] == ["SELECT 1"]
    assert out["cited_review_ids"] == [5]  # 99 was never retrieved, so it is not a citation


def test_agent_answers_without_tools(settings):
    agent, llm = _agent(settings, [text("I can't delete data; I am read-only.")])
    out = agent.answer("Drop the table")
    assert out["tool_calls"] == 0 and "read-only" in out["answer"]
    assert llm.requests[0]["messages"][0]["role"] == "system"


def test_agent_stops_at_tool_budget(settings):
    loop = [call("search_reviews", {"query": "x"}, id_=f"c{i}") for i in range(3)]
    agent, llm = _agent(settings, [*loop, text("Stopping here.")])
    out = agent.answer("loop forever")
    assert out["tool_calls"] == 3 and out["stopped_by_budget"] is True
    assert "tools" not in llm.requests[-1]  # final call is made with tools disabled
    assert out["answer"] == "Stopping here."


def test_agent_handles_bad_tool_arguments(settings):
    bad = NS(
        content=None, tool_calls=[NS(id="c1", function=NS(name="search_reviews", arguments="{not json"))]
    )
    agent, llm = _agent(settings, [bad, text("Sorry.")])
    agent.answer("x")
    tool_msg = llm.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "bad_arguments" in tool_msg["content"]


def test_tool_specs_are_valid_function_definitions():
    specs = tool_specs("TABLE silver_transactions")
    names = {s["function"]["name"] for s in specs}
    assert names == {"query_sales_data", "search_reviews"}
    for s in specs:
        assert s["function"]["parameters"]["type"] == "object"


# --- Genie client ----------------------------------------------------------------


def test_ask_genie_collects_sql_rows_and_text():
    from databricks.sdk.service.dashboards import MessageStatus

    from evals.genie import ask_genie

    col = lambda n, t: NS(name=n, type_name=NS(value=t))  # noqa: E731
    statement = NS(
        manifest=NS(schema=NS(columns=[col("franchise_name", "STRING"), col("revenue", "DOUBLE")])),
        result=NS(data_array=[["Shop A", "10.5"]]),
    )
    message = NS(
        conversation_id="c",
        message_id="m",
        status=MessageStatus.COMPLETED,
        error=None,
        attachments=[
            NS(
                attachment_id="a1",
                text=None,
                query=NS(query="SELECT ...", description="Top franchise by revenue"),
            ),
        ],
    )
    genie = NS(
        start_conversation_and_wait=lambda space_id, content, timeout: message,
        get_message_attachment_query_result=lambda s, c, m, a: NS(statement_response=statement),
    )
    out = ask_genie(NS(genie=genie), "space", "Which franchise earns most?")
    assert out["status"] == "COMPLETED" and out["sql"] == "SELECT ..."
    assert out["rows"] == [["Shop A", 10.5]]
    assert "Top franchise by revenue" in out["answer"] and "Shop A" in out["answer"]
