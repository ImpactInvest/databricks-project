"""SQL executors, the LLM client, MLflow setup and agent robustness, with fakes only."""

import json
import threading
from types import SimpleNamespace as NS

import httpx
import pytest

from agent.agent import BakehouseAgent, cited_ids
from agent.config import load_settings
from agent.llm import _DatabricksAuth, make_llm_client
from agent.tools import (
    SparkExecutor,
    SqlExecutionError,
    WarehouseExecutor,
    cast_value,
    make_executor,
)


@pytest.fixture
def settings():
    return load_settings(
        overrides={
            "llm": {"agent_endpoint": "test-llm"},
            "agent": {"schema_context": "TABLE silver_transactions", "max_tool_calls": 3},
            "sql": {"warehouse_id": "wh"},
        }
    )


# --- SparkExecutor ------------------------------------------------------------------


class FakeSpark:
    """Spark Connect-like session: tags, interrupts and a controllable query."""

    def __init__(self, rows=None, error=None, block=False):
        self.rows, self.error, self.block = rows or [], error, block
        self.tags, self.interrupted = [], []
        self._release = threading.Event()

    def addTag(self, tag):  # noqa: N802 - Spark API name
        self.tags.append(tag)

    def removeTag(self, tag):  # noqa: N802
        self.tags.remove(tag)

    def interruptTag(self, tag):  # noqa: N802
        self.interrupted.append(tag)
        self._release.set()

    def sql(self, query):
        spark = self

        class DF:
            columns = ["n", "d"]

            def collect(self):
                if spark.block:
                    spark._release.wait(5)
                    raise RuntimeError("OPERATION_CANCELED")
                if spark.error:
                    raise spark.error
                return spark.rows

        return DF()


def test_spark_executor_returns_json_friendly_rows():
    import datetime as dt
    import decimal

    spark = FakeSpark(rows=[(decimal.Decimal("1.50"), dt.date(2024, 5, 1))])
    cols, rows = SparkExecutor(spark).execute("SELECT 1", timeout_seconds=5)
    assert cols == ["n", "d"] and rows == [[1.5, "2024-05-01"]]
    assert spark.tags == []  # tag removed afterwards


def test_spark_executor_error_is_short_and_structured():
    spark = FakeSpark(error=RuntimeError("[UNRESOLVED_COLUMN] foo\n  long JVM stack trace"))
    with pytest.raises(SqlExecutionError) as e:
        SparkExecutor(spark).execute("SELECT foo", timeout_seconds=5)
    assert e.value.kind == "execution_error" and str(e.value) == "[UNRESOLVED_COLUMN] foo"


def test_spark_executor_timeout_interrupts_the_query():
    spark = FakeSpark(block=True)
    with pytest.raises(SqlExecutionError) as e:
        SparkExecutor(spark).execute("SELECT slow", timeout_seconds=0.05)
    assert e.value.kind == "timeout" and len(spark.interrupted) == 1


def test_make_executor_modes(settings):
    assert isinstance(make_executor(settings, spark=FakeSpark()), SparkExecutor)
    s = load_settings(overrides={"sql": {"executor": "spark"}})
    with pytest.raises(ValueError):
        make_executor(s, spark=None)
    s = load_settings(overrides={"sql": {"executor": "warehouse", "warehouse_id": None}})
    with pytest.raises(ValueError):
        make_executor(s, spark=FakeSpark())  # warehouse mode needs a warehouse id


# --- WarehouseExecutor ------------------------------------------------------------------


def _col(name, type_name):
    return NS(name=name, type_name=NS(value=type_name))


def test_warehouse_executor_fetches_every_result_chunk_and_clamps_the_wait():
    from databricks.sdk.service.sql import StatementState

    calls = {}

    def execute_statement(**kw):
        calls.update(kw)
        return NS(
            statement_id="st1",
            status=NS(state=StatementState.SUCCEEDED, error=None),
            manifest=NS(schema=NS(columns=[_col("n", "INT"), _col("ok", "BOOLEAN")])),
            result=NS(data_array=[["1", "true"]], next_chunk_index=1),
        )

    chunks = {
        1: NS(data_array=[["2", "false"]], next_chunk_index=2),
        2: NS(data_array=[["3", None]], next_chunk_index=None),
    }
    api = NS(execute_statement=execute_statement, get_statement_result_chunk_n=lambda sid, i: chunks[i])
    cols, rows = WarehouseExecutor("wh", workspace_client=NS(statement_execution=api)).execute("SELECT", 300)
    assert cols == ["n", "ok"] and rows == [[1, True], [2, False], [3, None]]
    assert calls["wait_timeout"] == "50s" and calls["warehouse_id"] == "wh"


def test_warehouse_executor_reports_failed_statements():
    from databricks.sdk.service.sql import StatementState

    resp = NS(status=NS(state=StatementState.FAILED, error=NS(message="TABLE_OR_VIEW_NOT_FOUND")))
    w = NS(statement_execution=NS(execute_statement=lambda **kw: resp))
    with pytest.raises(SqlExecutionError) as e:
        WarehouseExecutor("wh", workspace_client=w).execute("SELECT 1", 1)
    assert e.value.kind == "execution_error" and "TABLE_OR_VIEW_NOT_FOUND" in str(e.value)


def test_warehouse_executor_requires_a_warehouse():
    with pytest.raises(ValueError):
        WarehouseExecutor("", workspace_client=NS())


@pytest.mark.parametrize(
    "value,type_name,expected",
    [
        ("7", "SMALLINT", 7),
        ("2.5", "DECIMAL", 2.5),
        ("TRUE", "BOOLEAN", True),
        ("x", "STRING", "x"),
        (None, "INT", None),
    ],
)
def test_cast_value(value, type_name, expected):
    assert cast_value(value, type_name) == expected


# --- LLM client ----------------------------------------------------------------------


class FakeConfig:
    host = "https://example.cloud.databricks.com/"

    def __init__(self):
        self.calls = 0

    def authenticate(self):
        self.calls += 1
        return {"Authorization": f"Bearer token-{self.calls}"}


def test_llm_client_refreshes_the_token_on_every_request():
    w = NS(config=FakeConfig())
    auth = _DatabricksAuth(w)
    first = next(auth.auth_flow(httpx.Request("POST", "https://x/serving-endpoints/chat")))
    second = next(auth.auth_flow(httpx.Request("POST", "https://x/serving-endpoints/chat")))
    assert first.headers["Authorization"] == "Bearer token-1"
    assert second.headers["Authorization"] == "Bearer token-2"

    client = make_llm_client(w)
    assert str(client.base_url).rstrip("/") == "https://example.cloud.databricks.com/serving-endpoints"


def test_llm_client_sends_the_databricks_token_not_the_placeholder():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["Authorization"]
        seen["url"] = str(request.url)
        body = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}
            ],
        }
        return httpx.Response(200, json=body)

    w = NS(config=FakeConfig())
    client = make_llm_client(w)
    client._client = httpx.Client(auth=_DatabricksAuth(w), transport=httpx.MockTransport(handler))
    out = client.chat.completions.create(model="databricks-x", messages=[{"role": "user", "content": "hi"}])
    assert out.choices[0].message.content == "hi"
    assert seen["auth"] == "Bearer token-1"
    assert seen["url"].endswith("/serving-endpoints/chat/completions")


# --- MLflow setup ----------------------------------------------------------------------


def test_setup_mlflow_binds_uc_trace_location(monkeypatch):
    import mlflow

    from agent import tracking

    calls = {}
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: calls.setdefault("tracking", uri))
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda uri: calls.setdefault("registry", uri))
    monkeypatch.setattr(
        mlflow, "set_experiment", lambda **kw: calls.setdefault("experiment", kw) and NS(experiment_id="42")
    )
    monkeypatch.delenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", raising=False)

    s = load_settings(overrides={"sql": {"warehouse_id": "wh9"}})
    assert tracking.setup_mlflow(s) == "42"
    assert calls["tracking"] == "databricks" and calls["registry"] == "databricks-uc"
    location = calls["experiment"]["trace_location"]
    assert (location.catalog_name, location.schema_name) == ("workspace", "genai_agent")
    import os

    assert os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] == "wh9"


def test_setup_mlflow_needs_a_warehouse_for_uc_traces_and_supports_classic_storage(monkeypatch):
    import mlflow

    from agent import tracking

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda uri: None)
    seen = {}
    monkeypatch.setattr(mlflow, "set_experiment", lambda **kw: seen.update(kw) or NS(experiment_id="1"))
    with pytest.raises(ValueError, match="warehouse_id"):
        tracking.setup_mlflow(load_settings())
    tracking.setup_mlflow(load_settings(overrides={"mlflow": {"trace_location": "experiment"}}))
    assert "trace_location" not in seen


# --- agent robustness --------------------------------------------------------------------


class ScriptedLLM:
    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return NS(choices=[NS(message=self.messages.pop(0))])


def _call(name, args, id_="c1"):
    raw = args if isinstance(args, str) else json.dumps(args)
    return NS(content=None, tool_calls=[NS(id=id_, function=NS(name=name, arguments=raw))])


def _text(content):
    return NS(content=content, tool_calls=None)


def _tool_output(llm, request_index=1):
    return json.loads(llm.requests[request_index]["messages"][-1]["content"])


def test_string_franchise_id_from_the_llm_is_converted(settings):
    seen = {}

    def search(query, franchise_id=None, k=5):
        seen.update(franchise_id=franchise_id, k=k)
        return []

    llm = ScriptedLLM(
        [_call("search_reviews", {"query": "x", "franchise_id": "3001", "k": 50}), _text("None.")]
    )
    BakehouseAgent(settings, llm_client=llm, text2sql=None, review_search=search).answer("q")
    assert seen == {"franchise_id": 3001, "k": 10}  # k is capped at 10
    assert _tool_output(llm)["note"] == "No matching reviews."


def test_non_numeric_franchise_id_becomes_a_tool_error(settings):
    llm = ScriptedLLM([_call("search_reviews", {"query": "x", "franchise_id": "Shop A"}), _text("Sorry.")])
    agent = BakehouseAgent(settings, llm_client=llm, review_search=lambda *a, **k: [])
    assert agent.answer("q")["answer"] == "Sorry."
    assert _tool_output(llm)["error"]["type"] == "bad_arguments"


def test_failing_tool_does_not_end_the_conversation(settings):
    def broken_search(*args, **kwargs):
        raise ConnectionError("AI Search endpoint is scaling up")

    llm = ScriptedLLM([_call("search_reviews", {"query": "x"}), _text("Search is unavailable right now.")])
    out = BakehouseAgent(settings, llm_client=llm, review_search=broken_search).answer("q")
    assert out["answer"] == "Search is unavailable right now."
    err = _tool_output(llm)["error"]
    assert err["type"] == "tool_error" and "scaling up" in err["message"]


@pytest.mark.parametrize("raw", ["[1, 2]", '"just a string"', "null"])
def test_non_object_arguments_are_rejected(settings, raw):
    llm = ScriptedLLM([_call("search_reviews", raw), _text("Sorry.")])
    BakehouseAgent(settings, llm_client=llm, review_search=lambda *a, **k: []).answer("q")
    assert _tool_output(llm)["error"]["type"] == "bad_arguments"


def test_unknown_tool_is_reported_to_the_llm(settings):
    llm = ScriptedLLM([_call("drop_table", {}), _text("I can't do that.")])
    BakehouseAgent(settings, llm_client=llm).answer("q")
    assert _tool_output(llm)["error"]["type"] == "unknown_tool"


def test_parallel_tool_calls_beyond_the_budget_are_refused(settings):
    many = NS(
        content=None,
        tool_calls=[
            NS(id=f"c{i}", function=NS(name="search_reviews", arguments='{"query": "x"}')) for i in range(5)
        ],
    )
    calls = []

    def search(*a, **k):
        calls.append(1)
        return []

    llm = ScriptedLLM([many, _text("Done.")])
    out = BakehouseAgent(settings, llm_client=llm, review_search=search).answer("q")
    assert len(calls) == 3 and out["tool_calls"] == 3 and out["stopped_by_budget"] is True
    tool_msgs = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 5 and "budget" in tool_msgs[-1]["content"]  # every call gets an answer
    assert "tools" not in llm.requests[1]


def test_failed_sql_is_listed_but_not_shown_as_a_result(settings):
    failed = {"error": {"type": "guard_rejected", "message": "no", "hint": ""}, "sql": "DROP TABLE x"}
    llm = ScriptedLLM([_call("query_sales_data", {"question": "drop"}), _text("I can't.")])
    out = BakehouseAgent(settings, llm_client=llm, text2sql=lambda q: failed).answer("q")
    assert out["sql_queries"] == ["DROP TABLE x"] and out["sql_results"] == []


@pytest.mark.parametrize(
    "answer,ids",
    [
        ("Queues are long [review 5].", {5}),
        ("See [reviews 5, 7 and 9] and review #11.", {5, 7, 9, 11}),
        ("Review_12 says so; REVIEW: 13 agrees.", {12, 13}),
        ("Reviews from 2024 are positive.", set()),
        ("", set()),
    ],
)
def test_citation_parsing(answer, ids):
    assert cited_ids(answer) == ids
