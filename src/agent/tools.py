"""The agent's tools.

``query_sales_data(question)``
    Text2SQL over the silver tables. An LLM writes SQL from the question and a
    schema summary, ``sql_guard`` checks and rewrites it, and ``run_sql``
    executes it. If the guard or the database rejects the query, the error is
    fed back to the LLM for exactly one retry. The tool returns
    ``{columns, rows, row_count, sql}`` or a structured error.

``search_reviews(query, franchise_id=None, k=5)``
    Semantic search over chunked customer reviews in the AI Search index,
    optionally filtered to one franchise. Traced as a RETRIEVER span so the
    MLflow groundedness judge can see exactly what the answer was based on.

Every tool is traced with an MLflow span so a trace shows the question, the
generated SQL, the rows returned and the retrieved chunks.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import threading
import uuid
from typing import Any, Protocol

import mlflow
from mlflow.entities import SpanType

from agent.config import Settings
from agent.prompts import extract_sql, render_text2sql_messages, text2sql_prompt_parts
from agent.sql_guard import check_sql


class SqlExecutionError(RuntimeError):
    def __init__(self, message: str, kind: str = "execution_error"):
        super().__init__(message)
        self.kind = kind


def to_jsonable(value: Any) -> Any:
    """Convert database values into JSON-friendly Python values."""
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


# ---------------------------------------------------------------------------
# SQL executors
# ---------------------------------------------------------------------------


class SqlExecutor(Protocol):
    def execute(self, sql: str, timeout_seconds: int) -> tuple[list[str], list[list[Any]]]: ...


class SparkExecutor:
    """Runs SQL with the notebook's Spark session (serverless Spark Connect).

    The timeout uses Spark Connect operation tags: a timer interrupts the tagged
    query if it runs too long.
    """

    def __init__(self, spark):
        self.spark = spark

    def execute(self, sql: str, timeout_seconds: int) -> tuple[list[str], list[list[Any]]]:
        tag = f"agent-sql-{uuid.uuid4().hex[:8]}"
        timed_out = threading.Event()
        timer = None
        can_interrupt = hasattr(self.spark, "addTag") and hasattr(self.spark, "interruptTag")
        if can_interrupt:
            self.spark.addTag(tag)

            def _interrupt():
                timed_out.set()
                self.spark.interruptTag(tag)

            timer = threading.Timer(timeout_seconds, _interrupt)
            timer.start()
        try:
            df = self.spark.sql(sql)
            rows = df.collect()
            columns = list(df.columns)
        except Exception as e:  # noqa: BLE001 - surface any Spark error to the LLM
            if timed_out.is_set():
                raise SqlExecutionError(f"Query exceeded the {timeout_seconds}s timeout.", "timeout") from e
            raise SqlExecutionError(_short_error(e)) from e
        finally:
            if timer:
                timer.cancel()
            if can_interrupt:
                self.spark.removeTag(tag)
        return columns, [[to_jsonable(v) for v in row] for row in rows]


class WarehouseExecutor:
    """Runs SQL on a SQL warehouse through the Statement Execution API.

    Used inside Model Serving, where there is no Spark session. Authentication
    comes from the resources declared when the agent was logged.
    """

    _INT_TYPES = {"BYTE", "SHORT", "INT", "LONG", "INTEGER", "BIGINT", "SMALLINT", "TINYINT"}
    _FLOAT_TYPES = {"FLOAT", "DOUBLE", "DECIMAL"}

    def __init__(self, warehouse_id: str, workspace_client=None):
        if not warehouse_id:
            raise ValueError("sql.warehouse_id is not set; run notebooks/00_setup.py.")
        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()
        self.w = workspace_client
        self.warehouse_id = warehouse_id

    def execute(self, sql: str, timeout_seconds: int) -> tuple[list[str], list[list[Any]]]:
        from databricks.sdk.service.sql import (
            Disposition,
            ExecuteStatementRequestOnWaitTimeout,
            Format,
            StatementState,
        )

        # The API accepts wait timeouts between 5 and 50 seconds.
        wait = max(5, min(50, int(timeout_seconds)))
        resp = self.w.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=self.warehouse_id,
            wait_timeout=f"{wait}s",
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CANCEL,
            disposition=Disposition.INLINE,
            format=Format.JSON_ARRAY,
        )
        state = resp.status.state if resp.status else None
        if state in (StatementState.CANCELED, StatementState.PENDING, StatementState.RUNNING):
            raise SqlExecutionError(f"Query exceeded the {wait}s timeout.", "timeout")
        if state not in (StatementState.SUCCEEDED, StatementState.CLOSED):
            message = resp.status.error.message if resp.status and resp.status.error else state
            raise SqlExecutionError(str(message)[:500])

        columns_meta = resp.manifest.schema.columns if resp.manifest and resp.manifest.schema else []
        columns = [c.name for c in columns_meta]
        types = [str(c.type_name.value if c.type_name else "STRING").upper() for c in columns_meta]
        data = (resp.result.data_array if resp.result else None) or []
        rows = [[self._cast(v, t) for v, t in zip(row, types)] for row in data]
        return columns, rows

    def _cast(self, value: str | None, type_name: str) -> Any:
        if value is None:
            return None
        if type_name in self._INT_TYPES:
            return int(value)
        if type_name in self._FLOAT_TYPES:
            return float(value)
        if type_name == "BOOLEAN":
            return value.lower() == "true"
        return value


def _short_error(e: Exception) -> str:
    text = str(e).strip().splitlines()
    return (text[0] if text else type(e).__name__)[:500]


def make_executor(settings: Settings, spark=None) -> SqlExecutor:
    mode = settings["sql"].get("executor", "auto")
    if mode == "spark" or (mode == "auto" and spark is not None):
        if spark is None:
            raise ValueError("sql.executor is 'spark' but no Spark session was provided.")
        return SparkExecutor(spark)
    return WarehouseExecutor(settings["sql"]["warehouse_id"])


# ---------------------------------------------------------------------------
# run_sql: guarded execution
# ---------------------------------------------------------------------------


def run_sql(query: str, settings: Settings, executor: SqlExecutor) -> dict:
    """Check ``query`` with the guard and execute it.

    Returns ``{"columns", "rows", "row_count", "sql"}`` on success, or
    ``{"error": {"type", "message", "hint"}, "sql": query}`` on failure. The
    full result (up to the LIMIT) is recorded on the ``run_sql`` span, which
    the evaluation scorers read; callers decide how many rows the LLM sees.
    """
    with mlflow.start_span(name="run_sql", span_type=SpanType.TOOL) as span:
        span.set_inputs({"query": query})
        result = _guard_and_execute(query, settings, executor)
        span.set_outputs(result)
        return result


def _guard_and_execute(query: str, settings: Settings, executor: SqlExecutor) -> dict:
    sql_cfg = settings["sql"]
    guard = check_sql(
        query,
        allowed_tables=settings.silver_tables,
        catalog=settings.catalog,
        schema=settings.schema,
        default_limit=sql_cfg["default_limit"],
        max_limit=sql_cfg["max_limit"],
    )
    if not guard.ok:
        return {"error": guard.to_error_dict(), "sql": query}
    try:
        columns, rows = executor.execute(guard.sql, sql_cfg["timeout_seconds"])
    except SqlExecutionError as e:
        return {
            "error": {"type": e.kind, "message": str(e), "hint": "Fix the SQL and try again."},
            "sql": guard.sql,
        }
    return {"columns": columns, "rows": rows, "row_count": len(rows), "sql": guard.sql}


# ---------------------------------------------------------------------------
# Text2SQL
# ---------------------------------------------------------------------------


class Text2Sql:
    """Question -> SQL with one self-correction attempt."""

    def __init__(self, settings: Settings, llm_client, executor: SqlExecutor, schema_context: str):
        self.settings = settings
        self.llm = llm_client
        self.executor = executor
        self.schema_context = schema_context
        variant = settings["agent"].get("prompt_variant", "baseline")
        self.instructions, self.demos = text2sql_prompt_parts(variant)

    @mlflow.trace(name="generate_sql", span_type=SpanType.LLM)
    def generate(self, question: str, previous_attempt: dict | None = None) -> str:
        messages = render_text2sql_messages(
            question, self.schema_context, self.instructions, self.demos, previous_attempt
        )
        resp = self.llm.chat.completions.create(
            model=self.settings["llm"]["agent_endpoint"],
            messages=messages,
            temperature=0.0,
            max_tokens=800,
        )
        return extract_sql(resp.choices[0].message.content or "")

    @mlflow.trace(name="query_sales_data", span_type=SpanType.TOOL)
    def __call__(self, question: str) -> dict:
        sql = self.generate(question)
        result = run_sql(sql, self.settings, self.executor)
        if "error" in result:
            retry_sql = self.generate(
                question,
                previous_attempt={"sql": sql, "error": result["error"]["message"]},
            )
            result = run_sql(retry_sql, self.settings, self.executor)
            result["retried"] = True
        return result


def result_for_llm(result: dict, max_rows: int) -> str:
    """Compact JSON of a SQL tool result, truncated to ``max_rows`` rows."""
    if "error" in result:
        return json.dumps(result)
    shown = result["rows"][:max_rows]
    payload = {
        "sql": result["sql"],
        "columns": result["columns"],
        "rows": shown,
        "row_count": result["row_count"],
    }
    if result["row_count"] > max_rows:
        payload["note"] = f"Only the first {max_rows} rows are shown."
    return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# AI Search retrieval
# ---------------------------------------------------------------------------

REVIEW_COLUMNS = ["chunk_id", "review_id", "franchise_id", "review_date", "chunk_text"]


class ReviewSearch:
    """Wrapper around the AI Search Delta Sync index of review chunks."""

    def __init__(self, settings: Settings, index=None):
        self.settings = settings
        if index is None:
            from databricks.ai_search.client import AISearchClient

            client = AISearchClient(disable_notice=True)
            index = client.get_index(
                endpoint_name=settings["ai_search"]["endpoint_name"],
                index_name=settings.index_name,
            )
        self.index = index

    @mlflow.trace(name="search_reviews", span_type=SpanType.RETRIEVER)
    def retrieve(self, query: str, franchise_id: int | None = None, k: int = 5) -> list[dict]:
        """Return documents in the shape MLflow expects for RETRIEVER spans."""
        filters = {"franchise_id": int(franchise_id)} if franchise_id is not None else None
        resp = self.index.similarity_search(
            query_text=query,
            columns=REVIEW_COLUMNS,
            filters=filters,
            num_results=int(k),
            disable_notice=True,
        )
        return parse_search_response(resp)

    def __call__(self, query: str, franchise_id: int | None = None, k: int | None = None) -> list[dict]:
        k = k or self.settings["ai_search"]["default_k"]
        docs = self.retrieve(query, franchise_id=franchise_id, k=k)
        return [
            {
                "chunk_id": d["metadata"]["chunk_id"],
                "review_id": d["metadata"]["review_id"],
                "franchise_id": d["metadata"]["franchise_id"],
                "review_date": d["metadata"]["review_date"],
                "text": d["page_content"],
                "score": d["metadata"]["score"],
            }
            for d in docs
        ]


def parse_search_response(resp: dict) -> list[dict]:
    """Turn an AI Search response into ``[{page_content, metadata, id}]``.

    The response has ``manifest.columns`` (the requested columns followed by a
    ``score`` column) and ``result.data_array`` (one list per hit).
    """
    columns = [c["name"] for c in resp.get("manifest", {}).get("columns", [])]
    docs = []
    for row in resp.get("result", {}).get("data_array", []) or []:
        rec = dict(zip(columns, row))
        review_id = rec.get("review_id")
        review_id = int(review_id) if review_id is not None else None
        franchise_id = rec.get("franchise_id")
        franchise_id = int(franchise_id) if franchise_id is not None else None
        docs.append(
            {
                "id": str(rec.get("chunk_id")),
                "page_content": rec.get("chunk_text", ""),
                "metadata": {
                    "doc_uri": f"review://{review_id}",
                    "chunk_id": str(rec.get("chunk_id")),
                    "review_id": review_id,
                    "franchise_id": franchise_id,
                    "review_date": to_jsonable(rec.get("review_date")),
                    "score": rec.get("score"),
                },
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Tool specifications (OpenAI function-calling format)
# ---------------------------------------------------------------------------


def tool_specs(schema_summary: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "query_sales_data",
                "description": (
                    "Answer a quantitative question about bakery sales, franchises, customers "
                    "or suppliers. Pass the question in plain English; the tool writes and runs "
                    "a read-only SQL query and returns the SQL, columns and rows. "
                    f"Available data:\n{schema_summary}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "A self-contained question, including any franchise, "
                            "product or date the user mentioned.",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_reviews",
                "description": (
                    "Semantic search over customer reviews of the bakery franchises. Returns "
                    "review chunks with review_id, franchise_id, date and text. Use franchise_id "
                    "to restrict results to one franchise."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look for in reviews."},
                        "franchise_id": {
                            "type": "integer",
                            "description": "Optional franchise_id filter.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of chunks to return (1 to 10).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Schema context (built once from information_schema, stored in config)
# ---------------------------------------------------------------------------


def schema_context_sql(settings: Settings) -> str:
    tables = ", ".join(f"'{t}'" for t in settings.silver_tables)
    return f"""
        SELECT c.table_name, t.comment AS table_comment, c.column_name, c.data_type,
               c.comment AS column_comment, c.ordinal_position
        FROM {settings.catalog}.information_schema.columns c
        JOIN {settings.catalog}.information_schema.tables t
          ON c.table_catalog = t.table_catalog AND c.table_schema = t.table_schema
         AND c.table_name = t.table_name
        WHERE c.table_schema = '{settings.schema}' AND c.table_name IN ({tables})
        ORDER BY c.table_name, c.ordinal_position
    """


def format_schema_context(rows: list[dict]) -> str:
    """Render information_schema rows as a compact text block for prompts."""
    lines: list[str] = []
    current = None
    for r in rows:
        if r["table_name"] != current:
            current = r["table_name"]
            comment = f" -- {r['table_comment']}" if r.get("table_comment") else ""
            lines.append(f"TABLE {current}{comment}")
        col_comment = f" -- {r['column_comment']}" if r.get("column_comment") else ""
        lines.append(f"  {r['column_name']} {r['data_type']}{col_comment}")
    return "\n".join(lines)


def summarise_tables(schema_context: str) -> str:
    """Only the TABLE lines of a schema context, for the agent's tool description."""
    return "\n".join(line for line in schema_context.splitlines() if line.startswith("TABLE "))
