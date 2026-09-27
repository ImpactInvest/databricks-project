"""Safety checks for SQL produced by the Text2SQL step.

The guard is deliberately strict. It accepts exactly one read-only query over
an allowlist of tables and rewrites it so it is safe to execute:

* the SQL must parse (Databricks dialect) into a single statement;
* the statement must be a SELECT (optionally with CTEs or UNION);
* no DDL, DML, commands, ``SELECT ... INTO`` or table-valued functions;
* no functions from a small denylist (for example ``reflect``, ``read_files``);
* every table must be on the allowlist, and is rewritten to its fully
  qualified Unity Catalog name so the query cannot escape the project schema;
* a LIMIT is added when missing and capped when too large.

Execution-time protections (the timeout and read-only credentials) live in
``tools.py`` and in Unity Catalog grants; this module only reasons about text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

# sqlglot logs a warning when it falls back to parsing something as a Command;
# we reject Commands anyway, so keep the logs quiet.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.Command,
    exp.Set,
    exp.Use,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Copy,
    exp.Describe,
    exp.Show,
    exp.Cache,
    exp.Uncache,
    exp.Refresh,
    exp.Pragma,
    exp.LoadData,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Into,
    exp.Analyze,
    exp.Kill,
    exp.Execute,
)

# Functions that can reach outside the allowlisted tables, run code, call
# models or stall the warehouse.
DENIED_FUNCTIONS = {
    "reflect",
    "java_method",
    "read_files",
    "read_kafka",
    "read_state_metadata",
    "read_statestore",
    "http_request",
    "ai_query",
    "ai_gen",
    "ai_extract",
    "ai_classify",
    "ai_summarize",
    "vector_search",
    "sleep",
    "current_user",
    "session_user",
    "secret",
    "try_secret",
    "input_file_name",
}

ALLOWED_ROOTS: tuple[type[exp.Expression], ...] = (exp.Select, exp.SetOperation, exp.Subquery)


class SqlGuardError(ValueError):
    """Raised when a query is rejected. ``hint`` is shown to the LLM."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


@dataclass
class GuardResult:
    ok: bool
    sql: str | None = None
    error: str | None = None
    hint: str | None = None
    tables: list[str] = field(default_factory=list)

    def to_error_dict(self) -> dict:
        return {"type": "guard_rejected", "message": self.error, "hint": self.hint}


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return (node.sql_name() or "").lower()


def check_sql(
    sql: str,
    allowed_tables: list[str],
    catalog: str,
    schema: str,
    default_limit: int = 200,
    max_limit: int = 1000,
) -> GuardResult:
    """Validate ``sql`` and return a safe, rewritten version or an error."""
    try:
        return GuardResult(
            ok=True,
            **_check(sql, allowed_tables, catalog, schema, default_limit, max_limit),
        )
    except SqlGuardError as e:
        return GuardResult(ok=False, error=str(e), hint=e.hint)


def _check(
    sql: str,
    allowed_tables: list[str],
    catalog: str,
    schema: str,
    default_limit: int,
    max_limit: int,
) -> dict:
    if not sql or not sql.strip():
        raise SqlGuardError("Empty query.", "Write a single SELECT statement.")

    try:
        statements = [s for s in sqlglot.parse(sql, dialect="databricks") if s is not None]
    except sqlglot.errors.SqlglotError as e:
        raise SqlGuardError(f"Could not parse SQL: {e}", "Check the SQL syntax.") from e

    if len(statements) != 1:
        raise SqlGuardError(
            f"Expected exactly one statement, got {len(statements)}.",
            "Send a single SELECT statement without semicolon-separated parts.",
        )
    tree = statements[0]

    if not isinstance(tree, ALLOWED_ROOTS):
        raise SqlGuardError(
            f"Only SELECT queries are allowed (got {type(tree).__name__}).",
            "This tool is read-only. Rewrite the request as a SELECT, or decline it.",
        )

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SqlGuardError(
                f"Statement type {type(node).__name__} is not allowed.",
                "This tool is read-only. Only SELECT queries are permitted.",
            )
        if isinstance(node, exp.Func) and _function_name(node) in DENIED_FUNCTIONS:
            raise SqlGuardError(
                f"Function {_function_name(node)}() is not allowed.",
                "Use standard SQL functions over the allowed tables only.",
            )

    allowed = {t.lower() for t in allowed_tables}
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    used_tables: list[str] = []

    for table in list(tree.find_all(exp.Table)):
        if not isinstance(table.this, exp.Identifier):
            raise SqlGuardError(
                "Table-valued functions are not allowed in FROM.",
                f"Query only these tables: {', '.join(sorted(allowed))}.",
            )
        name = table.name.lower()
        db = (table.db or "").lower()
        cat = (table.catalog or "").lower()

        if not db and not cat and name in cte_names:
            continue  # reference to a CTE defined in this query
        if name not in allowed:
            raise SqlGuardError(
                f"Table '{table.sql(dialect='databricks')}' is not on the allowlist.",
                f"Query only these tables: {', '.join(sorted(allowed))}.",
            )
        if db and db != schema.lower():
            raise SqlGuardError(f"Schema '{db}' is not allowed.", f"Use tables in {catalog}.{schema} only.")
        if cat and cat != catalog.lower():
            raise SqlGuardError(f"Catalog '{cat}' is not allowed.", f"Use tables in {catalog}.{schema} only.")

        table.set("catalog", exp.to_identifier(catalog))
        table.set("db", exp.to_identifier(schema))
        table.set("this", exp.to_identifier(name))
        used_tables.append(name)

    if not used_tables:
        raise SqlGuardError(
            "The query does not read any allowed table.",
            f"Query at least one of: {', '.join(sorted(allowed))}.",
        )

    tree = _enforce_limit(tree, default_limit, max_limit)
    return {"sql": tree.sql(dialect="databricks", comments=False), "tables": sorted(set(used_tables))}


def _enforce_limit(tree: exp.Expression, default_limit: int, max_limit: int) -> exp.Expression:
    limit = tree.args.get("limit")
    if limit is None:
        return tree.limit(default_limit, copy=False)
    value = limit.expression
    if isinstance(value, exp.Literal) and value.is_int:
        if int(value.this) > max_limit:
            limit.set("expression", exp.Literal.number(max_limit))
        return tree
    # Non-literal limits (for example expressions) are replaced with the default.
    limit.set("expression", exp.Literal.number(default_limit))
    return tree
