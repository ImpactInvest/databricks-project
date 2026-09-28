"""Ask a Genie Agent (formerly Genie Space) questions through the Conversation API.

Used only by the benchmark notebook. The API and SDK still use the old
``genie`` / ``space`` names.
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

from agent.tools import cast_value


def parse_statement_response(statement_response) -> tuple[list[str], list[list[Any]]]:
    """Columns and rows from a SQL ``StatementResponse`` (values arrive as strings)."""
    if statement_response is None:
        return [], []
    manifest = statement_response.manifest
    columns = [c.name for c in manifest.schema.columns] if manifest and manifest.schema else []
    types = (
        [str(c.type_name.value if c.type_name else "STRING").upper() for c in manifest.schema.columns]
        if columns
        else []
    )
    data = (statement_response.result.data_array if statement_response.result else None) or []
    return columns, [[cast_value(v, t) for v, t in zip(row, types)] for row in data]


def ask_genie(w, space_id: str, question: str, retries: int = 3) -> dict:
    """Start a new conversation with one question and collect Genie's answer.

    Returns the text answer, the SQL Genie generated, its result rows and the
    wall-clock latency. Rate-limit errors are retried with a back-off.
    """
    for attempt in range(retries):
        start = time.time()
        try:
            message = w.genie.start_conversation_and_wait(space_id, question, timeout=timedelta(minutes=5))
            latency = time.time() - start
            break
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate" in str(e).lower():
                time.sleep(30 * (attempt + 1))
                continue
            return {
                "status": "ERROR",
                "error": str(e)[:500],
                "latency_seconds": time.time() - start,
                "answer": "",
                "sql": None,
                "columns": [],
                "rows": [],
            }
    else:
        return {
            "status": "RATE_LIMITED",
            "error": "Gave up after retries",
            "latency_seconds": None,
            "answer": "",
            "sql": None,
            "columns": [],
            "rows": [],
        }

    text_parts, sql, columns, rows = [], None, [], []
    for att in message.attachments or []:
        if att.text and att.text.content:
            text_parts.append(att.text.content)
        if att.query:
            sql = att.query.query
            if att.query.description:
                text_parts.append(att.query.description)
            try:
                result = w.genie.get_message_attachment_query_result(
                    space_id, message.conversation_id, message.message_id, att.attachment_id
                )
                columns, rows = parse_statement_response(result.statement_response)
            except Exception as e:  # noqa: BLE001
                text_parts.append(f"(could not fetch query result: {e})")

    # Genie shows its result table to the user, so the judge sees a preview too.
    if rows:
        preview = "\n".join(", ".join(str(v) for v in r) for r in rows[:20])
        text_parts.append(f"Result ({', '.join(columns)}):\n{preview}")

    return {
        "status": message.status.value if message.status else None,
        "error": message.error.error if message.error else None,
        "latency_seconds": round(latency, 2),
        "answer": "\n\n".join(text_parts),
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "conversation_id": message.conversation_id,
    }
