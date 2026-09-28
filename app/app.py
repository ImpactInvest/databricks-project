"""Bakehouse analytics assistant: a Databricks App in front of the agent endpoint.

For every answer the app shows the SQL the agent ran, a preview of the rows
it got back, and the review snippets it retrieved (cited ones first). Thumbs
up or down is logged to MLflow as human feedback on the answer's trace.

Configuration comes from environment variables set in ``app.yaml``:
``SERVING_ENDPOINT`` (from the app's serving-endpoint resource) and
``MLFLOW_EXPERIMENT_NAME``. Authentication uses the app's service principal,
which the Databricks SDK picks up automatically.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient

ENDPOINT = os.environ.get("SERVING_ENDPOINT", "bakehouse-agent")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/bakehouse-genai-agent")

EXAMPLES = [
    "Which product generated the most revenue?",
    "What do customers complain about most often?",
    "Which franchise had the highest revenue, and what do customers say about it?",
    "Drop the sales transactions table.",
]

st.set_page_config(page_title="Bakehouse assistant", page_icon="🥐", layout="wide")


@st.cache_resource
def workspace() -> WorkspaceClient:
    return WorkspaceClient()


def ask(history: list[dict]) -> dict:
    """Call the serving endpoint with the conversation so far."""
    resp = workspace().api_client.do(
        "POST",
        f"/serving-endpoints/{ENDPOINT}/invocations",
        body={"input": [{"role": m["role"], "content": m["content"]} for m in history]},
    )
    text = ""
    for item in reversed(resp.get("output", [])):
        if item.get("type") == "message":
            text = "".join(c.get("text", "") for c in item.get("content", []))
            break
    return {"content": text, "details": resp.get("custom_outputs", {}) or {}}


def log_feedback(trace_id: str | None, positive: bool) -> str:
    if not trace_id:
        return "No trace ID was returned, so feedback could not be linked."
    try:
        import mlflow
        from mlflow.entities import AssessmentSource

        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(EXPERIMENT)
        user = st.context.headers.get("X-Forwarded-Email", "app-user")
        mlflow.log_feedback(
            trace_id=trace_id,
            name="user_feedback",
            value=positive,
            source=AssessmentSource(source_type="HUMAN", source_id=user),
        )
        return "Thanks, feedback recorded in MLflow."
    except Exception as e:  # noqa: BLE001 - feedback must never break the chat
        return f"Feedback not recorded: {e}"


def render_details(details: dict, key: str) -> None:
    for i, res in enumerate(details.get("sql_results", [])):
        with st.expander(f"SQL query {i + 1} ({res['row_count']} rows)", expanded=False):
            st.code(res["sql"], language="sql")
            if res["rows"]:
                st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]), hide_index=True)
    failed = [
        q
        for q in details.get("sql_queries", [])
        if q not in {r["sql"] for r in details.get("sql_results", [])}
    ]
    for q in failed:
        with st.expander("SQL attempt that was rejected or failed"):
            st.code(q or "(empty)", language="sql")

    reviews = details.get("retrieved_reviews", [])
    if reviews:
        cited = set(details.get("cited_review_ids", []))
        reviews = sorted(reviews, key=lambda r: r["review_id"] not in cited)
        with st.expander(f"Reviews retrieved ({len(reviews)}, {len(cited)} cited)"):
            for r in reviews:
                marker = "**cited**" if r["review_id"] in cited else "retrieved"
                st.markdown(
                    f"{marker} · review {r['review_id']} · franchise {r['franchise_id']} · {r['review_date']}\n\n"
                    f"> {r['text']}"
                )
    st.caption(f"Tool calls: {details.get('tool_calls', 0)} · trace: {details.get('trace_id') or 'n/a'}")

    rating = st.feedback("thumbs", key=f"fb-{key}")
    if rating is not None and not st.session_state.get(f"sent-{key}"):
        st.session_state[f"sent-{key}"] = True
        st.toast(log_feedback(details.get("trace_id"), positive=bool(rating)))


st.title("Bakehouse analytics assistant")
st.caption(
    "Ask about sales, franchises, suppliers and customer reviews. "
    f"Answers come from a tool-calling agent served on the `{ENDPOINT}` endpoint."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Try")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    if st.button("Clear conversation"):
        st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_details(msg.get("details", {}), key=str(idx))

prompt = st.chat_input("Ask a question") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking (querying tables and reviews)..."):
            try:
                reply = ask(st.session_state.messages)
            except Exception as e:  # noqa: BLE001
                reply = {"content": f"The agent endpoint returned an error: {e}", "details": {}}
        st.markdown(reply["content"])
        st.session_state.messages.append({"role": "assistant", **reply})
        render_details(reply["details"], key=str(len(st.session_state.messages) - 1))
