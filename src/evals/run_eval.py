"""One evaluation run, shared by the baseline (07) and DSPy-optimised (08) notebooks.

Both runs call ``run_evaluation`` with the same test rows and scorers; only
the agent's prompt variant differs. That is what makes the comparison fair.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from evals.golden_set import to_eval_rows
from evals.scorers import build_scorers, compute_pass

ID_KEY = "question"


def judge_uri(settings) -> str:
    return f"databricks:/{settings['llm']['judge_endpoint']}"


def summarise(result_df: pd.DataFrame, records: list[dict]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add ``category`` and ``passed`` to the per-question results and compute
    the headline metrics (overall and per category)."""
    by_question = {r["inputs"]["question"]: r for r in records}
    rows = []
    for _, row in result_df.iterrows():
        question = _question(row)
        rec = by_question[question]
        has_ref = bool(rec["expectations"].get("reference_sql"))
        rows.append(
            {
                "id": rec["id"],
                "category": rec["category"],
                "tags": ",".join(rec["tags"]),
                "question": question,
                "answer": _text(row.get("response")),
                "correctness": row.get("correctness/value"),
                "safety": row.get("safety/value"),
                "sql_result_match": row.get("sql_result_match/value"),
                "sql_executes": row.get("sql_executes/value"),
                "retrieval_groundedness": row.get("retrieval_groundedness/value"),
                "relevance_to_query": row.get("relevance_to_query/value"),
                "cites_review_ids": row.get("cites_review_ids/value"),
                "refuses_unsafe": row.get("refuses_unsafe/value"),
                "latency_seconds": row.get("latency_seconds/value"),
                "total_tokens": row.get("total_tokens/value"),
                "tool_calls_count": row.get("tool_calls_count/value"),
                "correctness_rationale": row.get("correctness/rationale"),
                "sql_match_rationale": row.get("sql_result_match/rationale"),
                "trace_id": row.get("trace_id"),
                "passed": compute_pass(
                    row.get("correctness/value"),
                    row.get("safety/value"),
                    row.get("sql_result_match/value"),
                    has_ref,
                ),
            }
        )
    df = pd.DataFrame(rows)
    return df, headline_metrics(df)


def _question(row) -> str:
    request = row.get("request")
    if isinstance(request, str):
        try:
            request = json.loads(request)
        except json.JSONDecodeError:
            return request
    if isinstance(request, dict):
        return request.get(ID_KEY) or request.get("inputs", {}).get(ID_KEY)
    return str(request)


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _yes_rate(series: pd.Series) -> float | None:
    values = [v for v in series if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if not values:
        return None
    yes = sum(1 for v in values if str(v).lower() in {"yes", "true", "1", "1.0"} or v is True)
    return round(yes / len(values), 4)


def headline_metrics(df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "n_questions": len(df),
        "pass_rate": round(df["passed"].mean(), 4),
        "sql_result_match_rate": _yes_rate(df["sql_result_match"]),
        "correctness_rate": _yes_rate(df["correctness"]),
        "groundedness_rate": _yes_rate(df["retrieval_groundedness"]),
        "safety_rate": _yes_rate(df["safety"]),
        "median_latency_seconds": float(df["latency_seconds"].median()),
        "mean_total_tokens": float(df["total_tokens"].dropna().mean())
        if df["total_tokens"].notna().any()
        else None,
    }
    for cat, part in df.groupby("category"):
        metrics[f"pass_rate_{cat}"] = round(part["passed"].mean(), 4)
    return metrics


def run_evaluation(agent, settings, records: list[dict], run_name: str, split: str = "test"):
    """Evaluate ``agent`` on the golden set split and log everything to MLflow.

    Returns ``(run_id, per_question_df, metrics)``.
    """
    import mlflow

    data = to_eval_rows(records, split)
    split_records = [r for r in records if r["split"] == split]

    # Tracing predict_fn makes it the root span, so every trace (and every judge)
    # sees the plain question as input and the plain answer as output.
    @mlflow.trace(name="answer_question", span_type="CHAIN")
    def predict_fn(question: str) -> str:
        return agent.answer(question)["answer"]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "prompt_variant": settings["agent"]["prompt_variant"],
                "agent_endpoint": settings["llm"]["agent_endpoint"],
                "judge_endpoint": settings["llm"]["judge_endpoint"],
                "split": split,
                "n_questions": len(data),
                "max_tool_calls": settings["agent"]["max_tool_calls"],
            }
        )
        result = mlflow.genai.evaluate(
            data=data, predict_fn=predict_fn, scorers=build_scorers(judge_uri(settings))
        )
        per_question, metrics = summarise(result.result_df, split_records)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_table(per_question, artifact_file="per_question_results.json")
        return run.info.run_id, per_question, metrics


def worst_failures(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The ``n`` failed questions to study first: SQL mismatches and incorrect answers,
    most expensive first."""
    failed = df[~df["passed"]].copy()
    failed["_severity"] = failed["correctness"].astype(str).str.lower().eq("no").astype(int) * 2 + failed[
        "sql_result_match"
    ].astype(str).str.lower().isin(["false", "0"]).astype(int)
    cols = [
        "id",
        "category",
        "question",
        "answer",
        "correctness_rationale",
        "sql_match_rationale",
        "trace_id",
    ]
    return failed.sort_values(["_severity", "latency_seconds"], ascending=False)[cols].head(n)
