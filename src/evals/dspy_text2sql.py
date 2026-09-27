"""DSPy program and metric for optimising the Text2SQL prompt.

The DSPy module mirrors the agent's Text2SQL step exactly:
``question, schema_context -> sql``, starting from the same baseline
instructions. The metric is execution accuracy: the generated SQL is passed
through the same guard and executor as the agent's tool, and its result is
compared with the reference result using the same ``results_match`` as the
``sql_result_match`` scorer.

After optimisation, ``export_program`` extracts the instructions and few-shot
demos so ``agent/prompts.py`` can render them in the agent's own chat
template. The agent does not run DSPy at serving time.
"""

from __future__ import annotations

from typing import Any

import dspy

from agent.config import Settings
from agent.prompts import TEXT2SQL_BASELINE_INSTRUCTIONS, extract_sql
from agent.tools import run_sql
from evals.scorers import results_match


class Text2SQL(dspy.Signature):
    question: str = dspy.InputField(desc="A business question about the bakery data")
    schema_context: str = dspy.InputField(desc="Tables, columns and comments that may be queried")
    sql: str = dspy.OutputField(desc="One Databricks SQL SELECT query, with no explanation or code fences")


def baseline_program() -> dspy.Module:
    """A plain Predict module (no chain of thought), so the exported demos are
    exactly question -> SQL pairs the agent can reuse."""
    return dspy.Predict(Text2SQL.with_instructions(TEXT2SQL_BASELINE_INSTRUCTIONS))


def make_examples(records: list[dict], schema_context: str) -> list[dspy.Example]:
    """Golden-set records with a reference SQL become DSPy examples."""
    examples = []
    for r in records:
        exp = r["expectations"]
        if not exp.get("reference_sql"):
            continue
        examples.append(
            dspy.Example(
                question=r["inputs"]["question"],
                schema_context=schema_context,
                sql=exp["reference_sql"],
                expected_result=exp["expected_result"],
                id=r["id"],
            ).with_inputs("question", "schema_context")
        )
    return examples


def make_metric(settings: Settings, executor):
    """Execution accuracy: 1.0 when the generated SQL returns the reference result."""

    def execution_accuracy(example: dspy.Example, prediction, trace=None) -> float:
        sql = extract_sql(getattr(prediction, "sql", "") or "")
        result = run_sql(sql, settings, executor)
        if "error" in result:
            return 0.0
        expected = example.expected_result
        ok = results_match(expected["columns"], expected["rows"], result["columns"], result["rows"])
        return 1.0 if ok else 0.0

    return execution_accuracy


def export_program(program: dspy.Module) -> dict[str, Any]:
    """Instructions and demos from a (compiled) Predict program."""
    predictor = program.predictors()[0]
    demos = []
    for demo in predictor.demos:
        get = demo.get if hasattr(demo, "get") else lambda k, d=None, _demo=demo: getattr(_demo, k, d)
        question, sql = get("question"), get("sql")
        if question and sql:
            demos.append({"question": question, "sql": extract_sql(sql)})
    return {"instructions": predictor.signature.instructions, "demos": demos}


def lm_usage(lm) -> dict[str, float]:
    """Calls, tokens and (where the provider reports it) cost from ``lm.history``."""
    calls, prompt, completion, cost = 0, 0, 0, 0.0
    for entry in getattr(lm, "history", []) or []:
        calls += 1
        usage = entry.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        cost += float(entry.get("cost") or 0.0)
    return {
        "lm_calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reported_cost": cost,
    }
