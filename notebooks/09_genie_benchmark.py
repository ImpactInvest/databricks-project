# Databricks notebook source
# MAGIC %md
# MAGIC # 09 Benchmark against a Genie Agent (SQL questions only)
# MAGIC
# MAGIC **Scope:** Genie answers questions over tables; it does not search the review documents. So the
# MAGIC comparison uses only the **SQL-category test questions that have a reference SQL**. The same
# MAGIC questions are pulled from the baseline and optimised runs, so all three rows of the table are
# MAGIC computed on an identical question set.
# MAGIC
# MAGIC **Before running:** create the Genie Agent in the UI (steps in `docs/DESIGN.md`, section
# MAGIC "Genie setup") and paste its ID into the widget or into `genie.space_id` in `config/runtime.yaml`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("genie_space_id", "")

# COMMAND ----------

import json
import os
import sys
import time

sys.path.append(os.path.abspath("../src"))

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from mlflow.genai.scorers import Correctness

from agent.config import load_settings, write_runtime_values
from agent.tracking import setup_mlflow
from evals.genie import ask_genie
from evals.golden_set import load_jsonl
from evals.run_eval import judge_uri
from evals.scorers import results_match

settings = load_settings()
space_id = dbutils.widgets.get("genie_space_id") or settings["genie"]["space_id"]
assert space_id, "Set the Genie space ID (widget or config)."
write_runtime_values({"genie": {"space_id": space_id}})
setup_mlflow(settings)
w = WorkspaceClient()

records = load_jsonl(os.path.abspath("../evals/golden_set.jsonl"))
sql_test = [r for r in records if r["split"] == "test" and r["category"] == "sql" and r["expectations"].get("reference_sql")]
print(len(sql_test), "SQL test questions")

# COMMAND ----------

# MAGIC %md ## Ask Genie

# COMMAND ----------

correctness = Correctness(model=judge_uri(settings))
rows = []
for r in sql_test:
    q = r["inputs"]["question"]
    out = ask_genie(w, space_id, q)
    exp = r["expectations"]["expected_result"]
    match = bool(out["rows"]) and results_match(exp["columns"], exp["rows"], out["columns"], out["rows"])
    verdict = correctness(inputs={"question": q}, outputs=out["answer"],
                          expectations={"expected_facts": r["expectations"]["expected_facts"]})
    rows.append({"id": r["id"], "question": q, "status": out["status"], "sql": out["sql"],
                 "sql_result_match": match, "correctness": verdict.value, "latency_seconds": out["latency_seconds"],
                 "answer": out["answer"][:500], "error": out["error"]})
    print(f"{r['id']} match={match} correct={verdict.value} {out['latency_seconds']}s {out['status']}")
    time.sleep(5)  # stay well inside the Genie API rate limit

genie_df = pd.DataFrame(rows)
display(genie_df)

# COMMAND ----------

# MAGIC %md ## Comparison table (identical question set)

# COMMAND ----------


def yes_rate(series):
    vals = [str(v).lower() for v in series if v is not None and str(v) != "nan"]
    return round(sum(v in ("yes", "true", "1", "1.0") for v in vals) / len(vals), 4) if vals else None


ids = set(genie_df["id"])
summary = []
for variant in ["baseline", "dspy_optimised"]:
    path = os.path.abspath(f"../evals/results/{variant}_per_question.csv")
    meta = json.load(open(os.path.abspath(f"../evals/results/{variant}.json")))
    df = pd.read_csv(path)
    df = df[df["id"].isin(ids)]
    summary.append({"variant": variant, "run_id": meta["run_id"], "n": len(df),
                    "pass_rate": round(df["passed"].mean(), 4),
                    "sql_result_match": yes_rate(df["sql_result_match"]),
                    "correctness": yes_rate(df["correctness"]),
                    "median_latency_seconds": float(df["latency_seconds"].median())})

with mlflow.start_run(run_name="genie_benchmark") as run:
    genie_metrics = {"sql_result_match": yes_rate(genie_df["sql_result_match"]),
                     "correctness": yes_rate(genie_df["correctness"]),
                     "median_latency_seconds": float(genie_df["latency_seconds"].dropna().median())}
    mlflow.log_param("genie_space_id", space_id)
    mlflow.log_metrics({k: v for k, v in genie_metrics.items() if v is not None})
    mlflow.log_table(genie_df, artifact_file="genie_answers.json")
    summary.append({"variant": "genie_agent", "run_id": run.info.run_id, "n": len(genie_df), "pass_rate": None,
                    **genie_metrics})

summary_df = pd.DataFrame(summary)
display(summary_df)
os.makedirs(os.path.abspath("../evals/results"), exist_ok=True)
summary_df.to_csv(os.path.abspath("../evals/results/genie_comparison.csv"), index=False)
genie_df.to_csv(os.path.abspath("../evals/results/genie_per_question.csv"), index=False)
