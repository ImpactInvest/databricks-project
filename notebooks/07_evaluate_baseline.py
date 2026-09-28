# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Evaluate the baseline agent (MLflow 3 GenAI)
# MAGIC
# MAGIC Runs `mlflow.genai.evaluate` on the **test** split with:
# MAGIC
# MAGIC * LLM judges: `Correctness` (against `expected_facts`), `RetrievalGroundedness` (on questions
# MAGIC   where reviews were retrieved), `RelevanceToQuery`, `Safety`, and two `Guidelines`
# MAGIC   (cites review ids; refuses unsafe requests);
# MAGIC * code scorers that read the trace: `sql_executes`, `sql_result_match` (the key deterministic
# MAGIC   metric), `tool_calls_count`, `latency_seconds`, `total_tokens`.
# MAGIC
# MAGIC **Pass** = Correctness yes AND Safety yes AND (for questions with a reference SQL)
# MAGIC `sql_result_match` true. The agent runs its SQL on the SQL warehouse, as it does when served.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.dropdown("prompt_variant", "baseline", ["baseline", "optimised"])
dbutils.widgets.text("run_name", "baseline")
prompt_variant = dbutils.widgets.get("prompt_variant")
run_name = dbutils.widgets.get("run_name")

# COMMAND ----------

import json
import os
import sys

sys.path.append(os.path.abspath("../src"))

import mlflow

from agent.agent import BakehouseAgent
from agent.config import load_settings
from agent.tracking import setup_mlflow
from evals.golden_set import load_jsonl
from evals.run_eval import run_evaluation, worst_failures

settings = load_settings(overrides={"agent": {"prompt_variant": prompt_variant}, "sql": {"executor": "warehouse"}})
setup_mlflow(settings)
mlflow.openai.autolog()

records = load_jsonl(os.path.abspath("../evals/golden_set.jsonl"))
print(sum(r["split"] == "test" for r in records), "test questions")

agent = BakehouseAgent(settings)

# COMMAND ----------

run_id, per_question, metrics = run_evaluation(agent, settings, records, run_name=run_name)
print("MLflow run ID:", run_id)
print(json.dumps(metrics, indent=2))

# COMMAND ----------

display(per_question)

# COMMAND ----------

# MAGIC %md ## Error analysis: the five worst failures

# COMMAND ----------

worst = worst_failures(per_question, n=5)
display(worst)
with mlflow.start_run(run_id=run_id):
    mlflow.log_table(worst, artifact_file="worst_failures.json")

# COMMAND ----------

# Save a compact summary next to the notebooks for the results report (10) and the README.
os.makedirs(os.path.abspath("../evals/results"), exist_ok=True)
with open(os.path.abspath(f"../evals/results/{run_name}.json"), "w") as f:
    json.dump({"run_name": run_name, "run_id": run_id, "prompt_variant": prompt_variant, "metrics": metrics}, f, indent=2)
per_question.to_csv(os.path.abspath(f"../evals/results/{run_name}_per_question.csv"), index=False)
