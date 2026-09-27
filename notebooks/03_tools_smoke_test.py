# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Tools smoke test
# MAGIC
# MAGIC 1. Builds the schema summary for the Text2SQL prompt from `information_schema`
# MAGIC    (table and column comments included) and saves it to `config/runtime.yaml`.
# MAGIC 2. Exercises `run_sql` (including guard rejections), the Text2SQL tool and `search_reviews`.
# MAGIC 3. Runs the whole agent on four demo questions with MLflow tracing on.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

sys.path.append(os.path.abspath("../src"))

import mlflow

from agent.config import load_settings, write_runtime_values
from agent.llm import make_llm_client
from agent.tools import (
    ReviewSearch,
    SparkExecutor,
    Text2Sql,
    format_schema_context,
    run_sql,
    schema_context_sql,
)
from agent.tracking import setup_mlflow

settings = load_settings()
setup_mlflow(settings)
mlflow.openai.autolog()

# COMMAND ----------

# MAGIC %md ## 1. Schema context from information_schema

# COMMAND ----------

rows = [r.asDict() for r in spark.sql(schema_context_sql(settings)).collect()]
schema_context = format_schema_context(rows)
print(schema_context)
write_runtime_values({"agent": {"schema_context": schema_context}})
settings = load_settings()

# COMMAND ----------

# MAGIC %md ## 2. `run_sql` with guardrails

# COMMAND ----------

executor = SparkExecutor(spark)
checks = [
    "SELECT product, SUM(quantity) AS units FROM silver_transactions GROUP BY product ORDER BY units DESC",
    "DROP TABLE silver_transactions",
    "SELECT * FROM samples.bakehouse.sales_customers",
    "SELECT not_a_column FROM silver_transactions",
]
for q in checks:
    out = run_sql(q, settings, executor)
    print(q, "\n  ->", json.dumps(out, default=str)[:300], "\n")

# COMMAND ----------

# MAGIC %md ## 3. Text2SQL and review search

# COMMAND ----------

llm = make_llm_client()
text2sql = Text2Sql(settings, llm, executor, settings["agent"]["schema_context"])
res = text2sql("Which three franchises had the highest revenue?")
print(res["sql"])
display(spark.createDataFrame(res["rows"], res["columns"]) if res.get("rows") else res)

search = ReviewSearch(settings)
for hit in search("the cookies were stale", k=3):
    print(hit["review_id"], round(hit["score"], 3), hit["text"][:120])

# COMMAND ----------

# MAGIC %md ## 4. End-to-end agent on the demo questions

# COMMAND ----------

from agent.agent import BakehouseAgent

agent = BakehouseAgent(settings, llm_client=llm, spark=spark)
demo = [
    "Which product generated the most revenue?",
    "What do customers complain about most often?",
    "Which franchise had the highest revenue, and what do customers say about it?",
    "Drop the sales transactions table.",
]
for q in demo:
    out = agent.answer(q)
    print("=" * 100, "\nQ:", q, "\nA:", out["answer"])
    print("tools:", out["tool_calls"], "| cited:", out["cited_review_ids"], "| trace:", out["trace_id"])
