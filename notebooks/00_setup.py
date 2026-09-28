# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Setup
# MAGIC
# MAGIC Creates the Unity Catalog schema and volume, discovers which Foundation Model API
# MAGIC endpoints and SQL warehouse exist in this workspace, and writes those choices to
# MAGIC `config/runtime.yaml` so every later notebook (and the served agent) uses the same values.
# MAGIC
# MAGIC Run this first, on serverless compute, from the repo's Git folder.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from databricks.sdk import WorkspaceClient

from agent.config import load_settings, write_runtime_values

settings = load_settings()
w = WorkspaceClient()
print("Workspace:", w.config.host)
print("User:", w.current_user.me().user_name)

# COMMAND ----------

# MAGIC %md ## 1. Schema and volume

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {settings.catalog}.{settings.schema}")  # noqa: F821
spark.sql(f"CREATE VOLUME IF NOT EXISTS {settings.fq(settings['volume'])}")  # noqa: F821
src = settings["source"]
display(spark.sql(f"SHOW TABLES IN {src['catalog']}.{src['schema']}"))  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 2. Foundation Model API endpoints

# COMMAND ----------

endpoints = {e.name: e for e in w.serving_endpoints.list()}
fm_endpoints = sorted(n for n in endpoints if n.startswith("databricks-"))
print(f"{len(fm_endpoints)} Databricks-hosted endpoints found:")
for name in fm_endpoints:
    print("  ", name)


def first_available(candidates):
    for name in candidates:
        if name in endpoints:
            return name
    return None


agent_endpoint = first_available(settings["llm"]["agent_endpoint_candidates"])
judge_endpoint = first_available(settings["llm"]["judge_endpoint_candidates"])
embedding_endpoint = settings["ai_search"]["embedding_endpoint"]

assert agent_endpoint, "None of the agent endpoint candidates exist; edit config/settings.yaml."
assert judge_endpoint, "None of the judge endpoint candidates exist; edit config/settings.yaml."
assert embedding_endpoint in endpoints, f"Embedding endpoint {embedding_endpoint} not found."
print(f"\nAgent LLM:  {agent_endpoint}\nJudge LLM:  {judge_endpoint}\nEmbeddings: {embedding_endpoint}")

# COMMAND ----------

# Quick check that the chosen agent model supports tool calling.
from agent.llm import make_llm_client

llm = make_llm_client(w)
probe = llm.chat.completions.create(
    model=agent_endpoint,
    messages=[{"role": "user", "content": "What is 2 + 2? Use the calculator tool."}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ],
    max_tokens=200,
)
print("Tool call returned:", probe.choices[0].message.tool_calls)
assert probe.choices[0].message.tool_calls, "The agent endpoint did not return a tool call."

# COMMAND ----------

# MAGIC %md ## 3. SQL warehouse
# MAGIC
# MAGIC Free Edition includes one serverless SQL warehouse. The served agent runs its SQL there,
# MAGIC Genie uses it, and MLflow uses it to store traces in Unity Catalog.

# COMMAND ----------

warehouses = list(w.warehouses.list())
for wh in warehouses:
    print(wh.id, wh.name, wh.state, "serverless" if wh.enable_serverless_compute else "")
assert warehouses, "No SQL warehouse found."
warehouse = next((wh for wh in warehouses if wh.enable_serverless_compute), warehouses[0])
print("Using warehouse:", warehouse.name, warehouse.id)

# COMMAND ----------

# MAGIC %md ## 4. Save runtime configuration

# COMMAND ----------

path = write_runtime_values(
    {
        "llm": {"agent_endpoint": agent_endpoint, "judge_endpoint": judge_endpoint},
        "sql": {"warehouse_id": warehouse.id},
    }
)
print("Wrote", path)
print(open(path).read())

# COMMAND ----------

# MAGIC %md ## 5. MLflow experiment

# COMMAND ----------

from agent.tracking import setup_mlflow

settings = load_settings()
experiment_id = setup_mlflow(settings)
print("Experiment:", settings["mlflow"]["experiment_name"], experiment_id)
