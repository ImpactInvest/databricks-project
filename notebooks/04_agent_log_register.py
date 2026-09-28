# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Log the agent to MLflow and register it in Unity Catalog
# MAGIC
# MAGIC * **Models-from-code**: MLflow stores `src/agent/serving_entry.py` and the `agent` package
# MAGIC   rather than a pickled object, so the served code is exactly the code in Git.
# MAGIC * **model_config**: the resolved settings (endpoints, warehouse, schema summary, prompt
# MAGIC   variant) travel with the model.
# MAGIC * **resources**: the endpoints, index, warehouse and tables the agent uses are declared, so
# MAGIC   Model Serving can issue short-lived credentials for them (automatic auth passthrough).
# MAGIC
# MAGIC Widget `prompt_variant` selects the Text2SQL prompt: `baseline` or `optimised` (after 08).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.dropdown("prompt_variant", "baseline", ["baseline", "optimised"])
prompt_variant = dbutils.widgets.get("prompt_variant")

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)

from agent.config import load_settings
from agent.tracking import setup_mlflow

settings = load_settings(overrides={"agent": {"prompt_variant": prompt_variant}})
assert settings["agent"].get("schema_context"), "Run 03_tools_smoke_test first to build the schema context."
setup_mlflow(settings)

model_config = settings.to_dict()
model_config["sql"]["executor"] = "warehouse"  # no Spark session inside Model Serving

resources = [
    DatabricksServingEndpoint(endpoint_name=settings["llm"]["agent_endpoint"]),
    DatabricksServingEndpoint(endpoint_name=settings["ai_search"]["embedding_endpoint"]),
    DatabricksVectorSearchIndex(index_name=settings.index_name),
    DatabricksSQLWarehouse(warehouse_id=settings["sql"]["warehouse_id"]),
    *[DatabricksTable(table_name=settings.fq(t)) for t in settings.silver_tables],
    DatabricksTable(table_name=settings.table("reviews_chunked")),
]

pip_requirements = [
    "mlflow==3.16.1",
    "databricks-sdk==0.143.0",
    "databricks-ai-search==0.78",
    "openai==2.54.0",
    "httpx==0.28.1",
    "sqlglot==30.20.0",
    "pyyaml==6.0.3",
]

input_example = {"input": [{"role": "user", "content": "Which product generated the most revenue?"}]}

# COMMAND ----------

# MAGIC %md ## Log

# COMMAND ----------

with mlflow.start_run(run_name=f"log_agent_{prompt_variant}") as run:
    mlflow.log_params({"prompt_variant": prompt_variant, "agent_endpoint": settings["llm"]["agent_endpoint"]})
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=os.path.abspath("../src/agent/serving_entry.py"),
        code_paths=[os.path.abspath("../src/agent")],
        model_config=model_config,
        resources=resources,
        pip_requirements=pip_requirements,
        input_example=input_example,
    )
print("Logged:", model_info.model_uri)

# COMMAND ----------

# MAGIC %md ## Validate the logged model before registering
# MAGIC
# MAGIC Loads the model from its artefacts (not from the repo) and asks one question. This is the
# MAGIC same code path Model Serving uses.

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(model_info.model_uri)
response = loaded.predict(input_example)
print(response["output"][-1]["content"][0]["text"])
print({k: v for k, v in response.get("custom_outputs", {}).items() if k in ("tool_calls", "sql_queries")})

# COMMAND ----------

# MAGIC %md ## Register in Unity Catalog

# COMMAND ----------

registered = mlflow.register_model(model_info.model_uri, settings.model_name)
client = mlflow.MlflowClient()
client.set_registered_model_alias(settings.model_name, prompt_variant, registered.version)
print(f"Registered {settings.model_name} version {registered.version} with alias '{prompt_variant}'")
dbutils.jobs.taskValues.set("model_version", int(registered.version))
