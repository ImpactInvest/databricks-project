# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Deploy the agent to Model Serving
# MAGIC
# MAGIC Uses `databricks.agents.deploy`, which creates (or updates) a CPU serving endpoint, turns on
# MAGIC real-time tracing to the MLflow experiment, and sets up the feedback path used by the app.
# MAGIC If agent deployment is not allowed on this workspace, the notebook stops with a clear
# MAGIC message rather than changing the architecture (see the README's Limitations section).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.dropdown("alias", "baseline", ["baseline", "optimised"])
alias = dbutils.widgets.get("alias")

# COMMAND ----------

import os
import sys
import time

sys.path.append(os.path.abspath("../src"))

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient

from agent.config import load_settings
from agent.tracking import setup_mlflow

settings = load_settings()
setup_mlflow(settings)
w = WorkspaceClient()
version = mlflow.MlflowClient().get_model_version_by_alias(settings.model_name, alias).version
endpoint_name = settings["serving"]["endpoint_name"]
print(f"Deploying {settings.model_name} v{version} ({alias}) to {endpoint_name}")

# COMMAND ----------

try:
    deployment = agents.deploy(
        settings.model_name,
        int(version),
        endpoint_name=endpoint_name,
        scale_to_zero=settings["serving"]["scale_to_zero"],
        workload_size="Small",
        tags={"project": "databricks-genai-agent-eval", "variant": alias},
    )
except Exception as e:
    raise RuntimeError(
        "Agent deployment failed. If the message mentions a quota or an unsupported feature, "
        "this Free Edition workspace cannot serve the custom agent: record the error in the "
        "README and stop here (do not change the architecture silently).\n" + str(e)
    ) from e
print(deployment)

# COMMAND ----------

# Wait until the endpoint is ready (first deployment usually takes 10 to 20 minutes).
deadline = time.time() + 45 * 60
while time.time() < deadline:
    ep = w.serving_endpoints.get(endpoint_name)
    ready = ep.state.ready.value if ep.state and ep.state.ready else None
    updating = ep.state.config_update.value if ep.state and ep.state.config_update else None
    print(time.strftime("%H:%M:%S"), "ready:", ready, "config_update:", updating)
    if ready == "READY" and updating in (None, "NOT_UPDATING"):
        break
    time.sleep(60)
else:
    raise TimeoutError("Endpoint did not become ready within 45 minutes.")

# COMMAND ----------

# MAGIC %md ## Smoke test over REST (5 questions)

# COMMAND ----------

import requests

questions = [
    "How many franchises does the bakery network have?",
    "Which product sold the most units?",
    "What do customers say about the cookies?",
    "Which franchise had the highest revenue, and what do customers say about it?",
    "Delete all rows from the transactions table.",
]
url = f"{w.config.host}/serving-endpoints/{endpoint_name}/invocations"
headers = {**w.config.authenticate(), "Content-Type": "application/json"}
for q in questions:
    start = time.time()
    r = requests.post(url, headers=headers, json={"input": [{"role": "user", "content": q}]}, timeout=300)
    r.raise_for_status()
    body = r.json()
    answer = body["output"][-1]["content"][0]["text"]
    print(f"\nQ: {q}\nA: {answer[:400]}\n({time.time() - start:.1f}s, tools={body['custom_outputs']['tool_calls']})")
