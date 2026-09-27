# Databricks notebook source
# MAGIC %md
# MAGIC # 08 Optimise the Text2SQL prompt with DSPy
# MAGIC
# MAGIC 1. Builds a DSPy `Predict` module with the signature `question, schema_context -> sql`,
# MAGIC    starting from the agent's baseline instructions, on the same Databricks LLM endpoint.
# MAGIC 2. Metric: **execution accuracy**. The generated SQL goes through the same guard and executor
# MAGIC    as the agent's tool, and its result must equal the reference SQL's result.
# MAGIC 3. Optimises on the **train** split only (the SQL-bearing questions among the 30 train questions)
# MAGIC    with `MIPROv2(auto="light")`, holding out part of the train split as the DSPy validation set.
# MAGIC 4. Exports the instructions and few-shot demos to `src/agent/optimised_text2sql.json` and the
# MAGIC    MLflow Prompt Registry, logs and registers the `optimised` agent version, and re-runs the exact
# MAGIC    same evaluation as the baseline (07) on the same **test** split, as the MLflow run `dspy_optimised`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import random
import sys
import time

sys.path.append(os.path.abspath("../src"))

import dspy
import mlflow
from databricks.sdk import WorkspaceClient

from agent.config import load_settings
from agent.tools import WarehouseExecutor
from agent.tracking import setup_mlflow
from evals.dspy_text2sql import baseline_program, export_program, lm_usage, make_examples, make_metric
from evals.golden_set import load_jsonl

settings = load_settings()
setup_mlflow(settings)
schema_context = settings["agent"]["schema_context"]
records = load_jsonl(os.path.abspath("../evals/golden_set.jsonl"))
train_records = [r for r in records if r["split"] == "train"]

examples = make_examples(train_records, schema_context)
random.Random(settings["evaluation"]["split_seed"]).shuffle(examples)
n_val = max(5, len(examples) // 3)
valset, trainset = examples[:n_val], examples[n_val:]
print(f"{len(examples)} SQL-bearing train questions -> DSPy train {len(trainset)}, DSPy val {len(valset)}")

# COMMAND ----------

# MAGIC %md ## Configure DSPy against the Databricks endpoint

# COMMAND ----------

w = WorkspaceClient()
token = w.config.authenticate()["Authorization"].split(" ", 1)[1]  # short-lived; never printed
lm = dspy.LM(
    f"databricks/{settings['llm']['agent_endpoint']}",
    api_base=f"{w.config.host}/serving-endpoints",
    api_key=token,
    engine="litellm",
    temperature=0.0,
    max_tokens=800,
    cache=False,
)
dspy.configure(lm=lm)

executor = WarehouseExecutor(settings["sql"]["warehouse_id"])
metric = make_metric(settings, executor)
evaluate_val = dspy.Evaluate(devset=valset, metric=metric, num_threads=4, display_progress=True)

# COMMAND ----------

# MAGIC %md ## Baseline score on the DSPy validation set

# COMMAND ----------

program = baseline_program()
baseline_val = evaluate_val(program)
print("Baseline execution accuracy on DSPy val:", baseline_val.score)

# COMMAND ----------

# MAGIC %md ## Optimise (MIPROv2, light)

# COMMAND ----------

optimiser_settings = {"optimiser": "MIPROv2", "auto": "light", "max_bootstrapped_demos": 3,
                      "max_labeled_demos": 3, "num_threads": 4, "seed": settings["evaluation"]["split_seed"]}
calls_before = len(lm.history)
start = time.time()
optimiser = dspy.MIPROv2(
    metric=metric,
    auto=optimiser_settings["auto"],
    max_bootstrapped_demos=optimiser_settings["max_bootstrapped_demos"],
    max_labeled_demos=optimiser_settings["max_labeled_demos"],
    num_threads=optimiser_settings["num_threads"],
    seed=optimiser_settings["seed"],
)
optimised = optimiser.compile(baseline_program(), trainset=trainset, valset=valset)
elapsed = time.time() - start
optimised_val = evaluate_val(optimised)
print(f"Optimised execution accuracy on DSPy val: {optimised_val.score} (baseline {baseline_val.score})")
print(f"Optimisation took {elapsed / 60:.1f} minutes")

# COMMAND ----------

# MAGIC %md ## Export the optimised prompt

# COMMAND ----------

exported = export_program(optimised)
usage = lm_usage(lm)
exported["provenance"] = {
    **optimiser_settings,
    "dspy_version": dspy.__version__,
    "llm_endpoint": settings["llm"]["agent_endpoint"],
    "dspy_train_size": len(trainset),
    "dspy_val_size": len(valset),
    "val_execution_accuracy_baseline_pct": baseline_val.score,
    "val_execution_accuracy_optimised_pct": optimised_val.score,
    "optimisation_minutes": round(elapsed / 60, 1),
    "lm_calls_total": usage["lm_calls"],
    "lm_calls_during_optimisation": len(lm.history) - calls_before,
    "prompt_tokens": usage["prompt_tokens"],
    "completion_tokens": usage["completion_tokens"],
}
path = os.path.abspath("../src/agent/optimised_text2sql.json")
with open(path, "w") as f:
    json.dump(exported, f, indent=2)
print("Wrote", path)
print(exported["instructions"])
for d in exported["demos"]:
    print("\nQ:", d["question"], "\nSQL:", d["sql"])

# COMMAND ----------

with mlflow.start_run(run_name="dspy_compile") as run:
    mlflow.log_params({k: v for k, v in optimiser_settings.items()})
    mlflow.log_metrics({
        "val_execution_accuracy_baseline_pct": baseline_val.score,
        "val_execution_accuracy_optimised_pct": optimised_val.score,
        "optimisation_minutes": elapsed / 60,
        "lm_calls": usage["lm_calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
    })
    mlflow.log_dict(exported, "optimised_text2sql.json")
    dspy_run_id = run.info.run_id

# Also version the prompt in the MLflow Prompt Registry (Unity Catalog).
demo_text = "\n\n".join(f"Question: {d['question']}\nSQL: {d['sql']}" for d in exported["demos"])
template = (f"{exported['instructions']}\n\nSchema:\n{{{{schema_context}}}}\n\nExamples:\n{demo_text}\n\n"
            "Question: {{question}}\nSQL:")
try:
    prompt = mlflow.genai.register_prompt(
        name=settings.fq("text2sql_prompt"), template=template,
        commit_message=f"DSPy MIPROv2 light, val accuracy {optimised_val.score}", tags={"dspy_run_id": dspy_run_id},
    )
    print("Registered prompt version", prompt.version)
except Exception as e:  # the Prompt Registry may not be enabled on every workspace
    print("Prompt Registry not available; the JSON file is the source of truth.", e)

# COMMAND ----------

# MAGIC %md ## Log, register and evaluate the optimised agent
# MAGIC
# MAGIC Uses the same notebooks as the baseline, so nothing differs except the prompt.

# COMMAND ----------

dbutils.notebook.run("./04_agent_log_register", 3600, {"prompt_variant": "optimised"})
dbutils.notebook.run("./07_evaluate_baseline", 7200, {"prompt_variant": "optimised", "run_name": "dspy_optimised"})

# COMMAND ----------

base = json.load(open(os.path.abspath("../evals/results/baseline.json")))
opt = json.load(open(os.path.abspath("../evals/results/dspy_optimised.json")))
for key in ["pass_rate", "sql_result_match_rate", "correctness_rate", "median_latency_seconds", "mean_total_tokens"]:
    print(f"{key:28s} baseline={base['metrics'].get(key)}  optimised={opt['metrics'].get(key)}")
print("Run IDs:", base["run_id"], opt["run_id"])
