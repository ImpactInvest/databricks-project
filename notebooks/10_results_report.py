# Databricks notebook source
# MAGIC %md
# MAGIC # 10 Results report
# MAGIC
# MAGIC Collects the numbers for the README from the saved run summaries (each carries its MLflow run
# MAGIC ID), and prepares the **judge spot-check**: 10 LLM-judge verdicts sampled from the baseline run
# MAGIC for a human to label. Agreement is only reported once a person has filled in the labels.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os

import pandas as pd

results_dir = os.path.abspath("../evals/results")
base = json.load(open(f"{results_dir}/baseline.json"))
opt = json.load(open(f"{results_dir}/dspy_optimised.json"))
genie_path = f"{results_dir}/genie_comparison.csv"
genie = pd.read_csv(genie_path) if os.path.exists(genie_path) else None


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{100 * x:.0f}%"


def secs(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.1f} s"


# COMMAND ----------

# MAGIC %md ## README results table (Markdown)

# COMMAND ----------

lines = [
    "| Variant | Test pass rate | SQL result match | Groundedness | Median latency | MLflow run ID |",
    "|---|---|---|---|---|---|",
]
for name, r in [("Baseline agent", base), ("DSPy-optimised agent", opt)]:
    m = r["metrics"]
    lines.append(f"| {name} | {pct(m['pass_rate'])} | {pct(m['sql_result_match_rate'])} | "
                 f"{pct(m['groundedness_rate'])} | {secs(m['median_latency_seconds'])} | `{r['run_id']}` |")
if genie is not None:
    g = genie[genie["variant"] == "genie_agent"].iloc[0]
    lines.append(f"| Genie Agent (SQL questions only) | n/a | {pct(g['sql_result_match'])} | n/a | "
                 f"{secs(g['median_latency_seconds'])} | `{g['run_id']}` |")
print("\n".join(lines))

print("\nPer category pass rate:")
for key in sorted(k for k in base["metrics"] if k.startswith("pass_rate_")):
    print(f"  {key[10:]:8s} baseline {pct(base['metrics'][key])}  optimised {pct(opt['metrics'].get(key))}")

if genie is not None:
    print("\nSQL-only comparison (identical question set):")
    display(genie)

# COMMAND ----------

# MAGIC %md ## Judge spot-check (10 verdicts)
# MAGIC
# MAGIC Writes `evals/results/judge_spot_check.csv`. Fill the `human_verdict` column with yes or no by
# MAGIC reading each question, answer and expected facts, then re-run the next cell to compute agreement.

# COMMAND ----------

per_q = pd.read_csv(f"{results_dir}/baseline_per_question.csv")
golden = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.abspath("../evals/golden_set.jsonl"))}
spot_path = f"{results_dir}/judge_spot_check.csv"
if not os.path.exists(spot_path):
    sample = per_q.sample(n=min(10, len(per_q)), random_state=7)
    sample = sample.assign(
        expected_facts=[json.dumps(golden[i]["expectations"]["expected_facts"]) for i in sample["id"]],
        judge_verdict=sample["correctness"], human_verdict="",
    )[["id", "question", "answer", "expected_facts", "judge_verdict", "correctness_rationale", "human_verdict"]]
    sample.to_csv(spot_path, index=False)
    print("Wrote", spot_path, "for labelling.")
display(pd.read_csv(spot_path))

# COMMAND ----------

spot = pd.read_csv(spot_path).fillna("")
labelled = spot[spot["human_verdict"].str.strip() != ""]
if len(labelled):
    agree = (labelled["human_verdict"].str.lower().str.strip() == labelled["judge_verdict"].astype(str).str.lower()).mean()
    print(f"Judge and human agree on {agree:.0%} of {len(labelled)} labelled verdicts.")
else:
    print("No human labels yet; agreement not reported.")
