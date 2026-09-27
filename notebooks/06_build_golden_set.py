# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Build the golden set (computed ground truth)
# MAGIC
# MAGIC Question specifications live in `src/evals/golden_set.py`. This notebook:
# MAGIC
# MAGIC 1. runs each `params_sql` and `reference_sql` against the silver tables, so every expected
# MAGIC    number, name and date comes from the data rather than being typed by hand;
# MAGIC 2. for review and hybrid questions, retrieves the relevant reviews and asks the judge LLM to
# MAGIC    list the facts a correct answer needs, with supporting `review_id`s (flagged for spot-check);
# MAGIC 3. assigns a deterministic, stratified 30-question train split (for DSPy) and a held-out test
# MAGIC    split (for all reported numbers);
# MAGIC 4. writes `evals/golden_set.jsonl` (commit it) and a Delta table copy.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# The golden set is built once, spot-checked and committed. Later runs reuse it so that
# every evaluation scores against the same ground truth. Set overwrite=true to rebuild.
dbutils.widgets.dropdown("overwrite", "false", ["false", "true"])
overwrite = dbutils.widgets.get("overwrite") == "true"

# COMMAND ----------

import json
import os
import sys

sys.path.append(os.path.abspath("../src"))

from agent.config import load_settings
from agent.llm import make_llm_client
from agent.tools import ReviewSearch, to_jsonable
from evals.golden_set import SPECS, assign_splits, build_record, write_jsonl

settings = load_settings()
path = os.path.abspath("../evals/golden_set.jsonl")
if os.path.exists(path) and not overwrite:
    dbutils.notebook.exit(f"{path} already exists; set overwrite=true to rebuild it.")
llm = make_llm_client()
search = ReviewSearch(settings)


def run_query(sql):
    df = spark.sql(sql)
    return df.columns, [[to_jsonable(v) for v in r] for r in df.collect()]


# COMMAND ----------

records = []
for spec in SPECS:
    rec = build_record(spec, settings, run_query, search_reviews=search, llm=llm,
                       judge_model=settings["llm"]["judge_endpoint"])
    records.append(rec)
    print(f"{rec['id']} [{rec['category']}] {rec['inputs']['question']}")
    for fact in rec["expectations"]["expected_facts"]:
        print("    -", fact)

records = assign_splits(records, settings["evaluation"]["train_size"], settings["evaluation"]["split_seed"])

# COMMAND ----------

# MAGIC %md ## Spot-check the LLM-extracted review facts
# MAGIC
# MAGIC Read these before trusting the review-question scores. Edit `evals/golden_set.jsonl` by hand
# MAGIC if a fact is wrong, and note the edit in `docs/DESIGN.md`.

# COMMAND ----------

for r in records:
    if r["facts_source"] != "computed_from_sql" and r["facts_source"] != "fixed":
        print(f"{r['id']} ({r['split']}): {r['inputs']['question']}")
        print("   facts:", r["expectations"]["expected_facts"])
        print("   supporting review ids:", r["expectations"].get("supporting_review_ids"))

# COMMAND ----------

# MAGIC %md ## Save

# COMMAND ----------

write_jsonl(records, path)
print("Wrote", path)

rows = [
    (r["id"], r["category"], r["tags"], r["split"], r["facts_source"], r["inputs"]["question"],
     json.dumps(r["expectations"], default=str))
    for r in records
]
df = spark.createDataFrame(rows, "id string, category string, tags array<string>, split string, "
                                 "facts_source string, question string, expectations_json string")
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(settings.table("golden_set"))
display(df.groupBy("split", "category").count().orderBy("split", "category"))
