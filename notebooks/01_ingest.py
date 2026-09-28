# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Ingest: `samples.bakehouse` to silver Delta tables (PySpark)
# MAGIC
# MAGIC * Reads the source tables with `spark.table`.
# MAGIC * Cleans them with the PySpark DataFrame API (`src/pipeline/transforms.py`): typed columns,
# MAGIC   de-duplication, null handling, derived columns (`transaction_date`, `transaction_hour`,
# MAGIC   `day_of_week`, `revenue`) and removal of personal data the agent never needs.
# MAGIC * Chunks the reviews (sentence windows, see `src/pipeline/chunking.py`) into `reviews_chunked`
# MAGIC   with Change Data Feed enabled, which the AI Search Delta Sync index requires.
# MAGIC * Adds table and column comments, which both the Text2SQL prompt and Genie read.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from pyspark.sql import functions as F

from agent.config import load_settings
from pipeline import transforms as tr

settings = load_settings()
src = f"{settings['source']['catalog']}.{settings['source']['schema']}"
print("Source:", src, "Target:", f"{settings.catalog}.{settings.schema}")

# COMMAND ----------

# MAGIC %md ## 1. Read and inspect the sources

# COMMAND ----------

sources = {
    name: spark.table(f"{src}.{name}")
    for name in ["sales_transactions", "sales_customers", "sales_franchises", "sales_suppliers", "media_customer_reviews"]
}
for name, df in sources.items():
    print(f"{name}: {df.count():,} rows")
    df.printSchema()

# COMMAND ----------

# MAGIC %md ## 2. Transform

# COMMAND ----------

silver = {
    settings["tables"]["transactions"]: tr.transform_transactions(sources["sales_transactions"]),
    settings["tables"]["customers"]: tr.transform_customers(sources["sales_customers"]),
    settings["tables"]["franchises"]: tr.transform_franchises(sources["sales_franchises"]),
    settings["tables"]["suppliers"]: tr.transform_suppliers(sources["sales_suppliers"]),
    settings["tables"]["reviews"]: tr.transform_reviews(sources["media_customer_reviews"]),
}
chunk_cfg = settings["chunking"]
silver[settings["tables"]["reviews_chunked"]] = tr.chunk_reviews(
    silver[settings["tables"]["reviews"]],
    max_words=chunk_cfg["max_words"],
    overlap_sentences=chunk_cfg["overlap_sentences"],
)

# Data-quality checks before writing anything.
tx = silver[settings["tables"]["transactions"]]
assert tx.where(F.col("revenue").isNull()).count() == 0, "revenue has nulls"
assert tx.groupBy("transaction_id").count().where("count > 1").count() == 0, "duplicate transaction_id"
orphans = tx.join(silver[settings["tables"]["franchises"]], "franchise_id", "left_anti").count()
print("Transactions whose franchise is missing:", orphans)

# COMMAND ----------

# MAGIC %md ## 3. Write Delta tables

# COMMAND ----------

for name, df in silver.items():
    target = settings.fq(name)
    (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(target))
    print(f"{target}: {spark.table(target).count():,} rows")

chunked = settings.table("reviews_chunked")
spark.sql(f"ALTER TABLE {chunked} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md ## 4. Table and column comments

# COMMAND ----------


def sql_str(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


for table, (table_comment, columns) in tr.TABLE_COMMENTS.items():
    fq = settings.fq(table)
    spark.sql(f"COMMENT ON TABLE {fq} IS '{sql_str(table_comment)}'")
    for column, comment in columns.items():
        spark.sql(f"ALTER TABLE {fq} ALTER COLUMN {column} COMMENT '{sql_str(comment)}'")
print("Comments applied.")

# COMMAND ----------

# MAGIC %md ## 5. Profile (useful context for the golden set and the walkthrough)

# COMMAND ----------

display(
    spark.table(settings.table("transactions")).agg(
        F.min("transaction_date").alias("first_date"),
        F.max("transaction_date").alias("last_date"),
        F.count("*").alias("transactions"),
        F.round(F.sum("revenue"), 2).alias("revenue"),
        F.countDistinct("franchise_id").alias("franchises_with_sales"),
        F.countDistinct("product").alias("products"),
    )
)
display(spark.table(settings.table("reviews_chunked")).groupBy("review_id").count().groupBy("count").count())
