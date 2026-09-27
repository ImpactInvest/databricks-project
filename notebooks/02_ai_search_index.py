# Databricks notebook source
# MAGIC %md
# MAGIC # 02 AI Search index over review chunks
# MAGIC
# MAGIC Databricks AI Search (formerly Vector Search). Free Edition allows one endpoint with one
# MAGIC search unit and does not support Direct Vector Access, so this uses a **Delta Sync index**
# MAGIC with **managed embeddings**: AI Search reads `reviews_chunked` through Change Data Feed and
# MAGIC calls the embedding endpoint itself.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))

from databricks.ai_search.client import AISearchClient

from agent.config import load_settings
from agent.tools import REVIEW_COLUMNS, parse_search_response

settings = load_settings()
cfg = settings["ai_search"]
client = AISearchClient(disable_notice=True)

# COMMAND ----------

# MAGIC %md ## 1. Endpoint (one per Free Edition workspace)

# COMMAND ----------

existing = [e["name"] for e in client.list_endpoints().get("endpoints", [])]
print("Existing endpoints:", existing)
if cfg["endpoint_name"] not in existing:
    if existing:
        raise RuntimeError(
            f"Free Edition allows one AI Search endpoint and '{existing[0]}' already exists. "
            f"Set ai_search.endpoint_name to '{existing[0]}' in config/settings.yaml, or delete it."
        )
    client.create_endpoint_and_wait(name=cfg["endpoint_name"], endpoint_type="STANDARD", verbose=True)
client.wait_for_endpoint(cfg["endpoint_name"], verbose=True)

# COMMAND ----------

# MAGIC %md ## 2. Delta Sync index with managed embeddings

# COMMAND ----------

index_name = settings.index_name
if client.index_exists(endpoint_name=cfg["endpoint_name"], index_name=index_name):
    index = client.get_index(endpoint_name=cfg["endpoint_name"], index_name=index_name)
    print("Index exists; triggering a sync so it picks up the latest table version.")
    index.sync()
else:
    index = client.create_delta_sync_index(
        endpoint_name=cfg["endpoint_name"],
        index_name=index_name,
        source_table_name=settings.table("reviews_chunked"),
        primary_key="chunk_id",
        pipeline_type=cfg["pipeline_type"],
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=cfg["embedding_endpoint"],
        columns_to_sync=REVIEW_COLUMNS,
    )
index.wait_until_ready(verbose=True, wait_for_updates=True)
print(index.describe()["status"])

# COMMAND ----------

# MAGIC %md ## 3. Test queries

# COMMAND ----------


def show(query, **kw):
    resp = index.similarity_search(query_text=query, columns=REVIEW_COLUMNS, num_results=3, **kw)
    print(f"\n=== {query} {kw or ''}")
    for d in parse_search_response(resp):
        m = d["metadata"]
        print(f"  [{m['score']:.3f}] review {m['review_id']} franchise {m['franchise_id']}: {d['page_content'][:120]}")


show("the cookies were stale")
show("friendly staff and quick service")
show("too expensive for what you get")

# COMMAND ----------

# Metadata filter on franchise_id, needed for the hybrid questions.
top_franchise = (
    spark.table(settings.table("reviews_chunked")).groupBy("franchise_id").count().orderBy("count", ascending=False).first()
)
show("what do customers think", filters={"franchise_id": int(top_franchise["franchise_id"])})
