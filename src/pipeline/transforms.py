"""PySpark transformations from ``samples.bakehouse`` to the silver tables.

Every function takes and returns a DataFrame, so each step can be run and
inspected on its own in ``notebooks/01_ingest.py``. Personal data that the
agent never needs (names, e-mail addresses, phone numbers, street addresses
and card numbers) is dropped at this stage, so it can never reach a prompt.
"""

from __future__ import annotations

from pyspark import cloudpickle
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pipeline import chunking
from pipeline.chunking import chunk_text, clean_text

# UDFs run on Spark workers, which cannot import modules from this repo.
# Pickling the chunking module by value ships its code with each UDF.
cloudpickle.register_pickle_by_value(chunking)


def col_like(df: DataFrame, *candidates: str) -> str:
    """Return the actual column name matching any candidate, case-insensitively.

    The sample tables use camelCase (``franchiseID``); this keeps the pipeline
    working if a future version of the sample renames a column's case.
    """
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    raise ValueError(f"None of {candidates} found. Available columns: {df.columns}")


def _trimmed(df: DataFrame, *candidates: str):
    return F.trim(F.col(col_like(df, *candidates)).cast("string"))


def transform_transactions(df: DataFrame) -> DataFrame:
    ts = F.to_timestamp(F.col(col_like(df, "dateTime", "date_time", "transaction_ts")))
    qty = F.col(col_like(df, "quantity")).cast("int")
    unit_price = F.col(col_like(df, "unitPrice", "unit_price")).cast("decimal(10,2)")
    total = F.col(col_like(df, "totalPrice", "total_price")).cast("decimal(12,2)")
    out = (
        df.select(
            F.col(col_like(df, "transactionID", "transaction_id")).cast("bigint").alias("transaction_id"),
            F.col(col_like(df, "customerID", "customer_id")).cast("bigint").alias("customer_id"),
            F.col(col_like(df, "franchiseID", "franchise_id")).cast("bigint").alias("franchise_id"),
            ts.alias("transaction_ts"),
            _trimmed(df, "product").alias("product"),
            qty.alias("quantity"),
            unit_price.alias("unit_price"),
            # Revenue is the recorded total; fall back to quantity x unit price.
            F.coalesce(total, (qty * unit_price).cast("decimal(12,2)")).alias("revenue"),
            F.initcap(_trimmed(df, "paymentMethod", "payment_method")).alias("payment_method"),
        )
        .where(F.col("transaction_id").isNotNull() & F.col("transaction_ts").isNotNull())
        .where(F.col("quantity") > 0)
        .dropDuplicates(["transaction_id"])
        .withColumn("transaction_date", F.to_date("transaction_ts"))
        .withColumn("transaction_hour", F.hour("transaction_ts"))
        .withColumn("day_of_week", F.date_format("transaction_ts", "EEEE"))
    )
    return out.select(
        "transaction_id",
        "customer_id",
        "franchise_id",
        "transaction_ts",
        "transaction_date",
        "transaction_hour",
        "day_of_week",
        "product",
        "quantity",
        "unit_price",
        "revenue",
        "payment_method",
    )


def transform_customers(df: DataFrame) -> DataFrame:
    return (
        df.select(
            F.col(col_like(df, "customerID", "customer_id")).cast("bigint").alias("customer_id"),
            _trimmed(df, "city").alias("city"),
            _trimmed(df, "state").alias("state"),
            _trimmed(df, "country").alias("country"),
            _trimmed(df, "continent").alias("continent"),
            F.lower(_trimmed(df, "gender")).alias("gender"),
        )
        .where(F.col("customer_id").isNotNull())
        .dropDuplicates(["customer_id"])
    )


def transform_franchises(df: DataFrame) -> DataFrame:
    return (
        df.select(
            F.col(col_like(df, "franchiseID", "franchise_id")).cast("bigint").alias("franchise_id"),
            _trimmed(df, "name", "franchise_name").alias("franchise_name"),
            _trimmed(df, "city").alias("city"),
            _trimmed(df, "district").alias("district"),
            _trimmed(df, "zipcode", "zip_code").alias("zipcode"),
            _trimmed(df, "country").alias("country"),
            F.lower(_trimmed(df, "size")).alias("size"),
            F.col(col_like(df, "supplierID", "supplier_id")).cast("bigint").alias("supplier_id"),
            F.col(col_like(df, "latitude")).cast("double").alias("latitude"),
            F.col(col_like(df, "longitude")).cast("double").alias("longitude"),
        )
        .where(F.col("franchise_id").isNotNull())
        .dropDuplicates(["franchise_id"])
    )


def transform_suppliers(df: DataFrame) -> DataFrame:
    approved_raw = F.lower(_trimmed(df, "approved"))
    return (
        df.select(
            F.col(col_like(df, "supplierID", "supplier_id")).cast("bigint").alias("supplier_id"),
            _trimmed(df, "name", "supplier_name").alias("supplier_name"),
            _trimmed(df, "ingredient").alias("ingredient"),
            _trimmed(df, "continent").alias("continent"),
            _trimmed(df, "city").alias("city"),
            _trimmed(df, "district").alias("district"),
            F.lower(_trimmed(df, "size")).alias("size"),
            approved_raw.isin("y", "yes", "true", "1").alias("approved"),
        )
        .where(F.col("supplier_id").isNotNull())
        .dropDuplicates(["supplier_id"])
    )


_clean_udf = F.udf(clean_text, T.StringType())


def transform_reviews(df: DataFrame) -> DataFrame:
    """Clean review text and assign a stable integer ``review_id``.

    The sample's ``new_id`` column is used when present; otherwise ids come
    from a deterministic ordering so they are stable across reruns.
    """
    text = _clean_udf(F.col(col_like(df, "review", "review_text")))
    base = (
        df.select(
            F.col(col_like(df, "franchiseID", "franchise_id")).cast("bigint").alias("franchise_id"),
            F.to_date(F.col(col_like(df, "review_date"))).alias("review_date"),
            text.alias("review_text"),
            *(
                [F.col(col_like(df, "new_id")).alias("_source_id")]
                if "new_id" in [c.lower() for c in df.columns]
                else []
            ),
        )
        .where(F.length("review_text") > 0)
        .dropDuplicates(["franchise_id", "review_date", "review_text"])
    )

    if "_source_id" in base.columns:
        base = base.withColumn("review_id", F.col("_source_id").cast("bigint")).drop("_source_id")
    else:
        w = Window.orderBy("franchise_id", "review_date", "review_text")
        base = base.withColumn("review_id", F.row_number().over(w).cast("bigint"))
    return base.select("review_id", "franchise_id", "review_date", "review_text")


def chunk_reviews(reviews: DataFrame, max_words: int, overlap_sentences: int) -> DataFrame:
    """One row per chunk: ``chunk_id``, ``review_id``, ``franchise_id``, ``review_date``, ``chunk_text``."""
    chunker = F.udf(
        lambda t: chunk_text(t, max_words=max_words, overlap_sentences=overlap_sentences),
        T.ArrayType(T.StringType()),
    )
    return (
        reviews.select(
            "review_id",
            "franchise_id",
            "review_date",
            F.posexplode(chunker(F.col("review_text"))).alias("chunk_no", "chunk_text"),
        )
        .withColumn(
            "chunk_id", F.concat_ws("-", F.col("review_id").cast("string"), F.col("chunk_no").cast("string"))
        )
        .select("chunk_id", "review_id", "franchise_id", "review_date", "chunk_text")
    )


# Table and column comments. They are read by the Text2SQL prompt (via
# information_schema) and by Genie, so both systems get the same metadata.
TABLE_COMMENTS = {
    "silver_transactions": (
        "One row per sales transaction at a bakery franchise. revenue is the transaction total in USD.",
        {
            "transaction_id": "Unique transaction identifier",
            "customer_id": "Customer who paid; joins to silver_customers.customer_id",
            "franchise_id": "Franchise where the sale happened; joins to silver_franchises.franchise_id",
            "transaction_ts": "Timestamp of the sale",
            "transaction_date": "Calendar date of the sale (derived from transaction_ts)",
            "transaction_hour": "Hour of day of the sale, 0 to 23",
            "day_of_week": "Weekday name of the sale, for example Monday",
            "product": "Name of the product sold, for example Golden Gate Ginger",
            "quantity": "Number of units sold",
            "unit_price": "Price per unit in USD",
            "revenue": "Total value of the transaction in USD",
            "payment_method": "Payment method, for example Visa, Mastercard or Amex",
        },
    ),
    "silver_customers": (
        "One row per customer. Personal contact details were removed during ingestion.",
        {
            "customer_id": "Unique customer identifier",
            "city": "Customer city",
            "state": "Customer state or region",
            "country": "Customer country",
            "continent": "Customer continent",
            "gender": "Customer gender as recorded",
        },
    ),
    "silver_franchises": (
        "One row per bakery franchise (shop).",
        {
            "franchise_id": "Unique franchise identifier",
            "franchise_name": "Trading name of the franchise",
            "city": "City where the franchise is located",
            "district": "District within the city",
            "zipcode": "Postal code",
            "country": "Country where the franchise is located",
            "size": "Franchise size category, for example s, m or l",
            "supplier_id": "Main supplier; joins to silver_suppliers.supplier_id",
            "latitude": "Latitude of the franchise",
            "longitude": "Longitude of the franchise",
        },
    ),
    "silver_suppliers": (
        "One row per ingredient supplier.",
        {
            "supplier_id": "Unique supplier identifier",
            "supplier_name": "Supplier name",
            "ingredient": "Main ingredient supplied",
            "continent": "Supplier continent",
            "city": "Supplier city",
            "district": "Supplier district",
            "size": "Supplier size category",
            "approved": "True if the supplier is approved",
        },
    ),
    "silver_reviews": (
        "One row per customer review of a franchise (cleaned text).",
        {
            "review_id": "Unique review identifier, cited by the agent",
            "franchise_id": "Reviewed franchise; joins to silver_franchises.franchise_id",
            "review_date": "Date of the review",
            "review_text": "Full review text",
        },
    ),
    "reviews_chunked": (
        "Review text split into sentence-window chunks; source of the AI Search index.",
        {
            "chunk_id": "Chunk identifier: <review_id>-<chunk number>",
            "review_id": "Review the chunk came from",
            "franchise_id": "Reviewed franchise",
            "review_date": "Date of the review",
            "chunk_text": "Chunk text that is embedded for search",
        },
    ),
}
