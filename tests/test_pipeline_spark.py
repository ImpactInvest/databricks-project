"""Local Spark checks for the ingestion transforms and the golden-set SQL.

These run only when PySpark and Java are available (they are skipped in CI).
They use a tiny synthetic dataset shaped like ``samples.bakehouse``; nothing
here produces a reported result. The point is to prove that every transform
and every reference query is valid Spark SQL before running on Databricks.
"""

from __future__ import annotations

import datetime as dt
import shutil

import pytest

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java is not installed", allow_module_level=True)

from pyspark.sql import SparkSession  # noqa: E402

from agent.config import load_settings  # noqa: E402
from agent.sql_guard import check_sql  # noqa: E402
from evals.golden_set import SPECS, build_sql_ground_truth  # noqa: E402
from pipeline import transforms as tr  # noqa: E402


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    warehouse = tmp_path_factory.mktemp("warehouse")
    try:
        session = (
            SparkSession.builder.master("local[1]")
            .appName("tests")
            .config("spark.sql.warehouse.dir", str(warehouse))
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
    except Exception as e:  # for example the Spark Connect-only client from databricks-connect
        pytest.skip(f"Local Spark is not available: {e}")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def settings():
    return load_settings(overrides={"catalog": "spark_catalog", "schema": "genai_agent"})


def _source_frames(spark):
    products = ["Golden Gate Ginger", "Outback Oatmeal", "Tokyo Tidbits"]
    tx = []
    for i in range(1, 61):
        tx.append(
            (
                i,
                100 + i % 7,
                3000 + i % 4,
                dt.datetime(2024, 5, 1 + i % 5, 8 + i % 9, 0, 0),
                products[i % 3],
                1 + i % 4,
                3,
                3 * (1 + i % 4),
                ["visa", "amex", "mastercard"][i % 3],
                4000_0000_0000_0000 + i,
            )
        )
    tx.append(tx[0])  # a duplicate to be removed
    transactions = spark.createDataFrame(
        tx,
        "transactionID long, customerID long, franchiseID long, dateTime timestamp, product string, "
        "quantity long, unitPrice long, totalPrice long, paymentMethod string, cardNumber long",
    )
    customers = spark.createDataFrame(
        [
            (
                100 + i,
                f"F{i}",
                f"L{i}",
                f"e{i}@x.com",
                "555",
                "1 St",
                "Tokyo",
                "TK",
                "Japan",
                "Asia",
                "100",
                "Male",
            )
            for i in range(7)
        ],
        "customerID long, first_name string, last_name string, email_address string, phone_number string, "
        "address string, city string, state string, country string, continent string, postal_zip_code string, gender string",
    )
    franchises = spark.createDataFrame(
        [
            (
                3000 + i,
                f"Shop {i}",
                "Sydney",
                "CBD",
                "2000",
                ["Australia", "Japan"][i % 2],
                "m",
                10 + i % 2,
                1.0,
                2.0,
            )
            for i in range(4)
        ],
        "franchiseID long, name string, city string, district string, zipcode string, country string, size string, "
        "supplierID long, longitude double, latitude double",
    )
    suppliers = spark.createDataFrame(
        [
            (10, "Mill Co", "Flour", "Oceania", "Perth", "X", "l", 1.0, 2.0, "Y"),
            (11, "Sugar Co", "Sugar", "Asia", "Osaka", "Y", "s", 1.0, 2.0, "N"),
        ],
        "supplierID long, name string, ingredient string, continent string, city string, district string, "
        "size string, longitude double, latitude double, approved string",
    )
    reviews = spark.createDataFrame(
        [
            (
                f"Great cookies at shop {i}.  Staff were slow though! Would return.",
                3000 + i % 4,
                dt.date(2024, 5, 1 + i % 5),
                900 + i,
            )
            for i in range(8)
        ],
        "review string, franchiseID long, review_date date, new_id long",
    )
    return transactions, customers, franchises, suppliers, reviews


@pytest.fixture(scope="module")
def silver(spark, settings):
    transactions, customers, franchises, suppliers, reviews = _source_frames(spark)
    spark.sql("CREATE DATABASE IF NOT EXISTS genai_agent")
    out = {
        "silver_transactions": tr.transform_transactions(transactions),
        "silver_customers": tr.transform_customers(customers),
        "silver_franchises": tr.transform_franchises(franchises),
        "silver_suppliers": tr.transform_suppliers(suppliers),
        "silver_reviews": tr.transform_reviews(reviews),
    }
    out["reviews_chunked"] = tr.chunk_reviews(out["silver_reviews"], max_words=6, overlap_sentences=1)
    for name, df in out.items():
        df.write.mode("overwrite").saveAsTable(f"genai_agent.{name}")
    return out


def test_transforms_shape(silver):
    tx = silver["silver_transactions"]
    assert tx.count() == 60  # duplicate removed
    assert "card_number" not in tx.columns and "cardNumber" not in tx.columns
    assert set(tx.columns) >= {"transaction_date", "transaction_hour", "day_of_week", "revenue"}
    assert "email_address" not in silver["silver_customers"].columns
    assert silver["silver_suppliers"].where("approved").count() == 1
    assert silver["silver_reviews"].select("review_id").distinct().count() == 8


def test_chunking_produces_unique_ids(silver):
    chunks = silver["reviews_chunked"]
    assert chunks.count() > silver["silver_reviews"].count()
    assert chunks.select("chunk_id").distinct().count() == chunks.count()


def test_every_reference_query_runs_and_passes_the_guard(spark, settings, silver):
    def run_query(sql):
        df = spark.sql(sql)
        return df.columns, [list(r) for r in df.collect()]

    for spec in SPECS:
        gt = build_sql_ground_truth(spec, settings, run_query)
        ref = gt["expectations"]["reference_sql"]
        if ref:
            guard = check_sql(ref, settings.silver_tables, settings.catalog, settings.schema)
            assert guard.ok, f"{spec['id']}: {guard.error}"
            assert gt["expectations"]["expected_result"]["rows"], spec["id"]
        assert gt["expectations"]["expected_facts"] or spec.get("review_query"), spec["id"]
        assert "{" not in gt["question"], spec["id"]
