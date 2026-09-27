"""MLflow setup shared by every notebook.

* Tracking goes to the Databricks workspace; models register in Unity Catalog.
* By default the experiment is bound to a Unity Catalog trace location, so
  traces are stored as Delta tables in the project schema (MLflow 3.11+,
  needs a SQL warehouse). Set ``mlflow.trace_location: experiment`` in the
  settings to use classic experiment storage instead; this is an explicit
  choice, never a silent fallback.
"""

from __future__ import annotations

import os

import mlflow

from agent.config import Settings


def setup_mlflow(settings: Settings) -> str:
    """Configure tracking and return the experiment ID."""
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    cfg = settings["mlflow"]
    name = cfg["experiment_name"]

    if cfg.get("trace_location", "uc") == "uc":
        from mlflow.entities.trace_location import UnityCatalog

        warehouse_id = settings["sql"]["warehouse_id"]
        if not warehouse_id:
            raise ValueError("sql.warehouse_id is required for UC trace storage; run 00_setup first.")
        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id
        experiment = mlflow.set_experiment(
            experiment_name=name,
            trace_location=UnityCatalog(catalog_name=settings.catalog, schema_name=settings.schema),
        )
    else:
        experiment = mlflow.set_experiment(experiment_name=name)
    return experiment.experiment_id
