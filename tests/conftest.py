"""Shared test setup: every test starts with clean MLflow global state.

Several tests point MLflow at their own temporary SQLite store and set an
active experiment. Without a reset, a later test would log traces to the previous
test's store, or to an experiment ID that does not exist in its own store.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def mlflow_store_outside_the_repo(tmp_path_factory):
    """Traces from tests without their own store go to a temporary SQLite file.

    The working directory moves too, because local MLflow stores put artefacts
    under ``./mlruns``. Tests locate repo files from ``__file__``, not the cwd.
    """
    mp = pytest.MonkeyPatch()
    workdir = tmp_path_factory.mktemp("mlflow")
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{workdir}/mlflow.db")
    mp.chdir(workdir)
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def reset_mlflow_state():
    yield
    try:
        import mlflow
        from mlflow.tracing.provider import reset as reset_tracer
        from mlflow.tracking import fluent
    except ImportError:
        return
    mlflow.flush_trace_async_logging()
    fluent._active_experiment_id = None
    mlflow.set_tracking_uri(None)
    reset_tracer()  # the trace exporter caches the store it was created for
