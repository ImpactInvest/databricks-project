"""The agent logs as models-from-code and loads back from its own artefacts.

This is the packaging path notebook 04 and Model Serving use. It runs against a
local MLflow store and makes no Databricks calls. MLflow validates a
ResponsesAgent by calling ``predict`` while logging, so the LLM client is
replaced with a scripted one; that also checks the output schema.
"""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

mlflow = pytest.importorskip("mlflow")

SRC = Path(__file__).resolve().parents[1] / "src"


def test_agent_logs_and_loads_from_its_own_artefacts(tmp_path, monkeypatch):
    from agent import agent as agent_module
    from agent.config import load_settings

    class LLM:
        def __init__(self):
            self.chat = NS(completions=self)
            self.models = []

        def create(self, **kwargs):
            self.models.append(kwargs["model"])
            return NS(choices=[NS(message=NS(content="Hello from the scripted LLM.", tool_calls=None))])

    llm = LLM()
    monkeypatch.setattr(agent_module, "make_llm_client", lambda: llm)
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("packaging-test")

    config = load_settings(
        overrides={
            "llm": {"agent_endpoint": "served-llm"},
            "sql": {"executor": "warehouse", "warehouse_id": "wh-served"},
            "agent": {"schema_context": "TABLE silver_transactions -- sales", "prompt_variant": "baseline"},
        }
    ).to_dict()
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=str(SRC / "agent" / "serving_entry.py"),
            code_paths=[str(SRC / "agent")],
            model_config=config,
            pip_requirements=["mlflow==3.16.1"],
        )

    artefacts = Path(mlflow.artifacts.download_artifacts(info.model_uri))
    assert (artefacts / "code" / "agent" / "tools.py").exists()
    assert llm.models and set(llm.models) == {"served-llm"}  # validation used the logged model_config

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    agent = loaded.get_raw_model()
    assert type(agent).__name__ == "BakehouseAgent"
    # The served agent uses the logged configuration, not the repo's settings files.
    assert agent.settings["llm"]["agent_endpoint"] == "served-llm"
    assert agent.settings["sql"]["warehouse_id"] == "wh-served"
    assert agent.settings["sql"]["executor"] == "warehouse"

    response = loaded.predict({"input": [{"role": "user", "content": "Hi"}]})
    assert response["output"][-1]["content"][0]["text"] == "Hello from the scripted LLM."
    assert response["custom_outputs"]["tool_calls"] == 0
