"""Models-from-code entry point used by ``mlflow.pyfunc.log_model``.

MLflow executes this file when the model is loaded (in a notebook or in Model
Serving). The configuration comes from the ``model_config`` passed at logging
time, so the served agent uses exactly the settings the notebooks evaluated.
When the file is run directly in the repo (development), the local settings
files are used instead.
"""

import mlflow
from mlflow.models import ModelConfig

from agent.agent import BakehouseAgent
from agent.config import load_settings

mlflow.openai.autolog()

try:
    _development_config = load_settings().to_dict()
except FileNotFoundError:  # inside Model Serving the repo's config folder is absent
    _development_config = None

config = ModelConfig(development_config=_development_config).to_dict()
AGENT = BakehouseAgent(config)
mlflow.models.set_model(AGENT)
