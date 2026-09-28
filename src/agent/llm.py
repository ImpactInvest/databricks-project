"""OpenAI-compatible client for Databricks Foundation Model API endpoints.

Databricks serving endpoints speak the OpenAI chat-completions protocol at
``<workspace>/serving-endpoints``. Authentication comes from the Databricks
SDK and is resolved on every request, so OAuth tokens that expire during a
long evaluation or DSPy run are refreshed automatically. In Model Serving
the SDK picks up the credentials injected for the declared resources.

We build the client here rather than depending on ``databricks-openai``:
that package pulls in ``databricks-connect``, which breaks the Spark session
when installed into a serverless notebook.
"""

from __future__ import annotations

import httpx
from openai import OpenAI


class _DatabricksAuth(httpx.Auth):
    def __init__(self, workspace_client):
        self.w = workspace_client

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = self.w.config.authenticate()["Authorization"]
        yield request


def make_llm_client(workspace_client=None, timeout: float = 120.0) -> OpenAI:
    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()
    return OpenAI(
        base_url=f"{workspace_client.config.host.rstrip('/')}/serving-endpoints",
        api_key="unused",  # the Authorization header is set by _DatabricksAuth
        http_client=httpx.Client(auth=_DatabricksAuth(workspace_client), timeout=timeout),
        max_retries=3,
    )
