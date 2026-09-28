# Design decisions and trade-offs

This document explains why the project is built the way it is. It is written to be defended in an interview: each section states the decision, the reason, and what would change in production.

## 0. Versions and API checks (September 2026)

Package versions were checked against PyPI and against the installed package source code (the Databricks and MLflow documentation sites were not reachable from the build environment). Verified points:

| Topic | What was checked | Result |
|---|---|---|
| AI Search (formerly Vector Search) | PyPI and package source | The canonical package is now `databricks-ai-search` (0.78) with `from databricks.ai_search.client import AISearchClient`. `databricks-vectorsearch` (0.75) is a deprecation shim that re-exports it. |
| Genie Agent (formerly Genie Space) | `databricks-sdk` 0.143.0 | The API still uses `w.genie.start_conversation_and_wait(space_id, content)` and `get_message_attachment_query_result`. A `create_space` API exists but needs an undocumented serialised payload, so the space is created in the UI. |
| Agent interface | MLflow 3.16.1 | `mlflow.pyfunc.ResponsesAgent` is the recommended interface; `predict` is traced automatically as an AGENT span. |
| Evaluation | MLflow 3.16.1 | `mlflow.genai.evaluate` with `Correctness`, `RetrievalGroundedness`, `RelevanceToQuery`, `Safety`, `Guidelines` and `@scorer`. `RetrievalGroundedness` raises on traces without a RETRIEVER span (handled, see section 4). |
| Trace storage | MLflow 3.16.1 | Experiments can be bound to a Unity Catalog trace location (`UnityCatalog(catalog_name, schema_name)`), which needs a SQL warehouse. |
| Model resources | MLflow 3.16.1 | `DatabricksServingEndpoint`, `DatabricksVectorSearchIndex`, `DatabricksSQLWarehouse`, `DatabricksTable` exist with the argument names used in notebook 04. |
| Deployment | `databricks-agents` 1.12.0 | `agents.deploy(model_name, model_version, scale_to_zero=..., endpoint_name=...)`. |
| DSPy | 3.4.0 | `dspy.LM(..., engine="litellm")`, `dspy.MIPROv2(auto="light")`, `dspy.Evaluate` returns a result with `.score` as a percentage. |
| Bundles | `databricks-bundles` 1.18.0 | The job and app definitions in `databricks.yml` load into the typed `Job` and `App` classes, which reject unknown fields. `databricks bundle validate` still needs to be run against a workspace. |

Two problems were found and designed around during the build:

* **`databricks-openai` breaks serverless notebooks.** It depends on `unitycatalog-ai[databricks]`, which installs `databricks-connect`, and that replaces the notebook's Spark client. The agent therefore builds its own OpenAI-compatible client in `src/agent/llm.py` (about 20 lines) using the Databricks SDK for authentication, refreshing the token on every request.
* **Spark workers cannot import repo modules inside UDFs.** The chunking UDF failed with `ModuleNotFoundError` in the local Spark test. `transforms.py` registers the chunking module with `cloudpickle.register_pickle_by_value`, so the code travels with the UDF.

## 1. Why tool calling, and when to use Genie instead

Questions fall into three shapes: numbers from tables, opinions from reviews, and both. A fixed pipeline would either always run both (slow, noisy context) or need a separate router. A tool-calling loop lets the LLM decide, and hybrid questions naturally become two steps: get the franchise from SQL, then search reviews filtered to that franchise.

The loop is written out explicitly in `src/agent/agent.py` rather than hidden in a framework such as LangGraph. With two tools and one loop, plain OpenAI-compatible tool calling is about 60 lines, easy to trace, and has no framework version risk. LangGraph becomes worth it with branching workflows, human-in-the-loop checkpoints or many agents.

**Use Genie instead** when the need is self-service analytics over tables for business users: Genie brings a curated semantic layer, sample questions, trusted assets, benchmarks and a UI with no code to maintain. Build a custom agent when you need to combine tables with unstructured documents, enforce custom guardrails, integrate into another application, or evaluate and optimise the behaviour yourself, which is the situation here. A good production pattern is both: call a Genie Agent as one of the agent's tools.

## 2. Text2SQL safety

Text2SQL is its own tool (`query_sales_data(question)`), so the agent LLM never writes SQL directly; a dedicated prompt does, and every query passes `src/agent/sql_guard.py` before execution:

* parsed with `sqlglot` (Databricks dialect); exactly one statement;
* the root must be a SELECT (CTEs and UNION allowed); any DDL, DML, `SET`, `USE`, `SHOW`, `SELECT INTO` or command node anywhere in the tree is rejected;
* table-valued functions (`read_files`) and a denylist of functions (`reflect`, `java_method`, `http_request`, `ai_query`, `current_user` and others) are rejected;
* every table must be on the allowlist (the four `silver_*` sales tables) and is rewritten to its fully qualified name, so a query cannot reach another schema or catalog;
* a LIMIT is added (200) or capped (1000); comments are stripped;
* execution has a timeout (Statement Execution API `wait_timeout` with `CANCEL`, or Spark Connect interrupt tags in notebooks);
* failures return structured errors (`guard_rejected`, `execution_error`, `timeout`) and the tool retries once with the error fed back, then gives up.

Personal data (names, e-mail addresses, phone numbers, street addresses, card numbers) is dropped at ingestion, so it cannot be queried at all. The guard is covered by more than 30 unit tests, including injection attempts.

**Why read-only with an allowlist?** The guard is defence in depth, not the main control. The main control in production is identity: the served agent's credentials should only have `SELECT` on the allowed tables. Text checks catch mistakes and obvious attacks early and give the LLM a useful error message.

**What production would add:** row filters and column masks in Unity Catalog, on-behalf-of-user authentication so each user sees only their rows, query cost limits and a dedicated small warehouse, audit via system tables, and rate limits in AI Gateway.

## 3. Retrieval

* **Chunking:** sentence windows of up to 120 words with a one-sentence overlap (`src/pipeline/chunking.py`). Bakehouse reviews are short, so most reviews are one chunk; sentence boundaries keep citations readable. Chunk ids are `<review_id>-<n>`, so every chunk maps back to a review.
* **Index:** a Delta Sync index with managed embeddings (`databricks-gte-large-en`), triggered sync, on `reviews_chunked` with Change Data Feed on. Free Edition does not support Direct Vector Access and allows one endpoint, so this is also the only option.
* **Top-k:** 5 by default, and the agent may ask for 1 to 10. Short reviews mean 5 chunks is a small, focused context.
* **Metadata filter:** `franchise_id`, used by hybrid questions so reviews come from the franchise the SQL identified.
* **Citations:** the prompt requires `[review <id>]` for every review claim. The agent then keeps only cited ids that were actually retrieved (`cited_review_ids`), a `Guidelines` judge checks that review claims carry citations, and `RetrievalGroundedness` checks that the answer is supported by the retrieved text.

## 4. Evaluation design

* **Computed ground truth.** Each SQL or hybrid question has a `reference_sql`. The builder runs it and stores both the result (`expected_result`, for the deterministic `sql_result_match` scorer) and facts rendered from it (`expected_facts`, for the Correctness judge). Questions that mention a date, product or franchise pick it from the data first with `params_sql`, so no question depends on a value typed by hand. All 58 reference queries were run on a synthetic bakehouse-shaped dataset in local Spark and pass the SQL guard.
* **Review facts** are extracted from the retrieved reviews by the judge LLM and flagged `llm_extracted` for a human spot-check. This is the weakest part of the ground truth and is labelled as such.
* **Mix:** 58 questions: 31 SQL, 17 review, 10 hybrid; including ties, date filters, joins, ambiguous wording, 3 no-answer questions and 3 unsafe or prompt-injection requests.
* **Held-out split.** A deterministic, stratified 30-question train split for DSPy and a 28-question test split for every reported number. Unsafe and no-answer questions are always in test.
* **`sql_result_match`** compares result sets after normalisation (rounding, case, date formats), ignoring row order, column order, column names and extra columns, but requiring the same number of rows and the same values row by row. It reads the SQL results from the `run_sql` spans in the trace, so it scores what the agent actually retrieved, not what it claimed.
* **Groundedness** uses the built-in `RetrievalGroundedness` judge, wrapped so that it returns "not applicable" on traces with no RETRIEVER span (pure SQL questions) instead of an error. The judge gives one verdict per retrieval step; they are combined into one, and an answer counts as grounded only if every step supports it.
* **Pass definition:** Correctness yes AND Safety yes AND, where a reference SQL exists, `sql_result_match` true.
* **Judge reliability.** Notebook 10 samples 10 judge verdicts for a person to label, and reports agreement only once they are labelled. Not yet done (no runs yet).
* **Same conditions for every variant.** Baseline and optimised runs use the same notebook, test rows, scorers and judge model, and run SQL on the SQL warehouse as the served agent does.

## 5. DSPy

* **What is optimised:** the Text2SQL step only, with the signature `question, schema_context -> sql`, starting from the baseline instructions. Using `dspy.Predict` (not chain of thought) means the exported demos are plain question and SQL pairs that the agent's own prompt template can reuse.
* **Metric:** execution accuracy, through the same guard, executor and `results_match` used in evaluation.
* **Data:** the SQL-bearing questions in the train split (21), divided into a DSPy train set and a DSPy validation set. The test split is never shown to DSPy.
* **Optimiser:** `MIPROv2(auto="light", max_bootstrapped_demos=3, max_labeled_demos=3)`, chosen as the cheapest MIPRO setting that still searches instructions and demos. The notebook records calls, tokens and wall-clock time.
* **Deployment of the result:** the instructions and demos are written to `src/agent/optimised_text2sql.json` (committed), registered in the MLflow Prompt Registry, and the agent is re-logged with `prompt_variant: optimised`. The agent never runs DSPy at serving time.
* **Result:** not yet run. If DSPy does not help, the likely reasons are a small training set, a strong baseline model, and schema comments that already carry most of the signal; that will be reported as it is.

## 6. Serving

* **Models-from-code:** `src/agent/serving_entry.py` plus the `agent` package are logged as code, not a pickle, so the served code is exactly the code in Git. Checked locally: the logged model loads from its own artefacts with the logged configuration.
* **model_config:** the resolved settings (endpoints, warehouse ID, schema summary, prompt variant) travel with the model, so the served agent uses exactly what was evaluated.
* **Declared resources and auth passthrough:** the LLM and embedding endpoints, the AI Search index, the SQL warehouse and the tables are declared at logging time, so Model Serving issues short-lived credentials scoped to them. The AI Search client detects the serving environment itself.
* **No Spark inside serving:** SQL goes through the Statement Execution API on the SQL warehouse.
* **Free Edition:** CPU serving only and a limited number of endpoints; `scale_to_zero` is on. Notebook 05 stops with a clear message if agent deployment is refused, rather than changing the architecture silently.
* **Tracing:** `ResponsesAgent.predict` is an AGENT span, the OpenAI client is autologged, and tools add TOOL, LLM and RETRIEVER spans. By default traces are stored in Unity Catalog Delta tables in the project schema (`mlflow.trace_location: uc`); set it to `experiment` for classic storage.

## 7. What changes at enterprise scale

* **Pipelines:** Lakeflow Declarative Pipelines for ingestion, with expectations as data-quality rules, instead of notebooks.
* **Monitoring:** the same scorers registered as production monitors on sampled live traces, with alerts on drops in groundedness or SQL errors; the app's thumbs up and down feed a labelled review queue.
* **CI/CD:** bundles with dev, staging and prod targets; the unit tests on every push; the evaluation job as a deployment gate (a new version must not regress pass rate on the golden set); service principals for jobs.
* **Governance:** Unity Catalog grants, row filters and column masks, lineage for tables, models and prompts, on-behalf-of-user auth, AI Gateway for rate limits, usage tracking and guardrails.
* **Cost:** a smaller model for the Text2SQL step once DSPy has closed the gap, caching of frequent questions, and budget policies on endpoints.

## Genie setup

Create the Genie Agent with the same tables and the same metadata the agent sees, so the comparison is fair:

1. In the workspace sidebar, open **Genie** and click **New**.
2. Add the tables `workspace.genai_agent.silver_transactions`, `silver_franchises`, `silver_customers` and `silver_suppliers`. The table and column comments added by `01_ingest.py` come with them.
3. Choose the SQL warehouse (the Free Edition serverless warehouse).
4. Title: `Bakehouse sales`. In the instructions box, paste the text of `TEXT2SQL_BASELINE_INSTRUCTIONS` from `src/agent/prompts.py` (the same guidance the baseline agent has). Do not add example SQL; the baseline agent has none.
5. Save, then copy the space ID from the URL (`.../genie/rooms/<space_id>`) and paste it into the widget in `09_genie_benchmark.py`.

Genie latency is wall-clock time for `start_conversation_and_wait`, which includes SDK polling; the agent's latency is the trace duration. The comparison notes this.

## Free Edition limits and workarounds

| Limit | Workaround |
|---|---|
| One AI Search endpoint, one search unit | One Delta Sync index on one endpoint; if another endpoint already exists, notebook 02 stops and names it so it can be reused. |
| No Direct Vector Access | Delta Sync index with managed embeddings. |
| Limited serving endpoints, CPU only | One agent endpoint with scale to zero; the LLM is a pay-per-token Foundation Model API endpoint. |
| Apps stop after 24 hours | `databricks bundle run bakehouse_app` restarts the app; documented in the README. |
| Serverless only | All jobs and notebooks run on serverless; `%pip install` per notebook; no `databricks-connect`. |
