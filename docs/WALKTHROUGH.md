# 20-minute walkthrough

A script for presenting the project. Timings are a guide. Numbers in square brackets come from `notebooks/10_results_report.py` after the runs; do not quote a number that is not in that report.

## 1. The problem (2 minutes)

"A bakery franchise network wants to ask plain-English questions that need both structured data and customer opinion, for example: *which franchise had the highest revenue last month, and what are customers complaining about there?* A BI dashboard answers the first half, a search box the second half; nothing answers both, and an LLM on its own will make numbers up. I wanted an agent that uses the right data for each part, shows its working, and that I can measure and improve."

## 2. Architecture (3 minutes)

Show `docs/images/architecture.png`. Walk left to right:

* **Data:** `samples.bakehouse`, ingested with PySpark into silver Delta tables in Unity Catalog. Personal data is dropped at this point. Table and column comments are added because both my Text2SQL prompt and Genie read them.
* **Retrieval:** reviews are chunked into sentence windows and indexed in AI Search with a Delta Sync index and managed embeddings.
* **Agent:** an MLflow `ResponsesAgent` with two tools. `query_sales_data` is Text2SQL with a strict guard; `search_reviews` is semantic search with a franchise filter.
* **Serving:** logged as code with its configuration and declared resources, registered in Unity Catalog, deployed to Model Serving, with a Databricks App in front.
* **Quality loop:** MLflow tracing on every call, a golden set with computed ground truth, LLM judges plus deterministic scorers, DSPy to optimise the Text2SQL prompt, and Genie as a benchmark.

## 3. Live demo (5 minutes)

In the app, ask four questions. For each, open the SQL and review panels and then the MLflow trace.

1. **SQL:** "Which product generated the most revenue?" Point out the generated SQL, the LIMIT the guard added, and the result table.
2. **Review:** "What do customers complain about most often?" Point out the review ids in the answer and the RETRIEVER span in the trace.
3. **Hybrid:** "Which franchise had the highest revenue, and what do customers say about it?" Show two tool calls, with the `franchise_id` from the SQL result passed as the search filter.
4. **Unsafe:** "Drop the sales transactions table." The agent refuses without calling a tool. Mention that even if it tried, the guard rejects anything but a single SELECT on four allowlisted tables, and in production the endpoint's credentials would only have SELECT.

## 4. Evaluation (4 minutes)

* The golden set has 58 questions: SQL, review and hybrid, with ties, date filters, ambiguous wording, questions with no answer and prompt-injection attempts.
* "Ground truth is computed, not typed": each SQL question has a reference query, and expected values come from running it. Questions that mention a date or product pick it from the data first.
* Scorers: built-in judges for Correctness, Groundedness, Relevance and Safety; guidelines for citations and refusals; and code scorers that read the trace, most importantly `sql_result_match`, which compares the agent's actual result set with the reference result.
* Pass means correct AND safe AND (for SQL questions) the result set matches.
* Results on the held-out test split: pass rate [baseline %], SQL result match [%], groundedness [%], median latency [s]. Mention the five worst failures and what they have in common [from notebook 07].

## 5. DSPy (3 minutes)

* "The Text2SQL step is its own prompt, so I could optimise it in isolation. DSPy's MIPROv2 searched instructions and few-shot examples on the 30-question train split, scored by execution accuracy through the same guard and executor as production."
* "The optimised instructions and examples are exported into the agent's prompt; DSPy is not in the serving path."
* Result: SQL result match went from [A%] to [B%], pass rate from [X%] to [Y%], latency [change], optimisation cost [calls and minutes]. If there was no gain, say why (see DESIGN.md section 5).

## 6. Genie comparison (2 minutes)

* "Genie is the product answer to Text2SQL on Databricks, so it is the right baseline. It does not search documents, so I compared on the SQL test questions only, with the same table comments and the same instructions text."
* Table: baseline agent, optimised agent, Genie: SQL result match and median latency [from notebook 09].
* "My takeaway: for self-service analytics over tables, I would start with Genie. The custom agent earns its keep when you need documents plus tables, custom guardrails, or to embed the capability in another product; a Genie Agent can also be one of its tools."

## 7. What I learned (1 minute)

* Evaluation design is most of the work: computed ground truth and deterministic scorers made the LLM judges much easier to trust.
* Package details matter on serverless: `databricks-openai` pulls in `databricks-connect`, which breaks the notebook's Spark session; and UDF code must be shipped by value.
* For a customer I would add production monitoring with the same scorers, a human review queue from the app's feedback, and Unity Catalog row filters with on-behalf-of-user auth.

## Likely interviewer questions

1. **Why not let the agent LLM write SQL directly in the tool call?**
   A dedicated Text2SQL prompt keeps SQL generation separate from conversation management. It can be optimised with DSPy in isolation, swapped for a cheaper model, or replaced by Genie, without touching the agent loop.

2. **How do you stop SQL injection or destructive queries?**
   Layers: the prompt says read-only; `sqlglot` parses and rejects anything but a single SELECT, blocks dangerous functions and non-allowlisted tables, and forces a LIMIT; execution has a timeout; personal data is not in the silver tables; and in production the credentials only have SELECT, with row filters in Unity Catalog. The guard has over 30 unit tests.

3. **How do you know the LLM judges are right?**
   The key metric, `sql_result_match`, is deterministic. For the judges, I sampled 10 verdicts for human labelling and report agreement [%]. Correctness is also anchored to computed expected facts rather than the judge's own knowledge.

4. **Why a held-out test split with so few questions?**
   Without it, DSPy's gains would be measured on the data it was tuned on. The split is small, so I treat one or two questions of difference as noise and look at per-question changes, not just the headline rate.

5. **What does `sql_result_match` actually compare?**
   The result sets, after normalising numbers, case and dates. It ignores row order, column order, names and extra columns, but needs the same rows with the same values. It reads the rows from the trace, so it measures what the agent actually retrieved.

6. **How does auth work when the agent is served?**
   The model is logged with its resources declared: LLM endpoint, embedding endpoint, AI Search index, SQL warehouse and tables. Model Serving issues short-lived credentials for exactly those. The next step would be on-behalf-of-user auth, so each user's permissions apply.

7. **How are citations enforced?**
   The prompt requires `[review id]`; the agent keeps only ids that were actually retrieved; a Guidelines judge checks review claims are cited; and RetrievalGroundedness checks the answer against the retrieved text.

8. **What did DSPy change, and what did it cost?**
   [From `optimised_text2sql.json` provenance and the MLflow runs: instructions, number of demos, calls, tokens, minutes, and the before and after metrics.]

9. **When would you use Genie instead of this agent?**
   For self-service analytics on tables, where business users benefit from Genie's semantic layer, UI and trusted assets with no code to maintain. Use a custom agent for documents plus tables, custom guardrails, or embedding in an application. Or combine them: a Genie Agent as a tool.

10. **What would you change for a real customer?**
    Lakeflow pipelines for ingestion; production monitors running the same scorers on sampled traffic; a human review queue; CI/CD with bundles and an evaluation gate; Unity Catalog row filters and on-behalf-of-user auth; AI Gateway rate limits and cost tracking.

## Walkthrough call agenda (with Ignác, about 20 minutes)

1. Architecture and data flow (5 minutes), using the diagram and `src/agent/agent.py`.
2. Guardrails and the SQL guard tests (3 minutes).
3. Golden set, scorers and pass definition (5 minutes), using `src/evals/golden_set.py` and `scorers.py`.
4. DSPy and Genie results (4 minutes), from notebooks 08 to 10.
5. Practise three of the interviewer questions above (3 minutes).
