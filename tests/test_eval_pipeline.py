"""Evaluation plumbing: golden-set building, the evaluation run, summaries and Genie errors."""

import json
import string
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from agent.config import load_settings
from agent.sql_guard import check_sql
from evals import golden_set as gs
from evals.run_eval import headline_metrics, summarise, worst_failures


@pytest.fixture
def settings():
    return load_settings(
        overrides={
            "llm": {"agent_endpoint": "agent-llm", "judge_endpoint": "judge-llm"},
            "agent": {"schema_context": "TABLE silver_transactions -- sales", "max_tool_calls": 4},
            "sql": {"warehouse_id": "wh"},
        }
    )


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        message = reply if isinstance(reply, NS) else NS(content=reply, tool_calls=None)
        return NS(choices=[NS(message=message)])


# --- golden set specs ------------------------------------------------------------------


def _placeholders(template):
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


TABLES = {"transactions", "franchises", "customers", "suppliers", "reviews"}


@pytest.mark.parametrize("spec", gs.SPECS, ids=lambda s: s["id"])
def test_every_spec_template_is_complete(spec):
    """Every placeholder is a table, a param or a column the SQL returns, so building cannot hit a KeyError."""
    assert spec["category"] in {"sql", "review", "hybrid"}
    params_sql = spec.get("params_sql") or ""
    assert _placeholders(params_sql) <= TABLES, spec["id"]
    # params come from the params_sql SELECT list; their names are the output aliases
    params = set()
    if params_sql:
        select_list = params_sql.upper().split("FROM")[0]
        params = {p for p in _placeholders(spec["question"]) if p.upper() in select_list}
        assert _placeholders(spec["question"]) <= params, spec["id"]
    else:
        assert not _placeholders(spec["question"]), spec["id"]
    if spec.get("reference_sql"):
        assert _placeholders(spec["reference_sql"]) <= TABLES | params, spec["id"]
    if spec["category"] in {"review", "hybrid"}:
        assert spec.get("review_query"), spec["id"]
    if spec["category"] == "hybrid":
        assert spec.get("reference_sql") and spec.get("review_filter") == "franchise_id", spec["id"]
        assert "franchise_id" in spec["reference_sql"].split("FROM")[-2], spec["id"]
    if spec["category"] != "review":
        assert spec.get("facts"), spec["id"]


def test_golden_set_has_the_documented_mix():
    cats = [s["category"] for s in gs.SPECS]
    assert (len(gs.SPECS), cats.count("sql"), cats.count("review"), cats.count("hybrid")) == (58, 31, 17, 10)
    no_answer = [s for s in gs.SPECS if "no_answer" in s["tags"]]
    assert len(no_answer) == 3
    records = [{"id": s["id"], "category": s["category"], "tags": s["tags"]} for s in gs.SPECS]
    split = gs.assign_splits(records, train_size=30, seed=17)
    assert sum(r["split"] == "test" for r in split) == 28


def test_sql_literal_escapes_quotes_and_backslashes():
    assert gs.sql_literal("Baker's \\ Delight") == "Baker\\'s \\\\ Delight"
    assert gs.sql_literal(42) == 42


def test_params_with_quotes_produce_valid_guarded_sql(settings):
    spec = next(s for s in gs.SPECS if s["id"] == "q018")
    seen = []

    def run_query(sql):
        seen.append(sql)
        if len(seen) == 1:
            return ["product"], [["Baker's Delight"]]
        return ["units"], [[12]]

    gt = gs.build_sql_ground_truth(spec, settings, run_query)
    assert gt["question"] == "How many units of Baker's Delight were sold in total?"
    assert "product = 'Baker\\'s Delight'" in gt["expectations"]["reference_sql"]
    assert check_sql(
        gt["expectations"]["reference_sql"], settings.silver_tables, "workspace", "genai_agent"
    ).ok
    assert gt["expectations"]["expected_facts"] == ["12 units of Baker's Delight were sold"]
    assert gt["expectations"]["expected_result"] == {"columns": ["units"], "rows": [[12]]}


def test_reference_sql_with_no_data_is_an_error(settings):
    spec = next(s for s in gs.SPECS if s["id"] == "q001")
    with pytest.raises(ValueError, match="no data"):
        gs.build_sql_ground_truth(spec, settings, lambda sql: (["total_revenue"], [[None]]))


def test_match_columns_restrict_the_expected_result(settings):
    spec = next(s for s in gs.SPECS if s["id"] == "q049")
    gt = gs.build_sql_ground_truth(
        spec, settings, lambda sql: (["franchise_id", "franchise_name", "revenue"], [[7, "Shop A", 10.0]])
    )
    assert gt["expectations"]["expected_result"] == {"columns": ["franchise_name"], "rows": [["Shop A"]]}
    assert gt["values"]["franchise_id"] == 7


def test_build_record_for_hybrid_question_filters_reviews_by_franchise(settings):
    spec = next(s for s in gs.SPECS if s["id"] == "q049")
    searches = []

    def search(query, franchise_id=None, k=5):
        searches.append((query, franchise_id, k))
        return [{"review_id": 5, "text": "Long queues."}, {"review_id": 6, "text": "Great bread."}]

    llm = ScriptedLLM(
        ['Sure: {"facts": ["Customers mention long queues"], "supporting_review_ids": [5, "99", "x"]}']
    )
    rec = gs.build_record(
        spec,
        settings,
        lambda sql: (["franchise_id", "franchise_name", "revenue"], [[7, "Shop A", 10.0]]),
        search_reviews=search,
        llm=llm,
        judge_model="judge-llm",
    )
    assert searches == [("what customers think", 7, 8)]
    assert rec["facts_source"] == "sql_plus_llm_extracted"
    assert rec["expectations"]["expected_facts"] == [
        "Shop A had the highest revenue",
        "Customers mention long queues",
    ]
    assert rec["expectations"]["supporting_review_ids"] == [5]  # unknown and non-numeric ids dropped
    assert llm.requests[0]["model"] == "judge-llm"
    assert "[review 5] Long queues." in llm.requests[0]["messages"][0]["content"]


def test_review_question_needs_search_and_llm(settings):
    spec = next(s for s in gs.SPECS if s["id"] == "q032")
    with pytest.raises(ValueError, match="needs search_reviews"):
        gs.build_record(spec, settings, lambda sql: ([], []))


@pytest.mark.parametrize("reply", ["I cannot help with that.", "{not json}", '{"facts": []}'])
def test_unusable_review_facts_raise_a_clear_error(reply):
    with pytest.raises(ValueError, match="usable JSON"):
        gs.extract_review_facts(ScriptedLLM([reply]), "judge", "q?", [{"review_id": 1, "text": "x"}])


def test_jsonl_round_trip_and_eval_rows(tmp_path):
    records = [
        {"id": "a", "split": "test", "inputs": {"question": "q1"}, "expectations": {"expected_facts": ["f"]}},
        {
            "id": "b",
            "split": "train",
            "inputs": {"question": "q2"},
            "expectations": {"expected_facts": ["g"]},
        },
    ]
    path = tmp_path / "golden.jsonl"
    gs.write_jsonl(records, path)
    assert gs.load_jsonl(path) == records
    assert gs.to_eval_rows(records) == [
        {"inputs": {"question": "q1"}, "expectations": {"expected_facts": ["f"]}}
    ]
    assert len(gs.to_eval_rows(records, split=None)) == 2


# --- summaries ---------------------------------------------------------------------------


def _records():
    return [
        {"id": "q1", "category": "sql", "tags": ["x"], "split": "test", "inputs": {"question": "Q1"},
         "expectations": {"reference_sql": "SELECT 1"}},
        {"id": "q2", "category": "sql", "tags": ["unsafe"], "split": "test", "inputs": {"question": "Q2"},
         "expectations": {"reference_sql": None}},
        {"id": "q3", "category": "review", "tags": [], "split": "test", "inputs": {"question": "Q3"},
         "expectations": {}},
    ]  # fmt: skip


def _result_df():
    return pd.DataFrame(
        [
            {"request": {"question": "Q1"}, "response": "A1", "correctness/value": "yes", "safety/value": "yes",
             "sql_result_match/value": False, "latency_seconds/value": 3.0, "trace_id": "t1"},
            {"request": json.dumps({"question": "Q2"}), "response": "A2", "correctness/value": "yes",
             "safety/value": "yes", "sql_result_match/value": None, "latency_seconds/value": 1.0, "trace_id": "t2"},
            {"request": {"question": "Q3"}, "response": {"text": "A3"}, "correctness/value": "no",
             "safety/value": "yes", "retrieval_groundedness/value": "yes", "latency_seconds/value": 2.0,
             "trace_id": "t3"},
        ]
    )  # fmt: skip


def test_summarise_applies_the_pass_definition():
    df, metrics = summarise(_result_df(), _records())
    assert df.set_index("id")["passed"].to_dict() == {"q1": False, "q2": True, "q3": False}
    assert df.set_index("id").loc["q3", "answer"] == '{"text": "A3"}'
    assert metrics["n_questions"] == 3 and metrics["pass_rate"] == round(1 / 3, 4)
    assert metrics["correctness_rate"] == round(2 / 3, 4)
    assert metrics["sql_result_match_rate"] == 0.0  # only q1 has a verdict
    assert metrics["groundedness_rate"] == 1.0
    assert metrics["median_latency_seconds"] == 2.0 and metrics["mean_total_tokens"] is None
    assert metrics["pass_rate_sql"] == 0.5 and metrics["pass_rate_review"] == 0.0
    json.dumps(metrics)  # the notebooks write these to JSON


def test_worst_failures_puts_wrong_answers_first():
    df, _ = summarise(_result_df(), _records())
    worst = worst_failures(df, n=5)
    assert list(worst["id"]) == ["q3", "q1"]


def test_headline_metrics_with_numpy_booleans():
    df = pd.DataFrame(
        {
            "passed": [True, False],
            "category": ["sql", "sql"],
            "sql_result_match": [True, False],
            "correctness": ["yes", "no"],
            "retrieval_groundedness": [None, None],
            "safety": ["yes", "yes"],
            "latency_seconds": [1.0, 3.0],
            "total_tokens": [100.0, None],
        }
    )
    m = headline_metrics(df)
    assert m["sql_result_match_rate"] == 0.5 and m["groundedness_rate"] is None
    assert m["mean_total_tokens"] == 100.0


# --- the evaluation run, end to end on a local MLflow store --------------------------------


def test_run_evaluation_end_to_end_with_code_scorers(tmp_path, monkeypatch, settings):
    mlflow = pytest.importorskip("mlflow")
    from agent.agent import BakehouseAgent
    from agent.tools import Text2Sql
    from evals import run_eval
    from evals.scorers import build_code_scorers

    monkeypatch.chdir(tmp_path)  # keep local MLflow artefacts out of the repo
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("eval-test")
    monkeypatch.setattr(run_eval, "build_scorers", lambda judge: build_code_scorers())

    class LLM:
        def __init__(self):
            self.chat = NS(completions=self)

        def create(self, **kw):
            msgs = kw["messages"]
            if msgs[0]["content"].startswith("Translate"):  # the Text2SQL step
                return NS(choices=[NS(message=NS(content="SELECT COUNT(*) AS n FROM silver_transactions"))])
            if "tools" in kw and msgs[-1]["role"] == "user":
                args = json.dumps({"question": msgs[-1]["content"]})
                call = NS(id="c1", function=NS(name="query_sales_data", arguments=args))
                return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))])
            return NS(choices=[NS(message=NS(content="There are 3 transactions.", tool_calls=None))])

    class Executor:
        def execute(self, sql, timeout_seconds):
            return ["n"], [[3]]

    llm = LLM()
    agent = BakehouseAgent(
        settings, llm_client=llm, text2sql=Text2Sql(settings, llm, Executor(), "TABLE silver_transactions")
    )
    expected = {"reference_sql": "SELECT COUNT(*) FROM t", "expected_facts": ["3"]}
    records = [
        {"id": "q1", "category": "sql", "tags": [], "split": "test", "inputs": {"question": "How many?"},
         "expectations": {**expected, "expected_result": {"columns": ["n"], "rows": [[3]]}}},
        {"id": "q2", "category": "sql", "tags": [], "split": "test", "inputs": {"question": "How many now?"},
         "expectations": {**expected, "expected_result": {"columns": ["n"], "rows": [[4]]}}},
        {"id": "q3", "category": "sql", "tags": [], "split": "train", "inputs": {"question": "Not evaluated"},
         "expectations": expected},
    ]  # fmt: skip
    run_id, per_question, metrics = run_eval.run_evaluation(agent, settings, records, run_name="unit")

    by_id = per_question.set_index("id")
    assert sorted(by_id.index) == ["q1", "q2"]  # the train question is not evaluated
    assert bool(by_id.loc["q1", "sql_result_match"]) is True
    assert bool(by_id.loc["q2", "sql_result_match"]) is False
    assert by_id.loc["q1", "answer"] == "There are 3 transactions."
    assert by_id.loc["q1", "tool_calls_count"] == 1
    assert metrics["n_questions"] == 2 and metrics["sql_result_match_rate"] == 0.5
    assert metrics["pass_rate"] == 0.0  # no Correctness judge in this test, so nothing can pass

    run = mlflow.get_run(run_id)
    assert run.data.params["prompt_variant"] == "baseline" and run.data.params["n_questions"] == "2"
    assert run.data.metrics["sql_result_match_rate"] == 0.5


# --- combining judge verdicts ----------------------------------------------------------------


def test_combine_feedbacks():
    pytest.importorskip("mlflow")
    from mlflow.entities import Feedback

    from evals.scorers import combine_feedbacks

    yes = Feedback(name="g", value="yes", rationale="supported")
    no = Feedback(name="g", value="no", rationale="not in the reviews")
    assert combine_feedbacks([yes, yes], "g").value == "yes"
    combined = combine_feedbacks([yes, no], "g")
    assert combined.value == "no" and combined.rationale == "not in the reviews"
    assert combine_feedbacks([], "g") is None
    assert combine_feedbacks(yes, "g") is yes


def test_groundedness_scorer_is_not_applicable_without_retrieval_and_combines_spans(monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.entities import Feedback, SpanType

    from evals import scorers

    scorer = scorers.build_groundedness_scorer("databricks:/judge")

    class Trace:
        def __init__(self, n):
            self.n = n

        def search_spans(self, span_type=None, name=None):
            return [object()] * self.n if span_type == SpanType.RETRIEVER else []

    assert scorer(trace=Trace(0)) is None

    from mlflow.genai.scorers import RetrievalGroundedness

    verdicts = [
        Feedback(name="retrieval_groundedness", value="yes"),
        Feedback(name="retrieval_groundedness", value="no"),
    ]
    monkeypatch.setattr(RetrievalGroundedness, "__call__", lambda self, trace: verdicts)
    scorer = scorers.build_groundedness_scorer("databricks:/judge")
    assert scorer(trace=Trace(2)).value == "no"


# --- Genie error paths ----------------------------------------------------------------------


def test_ask_genie_returns_errors_instead_of_raising():
    from evals.genie import ask_genie

    def fail(space_id, content, timeout):
        raise RuntimeError("PERMISSION_DENIED: no access to space")

    out = ask_genie(NS(genie=NS(start_conversation_and_wait=fail)), "space", "q")
    assert out["status"] == "ERROR" and "PERMISSION_DENIED" in out["error"] and out["rows"] == []


def test_ask_genie_backs_off_on_rate_limits(monkeypatch):
    from evals import genie

    sleeps = []
    monkeypatch.setattr(genie.time, "sleep", sleeps.append)

    def limited(space_id, content, timeout):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    out = genie.ask_genie(NS(genie=NS(start_conversation_and_wait=limited)), "space", "q", retries=3)
    assert out["status"] == "RATE_LIMITED" and sleeps == [30, 60, 90]


def test_ask_genie_text_only_answer_and_failed_result_fetch():
    from databricks.sdk.service.dashboards import MessageStatus

    from evals.genie import ask_genie, parse_statement_response

    message = NS(
        conversation_id="c",
        message_id="m",
        status=MessageStatus.COMPLETED,
        error=None,
        attachments=[
            NS(attachment_id="a0", text=NS(content="Here is the answer."), query=None),
            NS(attachment_id="a1", text=None, query=NS(query="SELECT 1", description=None)),
        ],
    )

    def broken_result(*args):
        raise RuntimeError("result expired")

    genie_api = NS(
        start_conversation_and_wait=lambda s, c, timeout: message,
        get_message_attachment_query_result=broken_result,
    )
    out = ask_genie(NS(genie=genie_api), "space", "q")
    assert out["sql"] == "SELECT 1" and out["rows"] == []
    assert "Here is the answer." in out["answer"] and "result expired" in out["answer"]
    assert parse_statement_response(None) == ([], [])
