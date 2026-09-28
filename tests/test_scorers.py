from types import SimpleNamespace as NS

import pytest

from evals.golden_set import SPECS, assign_splits, fact_values
from evals.scorers import compute_pass, normalise_value, results_match, sql_runs, tool_call_count

# --- normalisation ------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        (12345.678, "12345.68"),
        (10, 10.0),
        ("  Sweet Crumbs ", "sweet crumbs"),
        ("2024-05-01T00:00:00", "2024-05-01"),
        (None, float("nan")),
    ],
)
def test_normalise_value_equivalences(a, b):
    assert normalise_value(a) == normalise_value(b)


def test_normalise_keeps_different_numbers_different():
    assert normalise_value(10.01) != normalise_value(10.02)


# --- result matching ----------------------------------------------------------


def test_exact_match():
    assert results_match(["a"], [[1], [2]], ["a"], [[1], [2]])


def test_row_order_is_ignored():
    assert results_match(["a", "b"], [[1, "x"], [2, "y"]], ["a", "b"], [[2, "y"], [1, "x"]])


def test_column_order_and_names_are_ignored():
    assert results_match(["name", "rev"], [["A", 10.0]], ["revenue", "franchise"], [[10.004, "a"]])


def test_extra_predicted_columns_are_allowed():
    assert results_match(["name"], [["A"], ["B"]], ["id", "name", "rev"], [[1, "A", 5], [2, "B", 6]])


def test_row_count_must_match():
    assert not results_match(["a"], [[1]], ["a"], [[1], [2]])


def test_missing_reference_column_fails():
    assert not results_match(["a", "b"], [[1, 2]], ["a"], [[1]])


def test_values_must_line_up_row_by_row():
    # Each column matches as a multiset, but rows are paired differently.
    ref = [["A", 1], ["B", 2]]
    pred = [["A", 2], ["B", 1]]
    assert not results_match(["n", "v"], ref, ["n", "v"], pred)


def test_duplicate_rows_are_counted():
    assert not results_match(["a"], [[1], [1]], ["a"], [[1], [2]])


def test_ties_need_all_rows():
    assert not results_match(["p"], [["x"], ["y"]], ["p"], [["x"]])


def test_empty_results_match():
    assert results_match(["a"], [], ["b"], [])


# --- pass definition ------------------------------------------------------------


def test_pass_requires_correct_and_safe():
    assert compute_pass("yes", "yes", None, has_reference_sql=False)
    assert not compute_pass("no", "yes", None, has_reference_sql=False)
    assert not compute_pass("yes", "no", None, has_reference_sql=False)


def test_pass_requires_sql_match_when_reference_exists():
    assert compute_pass("yes", "yes", True, has_reference_sql=True)
    assert not compute_pass("yes", "yes", False, has_reference_sql=True)
    assert not compute_pass("yes", "yes", None, has_reference_sql=True)


# --- trace helpers (with a fake trace) --------------------------------------------


def _span(name, start, outputs=None):
    return NS(name=name, start_time_ns=start, outputs=outputs)


class FakeTrace:
    def __init__(self, spans):
        self.spans = spans

    def search_spans(self, span_type=None, name=None):
        return [s for s in self.spans if name is None or s.name == name]


def test_sql_runs_are_ordered():
    t = FakeTrace([_span("run_sql", 2, {"sql": "b"}), _span("run_sql", 1, {"sql": "a"})])
    assert [r["sql"] for r in sql_runs(t)] == ["a", "b"]


def test_tool_call_count_counts_top_level_tools():
    t = FakeTrace([_span("query_sales_data", 1), _span("run_sql", 2), _span("search_reviews", 3)])
    assert tool_call_count(t) == 2


# --- golden set helpers -------------------------------------------------------------


def test_fact_values_handles_ties():
    v = fact_values(["product", "units"], [["A", 5], ["B", 5]], {"day": "2024-05-01"})
    assert v["all_product"] == "A and B" and v["units"] == 5 and v["day"] == "2024-05-01"


def test_golden_set_mix_and_ids():
    ids = [s["id"] for s in SPECS]
    assert len(ids) == len(set(ids))
    assert 50 <= len(SPECS) <= 60
    cats = [s["category"] for s in SPECS]
    assert cats.count("sql") > cats.count("review") > cats.count("hybrid")
    unsafe = [s for s in SPECS if "unsafe" in s["tags"]]
    assert len(unsafe) == 3


def test_split_is_deterministic_and_keeps_unsafe_in_test():
    records = [{"id": s["id"], "category": s["category"], "tags": s["tags"]} for s in SPECS]
    a = assign_splits(records, train_size=30, seed=17)
    b = assign_splits(records, train_size=30, seed=17)
    assert a == b
    assert sum(r["split"] == "train" for r in a) == 30
    for r in a:
        if {"unsafe", "no_answer"} & set(r["tags"]):
            assert r["split"] == "test"
    train_cats = {r["category"] for r in a if r["split"] == "train"}
    assert train_cats == {"sql", "review", "hybrid"}


# --- code scorers on a real (local) MLflow trace -------------------------------------


def test_code_scorers_read_real_trace(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("scorer-test")

    from agent.config import load_settings
    from agent.tools import run_sql
    from evals.scorers import build_code_scorers

    settings = load_settings()

    class Ex:
        def execute(self, sql, timeout_seconds):
            return ["franchise_name", "revenue"], [["Shop A", 10.0]]

    @mlflow.trace(name="query_sales_data", span_type="TOOL")
    def tool():
        return run_sql("SELECT franchise_name, revenue FROM silver_franchises", settings, Ex())

    tool()
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    scorers = {s.name: s for s in build_code_scorers()}
    expectations = {
        "reference_sql": "SELECT ...",
        "expected_result": {"columns": ["franchise_name"], "rows": [["shop a"]]},
    }
    assert scorers["sql_executes"](expectations=expectations, trace=trace).value is True
    assert scorers["sql_result_match"](expectations=expectations, trace=trace).value is True
    assert scorers["tool_calls_count"](trace=trace) == 1
    assert scorers["sql_result_match"](expectations={"reference_sql": None}, trace=trace) is None


def test_llm_judges_can_be_configured_without_a_connection():
    pytest.importorskip("mlflow")
    from evals.scorers import build_scorers

    names = [s.name for s in build_scorers("databricks:/some-endpoint")]
    for expected in [
        "correctness",
        "retrieval_groundedness",
        "relevance_to_query",
        "safety",
        "cites_review_ids",
        "refuses_unsafe",
        "sql_result_match",
    ]:
        assert expected in names


# --- DSPy export and metric (no LLM calls) -------------------------------------------


def test_dspy_export_and_metric():
    dspy = pytest.importorskip("dspy")
    from agent.config import load_settings
    from evals.dspy_text2sql import baseline_program, export_program, make_examples, make_metric

    program = baseline_program()
    exported = export_program(program)
    assert "SELECT" in exported["instructions"] and exported["demos"] == []

    program.predictors()[0].demos = [dspy.Example(question="How many?", sql="```sql\nSELECT 1\n```")]
    assert export_program(program)["demos"] == [{"question": "How many?", "sql": "SELECT 1"}]

    records = [
        {
            "id": "a",
            "inputs": {"question": "q"},
            "expectations": {
                "reference_sql": "SELECT 1",
                "expected_result": {"columns": ["n"], "rows": [[3]]},
            },
        },
        {"id": "b", "inputs": {"question": "unsafe"}, "expectations": {"reference_sql": None}},
    ]
    examples = make_examples(records, "TABLE silver_transactions")
    assert len(examples) == 1 and set(examples[0].inputs().keys()) == {"question", "schema_context"}

    class Ex:
        def execute(self, sql, timeout_seconds):
            return ["count"], [[3]]

    metric = make_metric(load_settings(), Ex())
    good = dspy.Prediction(sql="SELECT COUNT(*) FROM silver_transactions")
    bad = dspy.Prediction(sql="DROP TABLE silver_transactions")
    assert metric(examples[0], good) == 1.0
    assert metric(examples[0], bad) == 0.0
