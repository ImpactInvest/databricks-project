"""Settings loading, prompt rendering and review chunking (pure Python)."""

import json
import shutil

import pytest
import yaml

from agent import prompts
from agent.config import CONFIG_DIR, Settings, _deep_merge, load_settings, write_runtime_values
from pipeline.chunking import chunk_text, clean_text, split_sentences

# --- settings -------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path):
    shutil.copy(CONFIG_DIR / "settings.yaml", tmp_path / "settings.yaml")
    return tmp_path


def test_deep_merge_overrides_nested_keys_without_mutating_inputs():
    base = {"a": {"x": 1, "y": 2}, "b": [1]}
    merged = _deep_merge(base, {"a": {"y": 3}, "c": 4})
    assert merged == {"a": {"x": 1, "y": 3}, "b": [1], "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1]}


def test_committed_defaults_load_and_naming_helpers(config_dir):
    s = load_settings(config_dir)
    assert s.catalog == "workspace" and s.schema == "genai_agent"
    assert s.table("transactions") == "workspace.genai_agent.silver_transactions"
    assert s.index_name == "workspace.genai_agent.reviews_chunked_index"
    assert s.model_name == "workspace.genai_agent.bakehouse_agent"
    assert s.silver_tables == [
        "silver_transactions",
        "silver_customers",
        "silver_franchises",
        "silver_suppliers",
    ]
    assert "silver_reviews" not in s.silver_tables  # reviews are reached through AI Search only


def test_runtime_file_and_overrides_win(config_dir):
    write_runtime_values({"llm": {"agent_endpoint": "ep-a"}, "sql": {"warehouse_id": "wh1"}}, config_dir)
    write_runtime_values({"llm": {"judge_endpoint": "ep-j"}}, config_dir)  # merges, keeps earlier keys
    s = load_settings(config_dir, overrides={"agent": {"prompt_variant": "optimised"}})
    assert s["llm"]["agent_endpoint"] == "ep-a" and s["llm"]["judge_endpoint"] == "ep-j"
    assert s["llm"]["temperature"] == 0.0  # untouched default survives the merge
    assert s["sql"]["warehouse_id"] == "wh1" and s["agent"]["prompt_variant"] == "optimised"
    runtime = yaml.safe_load((config_dir / "runtime.yaml").read_text())
    assert runtime == {
        "llm": {"agent_endpoint": "ep-a", "judge_endpoint": "ep-j"},
        "sql": {"warehouse_id": "wh1"},
    }


def test_empty_runtime_file_is_ignored(config_dir):
    (config_dir / "runtime.yaml").write_text("")
    assert load_settings(config_dir)["sql"]["warehouse_id"] is None


def test_to_dict_is_a_copy():
    s = Settings({"a": {"b": 1}})
    d = s.to_dict()
    d["a"]["b"] = 2
    assert s["a"]["b"] == 1 and s.get("missing", 5) == 5


# --- prompts --------------------------------------------------------------------


def test_prompt_variants(tmp_path, monkeypatch):
    instructions, demos = prompts.text2sql_prompt_parts("baseline")
    assert instructions == prompts.TEXT2SQL_BASELINE_INSTRUCTIONS and demos == []

    monkeypatch.setattr(prompts, "OPTIMISED_PROMPT_PATH", tmp_path / "optimised_text2sql.json")
    with pytest.raises(FileNotFoundError):
        prompts.text2sql_prompt_parts("optimised")

    exported = {"instructions": "Be precise.", "demos": [{"question": "q", "sql": "SELECT 1"}]}
    (tmp_path / "optimised_text2sql.json").write_text(json.dumps(exported))
    assert prompts.text2sql_prompt_parts("optimised") == ("Be precise.", exported["demos"])

    with pytest.raises(ValueError):
        prompts.text2sql_prompt_parts("something-else")


def test_retry_prompt_feeds_back_the_failed_sql_and_error():
    msgs = prompts.render_text2sql_messages(
        "q?", "TABLE t", "INSTR", previous_attempt={"sql": "SELECT x", "error": "UNRESOLVED_COLUMN x"}
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "SELECT x" and "UNRESOLVED_COLUMN" in msgs[3]["content"]


@pytest.mark.parametrize(
    "reply,sql",
    [
        ("```SQL\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 2;\n```\nThis counts rows.", "SELECT 2"),
        ("  SELECT 3 ;  ", "SELECT 3"),
        ("", ""),
        (None, ""),
    ],
)
def test_extract_sql_variants(reply, sql):
    assert prompts.extract_sql(reply) == sql


def test_system_prompt_states_the_safety_rules():
    text = prompts.AGENT_SYSTEM_PROMPT.lower()
    for rule in ["read-only", "[review 123]", "i don't know", "never invent"]:
        assert rule in text


# --- chunking -------------------------------------------------------------------


def test_clean_text():
    assert clean_text(None) == ""
    assert clean_text("  a \n\t b  ") == "a b"


def test_split_sentences():
    assert split_sentences("Great bread!  Slow staff. Would return?") == [
        "Great bread!",
        "Slow staff.",
        "Would return?",
    ]
    assert split_sentences("") == []


def test_short_review_is_one_chunk():
    assert chunk_text("Lovely cookies. Friendly staff.", max_words=120) == ["Lovely cookies. Friendly staff."]


def test_empty_review_has_no_chunks():
    assert chunk_text(None) == [] and chunk_text("   ") == []


def test_long_review_uses_sentence_windows_with_overlap():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_text(text, max_words=6, overlap_sentences=1)
    assert chunks == [
        "One two three. Four five six.",
        "Four five six. Seven eight nine.",
        "Seven eight nine. Ten eleven twelve.",
    ]
    assert all(len(c.split()) <= 6 for c in chunks)


def test_no_overlap():
    text = "One two three. Four five six. Seven eight nine."
    assert chunk_text(text, max_words=6, overlap_sentences=0) == [
        "One two three. Four five six.",
        "Seven eight nine.",
    ]


def test_sentence_longer_than_the_limit_is_split_on_words():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text(" ".join(words), max_words=10, overlap_sentences=0)
    assert [len(c.split()) for c in chunks] == [10, 10, 5]
    assert " ".join(chunks).split() == words


def test_large_overlap_still_makes_progress_and_covers_every_sentence():
    sentences = [f"Sentence number {i}." for i in range(12)]
    chunks = chunk_text(" ".join(sentences), max_words=6, overlap_sentences=5)
    assert len(chunks) < 50  # terminates
    for s in sentences:
        assert any(s in c for c in chunks)


def test_invalid_max_words():
    with pytest.raises(ValueError):
        chunk_text("x", max_words=0)
