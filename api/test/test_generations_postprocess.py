"""
Unit tests for the post-treatment steps of generative pipelines
(activetigger.generations, issue #1100)
"""

import pytest

from activetigger.datamodels import PostprocessStep
from activetigger.generations import Generations

LABELS = ["Positive", "Negative", "Neutral"]


def apply(raw: str, steps: list[dict], labels: list[str] | None = LABELS):
    return Generations.apply_postprocess(raw, [PostprocessStep(**s) for s in steps], labels)


# ---------------------------------------------------------------------------
# text transforms
# ---------------------------------------------------------------------------


def test_strip_and_free_output():
    predicted, error = apply("  hello \n", [{"name": "strip"}], labels=None)
    assert predicted == "hello"
    assert error is None


def test_case_transforms():
    assert apply("AbC", [{"name": "lowercase"}], labels=None)[0] == "abc"
    assert apply("AbC", [{"name": "uppercase"}], labels=None)[0] == "ABC"


def test_remove_punctuation():
    assert apply("po-si.tive!", [{"name": "remove_punctuation"}], labels=None)[0] == "positive"


def test_regex_sub():
    steps = [{"name": "regex_sub", "params": {"pattern": "label\\s*:\\s*", "replacement": ""}}]
    assert apply("label: Positive", steps, labels=None)[0] == "Positive"


def test_regex_sub_ignore_case():
    steps = [
        {
            "name": "regex_sub",
            "params": {"pattern": "LABEL: ", "replacement": "", "ignore_case": True},
        }
    ]
    assert apply("label: Positive", steps, labels=None)[0] == "Positive"


# ---------------------------------------------------------------------------
# json steps
# ---------------------------------------------------------------------------


def test_json_extract_from_verbose_output():
    raw = 'Sure! Here is the JSON:\n```json\n{"label": "Positive", "score": 1}\n```'
    predicted, error = apply(raw, [{"name": "json_extract"}], labels=None)
    assert predicted == '{"label": "Positive", "score": 1}'


def test_json_extract_skips_invalid_candidates():
    raw = '{not json} but then {"label": "Neutral"}'
    predicted, _ = apply(raw, [{"name": "json_extract"}], labels=None)
    assert predicted == '{"label": "Neutral"}'


def test_json_extract_no_json_is_na():
    predicted, error = apply("no json here", [{"name": "json_extract"}])
    assert predicted is None
    assert "no JSON" in error


def test_json_key():
    raw = '{"label": "Positive", "why": "..."}'
    steps = [{"name": "json_key", "params": {"key": "label"}}]
    assert apply(raw, steps, labels=None)[0] == "Positive"


def test_json_key_missing_is_na():
    steps = [{"name": "json_key", "params": {"key": "label"}}]
    predicted, error = apply('{"other": 1}', steps)
    assert predicted is None
    assert "label" in error


def test_json_key_on_non_json_is_na():
    steps = [{"name": "json_key", "params": {"key": "label"}}]
    predicted, error = apply("plain text", steps)
    assert predicted is None
    assert "not valid JSON" in error


# ---------------------------------------------------------------------------
# matching steps
# ---------------------------------------------------------------------------


def test_exact_match():
    predicted, error = apply("Positive", [{"name": "exact_match"}])
    assert predicted == "Positive"
    assert error is None


def test_exact_match_ignore_case_default():
    predicted, _ = apply("positive", [{"name": "exact_match"}])
    assert predicted == "Positive"


def test_exact_match_case_sensitive():
    steps = [{"name": "exact_match", "params": {"ignore_case": False}}]
    predicted, error = apply("positive", steps)
    assert predicted is None
    assert error is not None


def test_exact_match_no_match_is_na():
    predicted, error = apply("I think it is positive overall", [{"name": "exact_match"}])
    assert predicted is None
    assert "does not match" in error


def test_fuzzy_match():
    steps = [{"name": "fuzzy_match", "params": {"max_distance": 2}}]
    predicted, _ = apply("Positve", steps)  # one deletion away
    assert predicted == "Positive"


def test_fuzzy_match_too_far_is_na():
    steps = [{"name": "fuzzy_match", "params": {"max_distance": 1}}]
    predicted, error = apply("Pos", steps)
    assert predicted is None
    assert "distance" in error


def test_regex_assign_match_stops_the_pipe():
    steps = [
        {
            "name": "regex_assign",
            "params": {"pattern": "pos", "label": "Positive", "ignore_case": True},
        },
        {"name": "exact_match"},
    ]
    predicted, error = apply("This is POSITIVE", steps)
    assert predicted == "Positive"
    assert error is None


def test_regex_assign_no_match_continues():
    steps = [
        {"name": "regex_assign", "params": {"pattern": "xyz", "label": "Positive"}},
        {"name": "exact_match"},
    ]
    predicted, _ = apply("Neutral", steps)
    assert predicted == "Neutral"


# ---------------------------------------------------------------------------
# whole pipes
# ---------------------------------------------------------------------------


def test_full_chain_json_then_match():
    raw = 'Answer:\n{"label": " negative ", "confidence": 0.9}'
    steps = [
        {"name": "json_extract"},
        {"name": "json_key", "params": {"key": "label"}},
        {"name": "strip"},
        {"name": "exact_match"},
    ]
    predicted, error = apply(raw, steps)
    assert predicted == "Negative"
    assert error is None


def test_scheme_pipe_without_matching_step_is_na():
    predicted, error = apply("Positive", [{"name": "strip"}])
    assert predicted is None
    assert "no step assigned a label" in error


def test_empty_pipe_free_returns_raw():
    predicted, error = apply("anything", [], labels=None)
    assert predicted == "anything"
    assert error is None


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_unknown_step_is_refused():
    with pytest.raises(Exception, match="Unknown post-treatment step"):
        Generations.validate_steps([PostprocessStep(name="does_not_exist")])


def test_invalid_regex_is_refused():
    with pytest.raises(Exception, match="Invalid parameters"):
        Generations.validate_steps([PostprocessStep(name="regex_sub", params={"pattern": "("})])


def test_missing_required_param_is_refused():
    with pytest.raises(Exception, match="Invalid parameters"):
        Generations.validate_steps([PostprocessStep(name="json_key")])
