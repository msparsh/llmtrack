"""Tests for usage and model extraction from various provider responses."""

from types import SimpleNamespace

from llmtrack.extractors import extract_usage


def test_anthropic_object_extraction():
    resp = SimpleNamespace(
        model="claude-3-5-sonnet-20241022",
        usage=SimpleNamespace(input_tokens=150, output_tokens=75),
    )
    model, inp, out = extract_usage(resp, provider="anthropic")
    assert model == "claude-3-5-sonnet-20241022"
    assert inp == 150
    assert out == 75


def test_openai_object_extraction():
    resp = SimpleNamespace(
        model="gpt-4o",
        usage=SimpleNamespace(prompt_tokens=200, completion_tokens=80),
    )
    model, inp, out = extract_usage(resp, provider="openai")
    assert model == "gpt-4o"
    assert inp == 200
    assert out == 80


def test_dict_response_extraction():
    resp = {
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
        },
    }
    model, inp, out = extract_usage(resp)
    assert model == "gpt-4o-mini"
    assert inp == 50
    assert out == 25


def test_custom_extractor():
    resp = {"data": {"m": "custom-v1", "in": 12, "out": 34}}

    def my_extractor(r):
        d = r.get("data", {})
        return d.get("m"), d.get("in"), d.get("out")

    model, inp, out = extract_usage(resp, custom_extractor=my_extractor)
    assert model == "custom-v1"
    assert inp == 12
    assert out == 34


def test_missing_attributes_fallback():
    model, inp, out = extract_usage("just a plain string")
    assert model is None
    assert inp is None
    assert out is None
