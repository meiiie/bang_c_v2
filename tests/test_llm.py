from types import SimpleNamespace
import pytest
from solver.llm import LLMClient, NoLLM


def client(monkeypatch, usage):
    monkeypatch.setenv('OPENROUTER_API_KEY','offline-unit-test')
    monkeypatch.setenv('SOLVER_MODEL','qwen/qwen3.7-flash')
    monkeypatch.setenv('SOLVER_INPUT_PRICE','1')
    monkeypatch.setenv('SOLVER_OUTPUT_PRICE','4')
    c=LLMClient()
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=usage,choices=[SimpleNamespace(message=SimpleNamespace(content='answer = 1'))])
    c.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return c,calls


def test_reported_usage(monkeypatch):
    c,calls=client(monkeypatch,SimpleNamespace(cost=.002,prompt_tokens=20,completion_tokens=10))
    text,cost,usage=c.complete([{'role':'user','content':'test'}])
    assert (text,cost) == ('answer = 1',.002)
    assert usage['prompt_tokens'] == 20
    assert usage['completion_tokens'] == 10
    assert calls[0]['max_tokens'] == 4000
    assert calls[0]['extra_body']['reasoning'] == {'effort':'low'}
    assert c.total_cost == .002
    assert calls[0]['model'] == 'qwen/qwen3.7-flash'


def test_estimated_usage_and_reservation(monkeypatch):
    c,calls=client(monkeypatch,SimpleNamespace(prompt_tokens=100,completion_tokens=10))
    text,cost,usage=c.complete([{'role':'user','content':'test'}],max_cost=.001)
    assert cost == pytest.approx(.00014)
    assert calls[0]['max_tokens'] < 1400
    with pytest.raises(NoLLM): c.complete([{'role':'user','content':'test'}],max_cost=.00001)
    assert len(calls) == 1


def test_model_restriction(monkeypatch):
    monkeypatch.setenv('SOLVER_MODEL','disallowed-model')
    with pytest.raises(ValueError): LLMClient()


@pytest.mark.parametrize('level,expected', [('off', {'enabled':False}), ('low', {'effort':'low'}), ('medium', {'effort':'medium'})])
def test_reasoning_and_token_configuration(monkeypatch, level, expected):
    monkeypatch.setenv('SOLVER_REASONING',level)
    monkeypatch.setenv('SOLVER_MAX_TOKENS','6000')
    c,calls=client(monkeypatch,SimpleNamespace(prompt_tokens=20,completion_tokens=40,
                   completion_tokens_details=SimpleNamespace(reasoning_tokens=30)))
    _,_,usage=c.complete([{'role':'user','content':'test'}])
    assert calls[0]['max_tokens'] == 6000
    assert calls[0]['extra_body']['reasoning'] == expected
    assert usage['reasoning_tokens'] == 30


def test_invalid_reasoning(monkeypatch):
    monkeypatch.setenv('SOLVER_REASONING','high')
    with pytest.raises(ValueError): client(monkeypatch,None)
