import sys
from types import SimpleNamespace

import pytest
import torch

from gcpo.encoder import encode_responses, load_frozen_encoder


class _FakeParameter:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


class _FakeSentenceTransformer:
    construction = None

    def __init__(self, model_path, *, device, local_files_only):
        type(self).construction = (model_path, device, local_files_only)
        self.training = True
        self._parameters = [_FakeParameter(), _FakeParameter()]

    def parameters(self):
        return iter(self._parameters)

    def eval(self):
        self.training = False
        return self


def test_load_frozen_encoder_is_local_eval_and_non_trainable(monkeypatch, tmp_path):
    model_path = tmp_path / "all-MiniLM-L6-v2"
    model_path.mkdir()
    fake_module = SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    model = load_frozen_encoder(str(model_path), device="cuda:3")

    assert _FakeSentenceTransformer.construction == (
        str(model_path.resolve()),
        "cuda:3",
        True,
    )
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())


def test_load_frozen_encoder_rejects_missing_local_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="semantic encoder"):
        load_frozen_encoder(str(tmp_path / "missing"), device="cpu")


class _FakeTokenizer:
    def batch_decode(self, response_ids, *, skip_special_tokens):
        assert torch.equal(response_ids, torch.tensor([[1, 2], [3, 0]]))
        assert skip_special_tokens is True
        return ["first response", "second response"]


class _FakeEncoder:
    def encode(
        self,
        texts,
        *,
        convert_to_tensor,
        show_progress_bar,
        batch_size,
        normalize_embeddings,
    ):
        assert texts == ["first response", "second response"]
        assert convert_to_tensor is True
        assert show_progress_bar is False
        assert batch_size == 2
        assert normalize_embeddings is False
        return torch.tensor([[3.0, 4.0], [5.0, 12.0]], requires_grad=True)


def test_encode_responses_preserves_unnormalized_detached_vectors():
    response_ids = torch.tensor([[1, 2], [3, 0]])

    embeddings = encode_responses(
        model=_FakeEncoder(),
        tokenizer=_FakeTokenizer(),
        response_ids=response_ids,
        batch_size=2,
    )

    torch.testing.assert_close(
        embeddings, torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    )
    assert embeddings.device == response_ids.device
    assert embeddings.requires_grad is False


def test_encode_responses_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        encode_responses(
            model=_FakeEncoder(),
            tokenizer=_FakeTokenizer(),
            response_ids=torch.ones(2, 2, dtype=torch.long),
            batch_size=0,
        )
