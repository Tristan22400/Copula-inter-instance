"""Mixed preprocessed feature widths must preserve predictions and gradients."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eval.spatial import tabldm_batched
from marginal_backbones import _tabldm_quantile_forward


class Scaler:
    def __init__(self, value):
        self.scale_ = np.array([value + 1.0])
        self.mean_ = np.array([value * 10.0])

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = []

    def predict_stats(self, xs, ys, *, output_type, inference_config=None, alphas=None):
        self.calls.append(tuple(xs.shape))
        levels = torch.arange(3 if alphas is None else len(alphas), device=xs.device)
        return self.weight * xs[:, ys.shape[1]:].mean(dim=-1, keepdim=True) + levels


@pytest.mark.parametrize('widths', [(11, 10, 11), (11, 11, 11)])
@pytest.mark.parametrize('probs', [None, [0.1, 0.9]])
def test_grouped_predictions_and_gradients_match_separate_episodes(monkeypatch, widths, probs):
    episodes = [
        (np.full((2, 5, width), b + 1, dtype=np.float32),
         np.zeros((2, 3), dtype=np.float32), Scaler(b + 1))
        for b, width in enumerate(widths)
    ]
    monkeypatch.setattr(tabldm_batched, '_episode_member_batch', lambda reg, x, y, q: episodes[x])
    monkeypatch.setattr('marginal_backbones._patch_tabldm_inference_manager', lambda: None)
    model = Model()
    handle = SimpleNamespace(inference_config_=None)
    bb = SimpleNamespace(module=model, handle=handle)
    ids = list(range(len(episodes)))
    actual = _tabldm_quantile_forward(bb, ids, ids, ids, probs)
    assert len(model.calls) == len(set(widths))
    assert model.calls[0][0] == 2 * widths.count(widths[0])
    reference = torch.cat([
        _tabldm_quantile_forward(bb, [b], [b], [b], probs) for b in ids
    ])
    torch.testing.assert_close(actual, reference)
    grad = torch.autograd.grad(actual.square().sum(), model.weight)[0]
    reference_grad = torch.autograd.grad(reference.square().sum(), model.weight)[0]
    torch.testing.assert_close(grad, reference_grad)
    assert grad.abs() > 0

    if probs is not None:
        def batch_forward(xs, ys, **kwargs):
            with torch.no_grad():
                return model.predict_stats(torch.from_numpy(xs), torch.from_numpy(ys), **kwargs).numpy()
        handle._batch_forward = batch_forward
        model.calls.clear()
        bank = tabldm_batched._quantile_bank_batched(handle, ids, ids, ids, np.array(probs))
        assert len(model.calls) == len(set(widths))
        np.testing.assert_allclose(bank, reference.detach().numpy())
