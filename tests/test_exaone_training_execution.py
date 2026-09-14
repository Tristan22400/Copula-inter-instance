"""Execution optimizations must preserve LoRA gradients across optimizer steps."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from lora import LoRAParametrization
from marginal_backbones import MarginalBackbone, _exaone_grad_forward


class RepeatedWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))
        parametrize.register_parametrization(self, "weight", LoRAParametrization(self.weight, 2, 4))
        self.parametrizations.weight.original.requires_grad_(False)

    def forward(self, support, label, query, **kwargs):
        hidden = F.linear(query + support.mean(1, keepdim=True), self.weight).tanh()
        return F.linear(hidden, self.weight) + label.mean(1)[:, None, None]


@pytest.mark.parametrize("checkpointing", [True, False])
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_cached_chunked_forward_matches_uncached_updates(checkpointing, chunk_size):
    torch.manual_seed(42)
    reference = RepeatedWeightModel()
    actual = deepcopy(reference)
    bb = MarginalBackbone("exaone", actual, SimpleNamespace(model=actual),
                          exaone_chunk_size=chunk_size,
                          exaone_activation_checkpointing=checkpointing)
    opts = [torch.optim.SGD(m.parameters(), lr=0.01) for m in (reference, actual)]
    for _ in range(2):
        support, label, query = torch.randn(5, 7, 3), torch.randn(5, 7), torch.randn(5, 4, 3)
        # Multiple outstanding forwards exercise checkpoint recomputation/cache isolation.
        expected = reference(support, label, query) + reference(support, label, query * 2)
        observed = _exaone_grad_forward(bb, support, label, query)
        observed = observed + _exaone_grad_forward(bb, support, label, query * 2)
        torch.testing.assert_close(observed, expected)
        expected.square().mean().backward()
        observed.square().mean().backward()
        for p, q in zip(reference.parameters(), actual.parameters()):
            if p.requires_grad:
                assert p.grad is not None and q.grad is not None
                torch.testing.assert_close(q.grad, p.grad)
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            torch.testing.assert_close(_exaone_grad_forward(bb, support, label, query),
                                       reference(support, label, query))


def test_invalid_chunk_size():
    model = RepeatedWeightModel()
    bb = MarginalBackbone("exaone", model, SimpleNamespace(model=model), exaone_chunk_size=0)
    with pytest.raises(ValueError, match="positive"):
        _exaone_grad_forward(bb, torch.randn(1, 7, 3), torch.randn(1, 7), torch.randn(1, 4, 3))
