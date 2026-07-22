"""energy_score.py — Energy Score for sample-based joint predictive evaluation.

Direct numpy port of ``src/loss.py::energy_score``'s scoring formula (its
``term1``/``term2`` computation, lines ~478-485). That torch function can't be
called as-is here: it *draws* its own Monte-Carlo samples internally from a
``mu + D^{1/2} eps_diag + V @ eps_low`` low-rank Gaussian, a parameterization
that only exists for the model's native Z-space output — TabICL's marginal
quantile grids are not Gaussian, so there is no ``(mu, D, V)`` to hand it.
``sample_trajectories`` (inference/copula_inference.py) already draws the
correct (possibly non-Gaussian) Y-space samples for every method in this
benchmark suite; this module just scores them with the same formula.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

__all__ = ["compute_energy_score"]


def compute_energy_score(samples: np.ndarray, y_true: np.ndarray) -> float:
    """Energy Score:  ES = E_P[||Y - y_true||] - 0.5 * E_P[||Y - Y'||].

    Args:
        samples: (K, N) — K trajectories drawn under the predictive joint.
        y_true : (N,)   — ground-truth target vector.

    Returns:
        Scalar Energy Score (lower is better).
    """
    term1 = np.linalg.norm(samples - y_true[None, :], axis=1).mean()
    term2 = cdist(samples, samples).mean()
    return float(term1 - 0.5 * term2)
