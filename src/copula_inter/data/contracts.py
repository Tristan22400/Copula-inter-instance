"""Episode schemas and boundary checks. Training's tensor step stays validation-free."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor


def validate_episode(ep: Mapping[str, Any]) -> None:
    """Check row/coordinate alignment once at persistence or collation."""
    try:
        p, n = int(ep["n_train"]), int(ep["n_test"])
        d = ep["x_norm_train"].shape[1]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("episode needs n_train, n_test and 2-D x_norm_train") from exc
    if p < 1 or n < 1 or d < 1:
        raise ValueError(f"episode dimensions must be positive: P={p}, N={n}, D={d}")
    shapes = {
        "x_norm_train": (p, d),
        "x_norm_test": (n, d),
        "y_train": (p,),
        "y_test": (n,),
        "z_train": (p,),
        "z_test": (n,),
        "log_pdf_test": (n,),
        "R_star": (n, n),
        "mu_star": (n,),
        "sigma_star": (n,),
    }
    optional = {
        "R_prior": (n, n),
        "Sigma_star": (n, n),
    }
    for key, shape in shapes.items():
        if key not in ep or tuple(ep[key].shape) != shape:
            raise ValueError(f"episode {key} must have shape {shape}")
    for key, shape in optional.items():
        if key in ep and tuple(ep[key].shape) != shape:
            raise ValueError(f"episode {key} must have shape {shape}")
    if ("x_kernel_train" in ep) != ("x_kernel_test" in ep):
        raise ValueError("raw kernel train/test coordinates must occur together")
    if "x_kernel_train" in ep:
        dk = ep["x_kernel_train"].shape[-1]
        if tuple(ep["x_kernel_train"].shape) != (p, dk) or tuple(ep["x_kernel_test"].shape) != (n, dk):
            raise ValueError("raw kernel coordinates must align with P/N and share feature width")


def assemble_episodes(
    tensors: dict[str, Tensor],
    extra: dict,
    discard: Tensor,
    chain_metadata: tuple[list[str], list[str], list[dict[str, Tensor]]] | None = None,
) -> list[dict]:
    """Apply the validity mask and attach per-episode and shared metadata."""
    keep = ~discard
    count = int(keep.sum())
    # Generation mixes CPU-resident fields with device-resident Cholesky
    # factors. Move the mask to each tensor instead of moving whole fields.
    fields = {key: val[keep.to(val.device)] for key, val in tensors.items()}
    episodes = [{key: val[b] for key, val in fields.items()} | extra for b in range(count)]
    if chain_metadata is not None:
        names, ops, components = chain_metadata
        filtered = [{key: val[keep.to(val.device)] for key, val in component.items()} for component in components]
        for b, episode in enumerate(episodes):
            episode["kernel_components"] = names
            episode["kernel_ops"] = ops
            episode["kernel_component_params"] = [
                {key: val[b].cpu() for key, val in component.items()} for component in filtered
            ]
    return episodes
