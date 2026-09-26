"""Episode schemas and boundary checks. Training's tensor step stays validation-free."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from torch import Tensor


class RawEpisode(TypedDict):
    x_norm_train: Tensor  # (P, D), same normalization as x_norm_test
    x_norm_test: Tensor  # (N, D)
    y_train: Tensor  # (P,)
    y_test: Tensor  # (N,)
    n_train: Tensor
    n_test: Tensor
    R_star: Tensor  # (N, N), prior oracle when available
    mu_star: Tensor  # (N,)
    sigma_star: Tensor  # (N,)
    x_kernel_train: NotRequired[Tensor]  # pre-normalization kernel space (P, Dk)
    x_kernel_test: NotRequired[Tensor]  # same kernel space (N, Dk)


class PITResult(TypedDict):
    z_train: Tensor  # (P,)
    z_test: Tensor  # (N,)
    log_pdf_test: Tensor  # (N,), density on original y scale


class PaddedBatch(TypedDict):
    x_train: Tensor
    x_test: Tensor
    y_train: Tensor
    y_test: Tensor
    z_train: Tensor
    z_test: Tensor
    log_pdf_test: Tensor
    train_mask: Tensor
    test_mask: Tensor
    R_star: Tensor
    Sigma_star: Tensor
    mu_star: Tensor
    sigma_star: Tensor
    n_train: Tensor
    n_test: Tensor
    R_prior: NotRequired[Tensor]


def validate_episode(ep: RawEpisode | dict) -> None:
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
