"""TempoPFN-style structural transforms of gated feature columns, applied before the kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from copula_inter.type_aliases import Device, HasDataConfig

if TYPE_CHECKING:
    pass


# Structural feature-warp categories, ported from TempoPFN's offline augmentor
# but applied to the inputs x, so R_star stays exact. Two-level sampling: 2-6
# categories without replacement, then one op per category in this order.
# "seasonality" keeps only amplitude_modulation; "analytic" is {smooth, first
# derivative, second derivative, integral}.
_STRUCTURAL_CATEGORIES: List[str] = [
    "invariances",
    "structure",
    "seasonality",
    "artifacts",
    "analytic",
    "discrete",
]


_CATEGORY_OPS: Dict[str, List[str]] = {
    "invariances": ["yflip", "time_flip"],
    "structure": ["regime_change", "shock_recovery"],
    "seasonality": ["amplitude_modulation"],
    "artifacts": ["resample_artifact"],
    "analytic": ["differential"],
    "discrete": ["quantize", "censor"],
}


# TempoPFN's own default category weights (offline_per_sample_iid_augmentations.py).
_DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "invariances": 0.6,
    "structure": 0.6,
    "seasonality": 0.5,
    "artifacts": 0.3,
    "analytic": 0.4,
    "discrete": 0.6,
}


# Sub-op weights for categories with more than one op.
_CATEGORY_SUB_OP_WEIGHTS: Dict[str, Dict[str, float]] = {
    "discrete": {"quantize": 0.6, "censor": 0.4},
}


def _structural_warp_column(col_data: Tensor, op: str, use_index_axis: bool = False) -> Tensor:
    """Apply one structural transform to one feature column (T,).

    The pseudo-time axis is the value rank within the column, or the row index
    when use_index_axis is True.
    """
    T = col_data.shape[0]
    device = col_data.device
    std = col_data.std()
    if not torch.isfinite(std) or std <= 0:
        std = torch.ones((), device=device)

    # Ops that don't need either pseudo-time axis at all.
    if op == "yflip":
        return -col_data

    if op == "censor":
        # Elementwise clip between two random quantiles — no ordering needed.
        q_low, q_high = float(torch.rand(1)), float(torch.rand(1))
        q_low, q_high = min(q_low, q_high), max(q_low, q_high)
        sorted_vals = torch.sort(col_data).values
        lo = sorted_vals[int(q_low * (T - 1))]
        hi = sorted_vals[int(q_high * (T - 1))]
        # No-op when the two quantile indices coincide (clamping would flatten the column).
        if not torch.isfinite(hi - lo) or (hi - lo).item() <= 0:
            return col_data
        return col_data.clamp(min=lo.item(), max=hi.item())

    if op == "quantize":
        # Snap to the nearest of n_levels levels: {min, max} plus random interior points.
        lo, hi = col_data.min(), col_data.max()
        if not torch.isfinite(hi - lo) or (hi - lo).item() <= 0:
            return col_data
        n_levels = int(torch.randint(3, 11, (1,)).item())
        n_interior = max(0, n_levels - 2)
        interior = lo + (hi - lo) * torch.rand(n_interior, device=device)
        levels = torch.sort(torch.cat([lo.view(1), hi.view(1), interior])).values
        idx = torch.argmin((col_data.unsqueeze(1) - levels.unsqueeze(0)).abs(), dim=1)
        return levels[idx]

    # Remaining ops use a pseudo-time axis: value rank, or row index if use_index_axis.
    if use_index_axis:
        sort_idx = torch.arange(T, device=device)
    else:
        sort_idx = torch.argsort(col_data)
    sorted_vals = col_data[sort_idx]
    rank = torch.arange(T, device=device, dtype=torch.float32)

    if op == "time_flip":
        # Reverse along the active pseudo-time axis.
        transformed = sorted_vals.flip(dims=[0])

    elif op == "regime_change":
        min_seg = max(4, T // 16)
        valid_hi = T - min_seg
        if valid_hi <= min_seg:
            transformed = sorted_vals
        else:
            num_cp = int(torch.randint(1, 4, (1,)).item())
            valid = torch.arange(min_seg, valid_hi, device=device)
            num_cp = min(num_cp, valid.numel())
            cp = torch.sort(valid[torch.randperm(valid.numel(), device=device)[:num_cp]]).values
            boundaries = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=cp.dtype),
                    cp,
                    torch.full((1,), T, device=device, dtype=cp.dtype),
                ]
            )
            transformed = sorted_vals.clone()
            for i in range(boundaries.numel() - 1):
                s, e = int(boundaries[i]), int(boundaries[i + 1])
                if e <= s:
                    continue
                seg = sorted_vals[s:e]
                scale = float(torch.empty(1).uniform_(0.8, 1.25))
                shift = float(torch.randn(1)) * 0.15 * std.item()
                seg_mean = seg.mean()
                transformed[s:e] = (seg - seg_mean) * scale + seg_mean + shift

    elif op == "shock_recovery":
        t_lo = max(1, T // 16)
        t_hi = max(t_lo + 1, T - T // 16)
        t0 = int(torch.randint(t_lo, t_hi, (1,)).item())
        mag = float(torch.empty(1).uniform_(0.5, 2.0)) * std.item()
        if torch.rand(1).item() < 0.5:
            mag = -mag
        half_life = max(1.0, float(torch.empty(1).uniform_(0.05, 0.3)) * T)
        decay = torch.exp(-(rank - t0).clamp(min=0) / half_life)
        transformed = sorted_vals + mag * decay

    elif op == "amplitude_modulation":
        # Rescale one contiguous window's amplitude around its local mean.
        min_w = max(4, T // 16)
        max_w = max(min_w + 1, T // 2)
        win = int(torch.randint(min_w, max_w + 1, (1,)).item())
        start = int(torch.randint(0, max(1, T - win) + 1, (1,)).item())
        end = min(T, start + win)
        transformed = sorted_vals.clone()
        seg = sorted_vals[start:end]
        if seg.numel() > 0:
            seg_mean = seg.mean()
            amp = float(torch.empty(1).uniform_(0.5, 1.8))
            transformed[start:end] = (seg - seg_mean) * amp + seg_mean

    elif op == "differential":
        # Smooth, 1st derivative, 2nd derivative or cumulative integral of a box-smoothed column, rescaled to the original range.
        k = max(3, T // 32)
        k = k + 1 if k % 2 == 0 else k
        box = torch.ones(k, device=device) / k
        padded = torch.nn.functional.pad(sorted_vals.view(1, 1, -1), (k // 2, k // 2), mode="reflect")
        smoothed = torch.nn.functional.conv1d(padded, box.view(1, 1, -1)).view(-1)

        sub_op = int(torch.randint(0, 4, (1,)).item())
        if sub_op == 0:
            raw = smoothed
        elif sub_op == 1:  # first derivative
            sk = torch.tensor([-1.0, 0.0, 1.0], device=device)
            p = torch.nn.functional.pad(smoothed.view(1, 1, -1), (1, 1), mode="reflect")
            raw = torch.nn.functional.conv1d(p, sk.view(1, 1, -1)).view(-1)
        elif sub_op == 2:  # second derivative
            sk = torch.tensor([1.0, -2.0, 1.0], device=device)
            p = torch.nn.functional.pad(smoothed.view(1, 1, -1), (1, 1), mode="reflect")
            raw = torch.nn.functional.conv1d(p, sk.view(1, 1, -1)).view(-1)
        else:  # cumulative integral, running from the left or right
            if torch.rand(1).item() < 0.5:
                raw = torch.cumsum(smoothed, dim=0)
            else:
                raw = torch.flip(torch.cumsum(torch.flip(smoothed, dims=[0]), dim=0), dims=[0])

        r_min, r_max = raw.min(), raw.max()
        s_min, s_max = sorted_vals.min(), sorted_vals.max()
        # No-op when the transformed column is flat.
        if not torch.isfinite(r_max - r_min) or (r_max - r_min).item() <= 1e-8:
            transformed = sorted_vals
        else:
            denom = r_max - r_min
            transformed = (raw - r_min) / denom * (s_max - s_min) + s_min

    elif op == "resample_artifact":
        # Downsample with a random phase, then upsample (linear, step-hold or linear+smooth).
        max_factor = max(2, min(8, T // 32))
        factor = int(torch.randint(2, max_factor + 1, (1,)).item())
        offset = int(torch.randint(0, factor, (1,)).item())
        ds_idx = torch.arange(offset, T, factor, device=device)
        if ds_idx.numel() < 3:
            transformed = sorted_vals
        else:
            ds_vals_np = sorted_vals[ds_idx].detach().cpu().numpy()
            ds_idx_np = ds_idx.detach().cpu().numpy().astype(np.float64)
            rank_np = rank.detach().cpu().numpy()
            mode_idx = int(torch.multinomial(torch.tensor([0.5, 0.2, 0.3]), 1).item())
            if mode_idx == 0:  # linear
                us_np = np.interp(rank_np, ds_idx_np, ds_vals_np)
            elif mode_idx == 1:  # step-hold: forward-fill from the last downsampled point
                us_np = ds_vals_np[np.searchsorted(ds_idx_np, rank_np, side="right") - 1]
            else:  # linear + light smoothing
                us_np = np.interp(rank_np, ds_idx_np, ds_vals_np)
                sm_k = max(3, T // 128)
                sm_kernel = np.ones(sm_k) / sm_k
                us_np = np.convolve(us_np, sm_kernel, mode="same")
            transformed = torch.from_numpy(us_np).to(device=device, dtype=sorted_vals.dtype)

    else:
        raise ValueError(f"Unknown structural warp op '{op}'")

    warped = torch.empty_like(col_data)
    warped[sort_idx] = transformed
    return warped


def _sample_structural_ops(category_weights: Dict[str, float], num_ops_min: int, num_ops_max: int) -> List[str]:
    """Draw 2..6 categories without replacement (weighted), then one op per category in canonical order."""
    eligible = [c for c in _STRUCTURAL_CATEGORIES if category_weights.get(c, 0.0) > 0.0]
    if not eligible:
        return []
    k = min(int(torch.randint(num_ops_min, num_ops_max + 1, (1,)).item()), len(eligible))
    weights = torch.tensor([category_weights[c] for c in eligible], dtype=torch.float32)
    weights = weights / weights.sum()
    idx = torch.multinomial(weights, k, replacement=False)
    chosen_categories = {eligible[i] for i in idx.tolist()}

    ops: List[str] = []
    for category in _STRUCTURAL_CATEGORIES:  # fixed canonical order, not draw order
        if category not in chosen_categories:
            continue
        candidates = _CATEGORY_OPS[category]
        if len(candidates) == 1:
            ops.append(candidates[0])
            continue
        sub_weights_map = _CATEGORY_SUB_OP_WEIGHTS.get(category)
        if sub_weights_map is None:
            sub_weights = torch.ones(len(candidates))
        else:
            sub_weights = torch.tensor([sub_weights_map[c] for c in candidates], dtype=torch.float32)
        sub_weights = sub_weights / sub_weights.sum()
        pick = int(torch.multinomial(sub_weights, 1).item())
        ops.append(candidates[pick])
    return ops


def _sample_structural_category_mask(
    M: int,
    category_weights: Dict[str, float],
    num_ops_min: int,
    num_ops_max: int,
    device: Device,
) -> Tuple[Tensor, List[str]]:
    """Batched category selection for M draws (Gumbel top-k).

    Each row picks k in [num_ops_min, num_ops_max] categories without replacement
    from the non-zero-weight ones.

    Returns:
        (chosen_mask (M, len(eligible)) bool, eligible category list).
    """
    eligible = [c for c in _STRUCTURAL_CATEGORIES if category_weights.get(c, 0.0) > 0.0]
    n_elig = len(eligible)
    if n_elig == 0:
        return torch.zeros(M, 0, dtype=torch.bool, device=device), eligible

    weights = torch.tensor([category_weights[c] for c in eligible], dtype=torch.float32, device=device)
    log_w = torch.log(weights / weights.sum())
    u = torch.rand(M, n_elig, device=device).clamp_min(1e-12)
    gumbel = -torch.log((-torch.log(u)).clamp_min(1e-12))
    scores = log_w.unsqueeze(0) + gumbel
    # rank[m, i] = position of category i in row m's Gumbel-perturbed order.
    rank = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)

    k = torch.randint(num_ops_min, num_ops_max + 1, (M,), device=device)
    k = torch.clamp(k, max=n_elig)
    chosen_mask = rank < k.unsqueeze(1)
    return chosen_mask, eligible


def _structural_warp_batch(col_data: Tensor, op: str, use_index_axis: Tensor) -> Tensor:
    """Batched _structural_warp_column: apply op to each row of col_data (M, T).

    use_index_axis is (M,) bool. "resample_artifact" loops over rows with
    _structural_warp_column.
    """
    M, T = col_data.shape
    device = col_data.device

    if op == "resample_artifact":
        out = torch.empty_like(col_data)
        for i in range(M):
            out[i] = _structural_warp_column(col_data[i], op, use_index_axis=bool(use_index_axis[i]))
        return out

    std = col_data.std(dim=1)
    std = torch.where(torch.isfinite(std) & (std > 0), std, torch.ones_like(std))

    if op == "yflip":
        return -col_data

    if op == "censor":
        q = torch.rand(M, 2, device=device)
        q_low = q.min(dim=1).values
        q_high = q.max(dim=1).values
        sorted_vals, _ = torch.sort(col_data, dim=1)
        lo_idx = (q_low * (T - 1)).long().clamp(0, T - 1)
        hi_idx = (q_high * (T - 1)).long().clamp(0, T - 1)
        lo = torch.gather(sorted_vals, 1, lo_idx.unsqueeze(1)).squeeze(1)
        hi = torch.gather(sorted_vals, 1, hi_idx.unsqueeze(1)).squeeze(1)
        # Coinciding quantile indices: use +-inf bounds so the clamp is a no-op.
        degenerate = ~torch.isfinite(hi - lo) | ((hi - lo) <= 0)
        lo_eff = torch.where(degenerate, torch.full_like(lo, float("-inf")), lo)
        hi_eff = torch.where(degenerate, torch.full_like(hi, float("inf")), hi)
        return torch.clamp(col_data, min=lo_eff.unsqueeze(1), max=hi_eff.unsqueeze(1))

    if op == "quantize":
        lo = col_data.min(dim=1).values
        hi = col_data.max(dim=1).values
        degenerate = ~torch.isfinite(hi - lo) | ((hi - lo) <= 0)
        max_interior = 8  # n_levels in [3,10] -> n_interior in [1,8]
        n_levels = torch.randint(3, 11, (M,), device=device)
        n_interior = (n_levels - 2).clamp(min=0)
        interior_raw = torch.rand(M, max_interior, device=device)
        interior = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * interior_raw
        slot_idx = torch.arange(max_interior, device=device).unsqueeze(0)
        interior_mask = slot_idx < n_interior.unsqueeze(1)
        # Padding levels are +inf so they never win the nearest-level argmin.
        interior = torch.where(interior_mask, interior, torch.full_like(interior, float("inf")))
        levels = torch.cat([lo.unsqueeze(1), hi.unsqueeze(1), interior], dim=1)
        levels, _ = torch.sort(levels, dim=1)
        diff = (col_data.unsqueeze(2) - levels.unsqueeze(1)).abs()
        idx = diff.argmin(dim=2)
        quantized = torch.gather(levels, 1, idx)
        return torch.where(degenerate.unsqueeze(1), col_data, quantized)

    # Pseudo-time axis per row: value rank, or row index if use_index_axis.
    argsort_idx = torch.argsort(col_data, dim=1)
    index_idx = torch.arange(T, device=device).unsqueeze(0).expand(M, T)
    sort_idx = torch.where(use_index_axis.unsqueeze(1), index_idx, argsort_idx)
    sorted_vals = torch.gather(col_data, 1, sort_idx)
    rank = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(M, T)

    if op == "time_flip":
        transformed = sorted_vals.flip(dims=[1])

    elif op == "regime_change":
        min_seg = max(4, T // 16)
        valid_hi = T - min_seg
        if valid_hi <= min_seg:
            transformed = sorted_vals
        else:
            # Each row draws up to 3 changepoints from the shared candidate range.
            valid = torch.arange(min_seg, valid_hi, device=device)
            n_valid = valid.numel()
            max_cp = 3
            num_cp = torch.randint(1, 4, (M,), device=device).clamp(max=n_valid)
            keys = torch.rand(M, n_valid, device=device)
            take = min(max_cp, n_valid)
            order = torch.argsort(keys, dim=1)[:, :take]  # (M, take)
            if take < max_cp:
                # Fewer candidates than slots: pad with 0 (masked out below).
                pad = torch.zeros(M, max_cp - take, dtype=order.dtype, device=device)
                order = torch.cat([order, pad], dim=1)
            chosen_pos = valid[order]  # (M, max_cp)
            slot_idx = torch.arange(max_cp, device=device).unsqueeze(0)
            valid_mask = slot_idx < num_cp.unsqueeze(1)
            # Unused slots are T, giving empty trailing segments.
            chosen_pos = torch.where(valid_mask, chosen_pos, torch.full_like(chosen_pos, T))
            cp_sorted, _ = torch.sort(chosen_pos, dim=1)
            boundaries = torch.cat(
                [
                    torch.zeros(M, 1, dtype=cp_sorted.dtype, device=device),
                    cp_sorted,
                    torch.full((M, 1), T, dtype=cp_sorted.dtype, device=device),
                ],
                dim=1,
            )
            pos = torch.arange(T, device=device).unsqueeze(0)
            transformed = sorted_vals.clone()
            for i in range(boundaries.shape[1] - 1):
                s = boundaries[:, i].unsqueeze(1)
                e = boundaries[:, i + 1].unsqueeze(1)
                in_seg = (pos >= s) & (pos < e)
                seg_count = in_seg.sum(dim=1).clamp(min=1)
                seg_mean = (sorted_vals * in_seg).sum(dim=1) / seg_count
                scale = torch.empty(M, device=device).uniform_(0.8, 1.25)
                shift = torch.randn(M, device=device) * 0.15 * std
                new_vals = (
                    (sorted_vals - seg_mean.unsqueeze(1)) * scale.unsqueeze(1)
                    + seg_mean.unsqueeze(1)
                    + shift.unsqueeze(1)
                )
                transformed = torch.where(in_seg, new_vals, transformed)

    elif op == "shock_recovery":
        lo_t = max(1, T // 16)
        hi_t = max(lo_t + 1, T - T // 16)
        t0 = torch.randint(lo_t, hi_t, (M,), device=device).float()
        mag_abs = torch.empty(M, device=device).uniform_(0.5, 2.0) * std
        sign = torch.where(torch.rand(M, device=device) < 0.5, -1.0, 1.0)
        mag = mag_abs * sign
        half_life = torch.empty(M, device=device).uniform_(0.05, 0.3) * T
        half_life = half_life.clamp(min=1.0)
        decay = torch.exp(-(rank - t0.unsqueeze(1)).clamp(min=0) / half_life.unsqueeze(1))
        transformed = sorted_vals + mag.unsqueeze(1) * decay

    elif op == "amplitude_modulation":
        min_w = max(4, T // 16)
        max_w = max(min_w + 1, T // 2)
        win = torch.randint(min_w, max_w + 1, (M,), device=device)
        span = torch.clamp(T - win, min=1)  # mirrors max(1, T - win)
        start = (torch.rand(M, device=device) * (span + 1).float()).floor().long().clamp(max=span)
        end = (start + win).clamp(max=T)
        pos = torch.arange(T, device=device).unsqueeze(0)
        in_seg = (pos >= start.unsqueeze(1)) & (pos < end.unsqueeze(1))
        seg_count = in_seg.sum(dim=1).clamp(min=1)
        seg_mean = (sorted_vals * in_seg).sum(dim=1) / seg_count
        amp = torch.empty(M, device=device).uniform_(0.5, 1.8)
        new_vals = (sorted_vals - seg_mean.unsqueeze(1)) * amp.unsqueeze(1) + seg_mean.unsqueeze(1)
        transformed = torch.where(in_seg, new_vals, sorted_vals)

    elif op == "differential":
        # Box average and 3-tap derivatives via slicing/cumsum (same result as conv1d, faster here).
        k = max(3, T // 32)
        k = k + 1 if k % 2 == 0 else k
        pad_k = k // 2
        padded = torch.nn.functional.pad(sorted_vals.unsqueeze(1), (pad_k, pad_k), mode="reflect").squeeze(1)
        csum = torch.nn.functional.pad(padded.cumsum(dim=1), (1, 0))  # csum[:,0] = 0
        smoothed = (csum[:, k:] - csum[:, :-k]) / k  # k-wide windowed mean, (M, T)

        # Compute all 4 sub-ops for the batch, then gather each row's choice.
        p1 = torch.nn.functional.pad(smoothed.unsqueeze(1), (1, 1), mode="reflect").squeeze(1)  # (M, T+2)
        raw_d1 = p1[:, 2:] - p1[:, :-2]
        raw_d2 = p1[:, :-2] - 2 * p1[:, 1:-1] + p1[:, 2:]
        int_fwd = torch.cumsum(smoothed, dim=1)
        int_bwd = torch.flip(torch.cumsum(torch.flip(smoothed, dims=[1]), dim=1), dims=[1])
        int_dir = torch.rand(M, 1, device=device) < 0.5
        raw_int = torch.where(int_dir, int_fwd, int_bwd)

        candidates = torch.stack([smoothed, raw_d1, raw_d2, raw_int], dim=1)  # (M, 4, T)
        sub_op = torch.randint(0, 4, (M,), device=device)
        raw = torch.gather(candidates, 1, sub_op.view(M, 1, 1).expand(-1, 1, T)).squeeze(1)

        r_min = raw.min(dim=1).values
        r_max = raw.max(dim=1).values
        s_min = sorted_vals.min(dim=1).values
        s_max = sorted_vals.max(dim=1).values
        degenerate = ~torch.isfinite(r_max - r_min) | ((r_max - r_min) <= 1e-8)
        denom = torch.where(degenerate, torch.ones_like(r_max), r_max - r_min)
        rescaled = (raw - r_min.unsqueeze(1)) / denom.unsqueeze(1) * (s_max - s_min).unsqueeze(1) + s_min.unsqueeze(1)
        transformed = torch.where(degenerate.unsqueeze(1), sorted_vals, rescaled)

    else:
        raise ValueError(f"Unknown structural warp op '{op}'")

    warped = torch.empty_like(col_data)
    warped.scatter_(1, sort_idx, transformed)
    return warped


def apply_structural_feature_warp(x: Tensor, cfg: HasDataConfig, device: Device) -> Tensor:
    """Apply TempoPFN-style structural transforms to gated feature columns, per episode.

    Each column is gated with structural_warp_prob; gated columns get one op from
    each of structural_warp_num_ops_min..max categories (weighted by
    structural_warp_category_weights), applied in canonical order. Order-dependent
    ops use value rank, or the row index with probability
    structural_warp_index_axis_ratio (chosen once per column).

    Args:
        x: (B, T, d) inputs.
        cfg: config; reads cfg.data.structural_warp_* (disabled by default).
        device: unused.

    Returns:
        (B, T, d) tensor.
    """
    if not bool(getattr(cfg.data, "structural_warp_enabled", False)):
        return x

    prob = float(getattr(cfg.data, "structural_warp_prob", 0.3))
    if prob <= 0.0:
        return x

    category_weights = dict(getattr(cfg.data, "structural_warp_category_weights", _DEFAULT_CATEGORY_WEIGHTS))

    num_ops_max = int(getattr(cfg.data, "structural_warp_num_ops_max", 6))
    num_ops_max = max(1, min(num_ops_max, len(_STRUCTURAL_CATEGORIES)))
    num_ops_min = int(getattr(cfg.data, "structural_warp_num_ops_min", 2))
    num_ops_min = max(1, min(num_ops_min, num_ops_max))

    index_axis_enabled = bool(getattr(cfg.data, "structural_warp_index_axis_enabled", False))
    index_axis_ratio = float(getattr(cfg.data, "structural_warp_index_axis_ratio", 0.0))

    B, T, d = x.shape
    dev = x.device

    # Per-(episode, column) gate in one draw.
    gate = torch.rand(B, d, device=dev) < prob
    gated_idx = gate.reshape(-1).nonzero(as_tuple=True)[0]
    if gated_idx.numel() == 0:
        return x.clone()  # matches the original's unconditional x.clone() up front

    if index_axis_enabled:
        axis_gate = torch.rand(B, d, device=dev) < index_axis_ratio
    else:
        axis_gate = torch.zeros(B, d, dtype=torch.bool, device=dev)

    # (B, T, d) -> (B*d, T); row b*d + col is column (b, col).
    flat = x.permute(0, 2, 1).reshape(B * d, T).clone()
    gated_cols = flat[gated_idx]
    use_index_axis = axis_gate.reshape(-1)[gated_idx]

    M = gated_idx.numel()
    chosen_mask, eligible = _sample_structural_category_mask(M, category_weights, num_ops_min, num_ops_max, dev)

    # Loop over categories, each applied to the gated columns that chose it.
    for category in _STRUCTURAL_CATEGORIES:
        if category not in eligible:
            continue
        cat_col = eligible.index(category)
        cat_mask = chosen_mask[:, cat_col]
        if not bool(cat_mask.any()):
            continue
        sel = cat_mask.nonzero(as_tuple=True)[0]
        candidates = _CATEGORY_OPS[category]
        if len(candidates) == 1:
            op = candidates[0]
            gated_cols[sel] = _structural_warp_batch(gated_cols[sel], op, use_index_axis[sel])
        else:
            sub_weights_map = _CATEGORY_SUB_OP_WEIGHTS.get(category)
            if sub_weights_map is None:
                sub_w = torch.ones(len(candidates))
            else:
                sub_w = torch.tensor([sub_weights_map[c] for c in candidates], dtype=torch.float32)
            sub_w = sub_w / sub_w.sum()
            picks = torch.multinomial(sub_w, sel.numel(), replacement=True)
            for k_op, op in enumerate(candidates):
                op_sel = sel[picks == k_op]
                if op_sel.numel() == 0:
                    continue
                gated_cols[op_sel] = _structural_warp_batch(gated_cols[op_sel], op, use_index_axis[op_sel])

    flat[gated_idx] = gated_cols
    warped_x = flat.reshape(B, d, T).permute(0, 2, 1).contiguous()
    return warped_x
