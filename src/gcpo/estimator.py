# Copyright 2026 GCPO Authors
# Licensed under the Apache License, Version 2.0 (the "License").

"""Paper-faithful GCPO credit assignment for an external semantic encoder."""

from collections import defaultdict
import math

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


def _all_subset_masks(n_players: int, device: torch.device) -> torch.Tensor:
    subset_ids = torch.arange(1 << n_players, device=device, dtype=torch.long)
    player_ids = torch.arange(n_players, device=device, dtype=torch.long)
    return ((subset_ids.unsqueeze(1) >> player_ids.unsqueeze(0)) & 1).bool()


def _exact_shapley(kernel: torch.Tensor, eta: float) -> torch.Tensor:
    """Compute exact Shapley values for v(S) = log det(I + eta * L_S)."""
    n_players = kernel.shape[0]
    masks = _all_subset_masks(n_players, kernel.device)
    mask_values = masks.to(dtype=kernel.dtype)
    masked_kernels = (
        eta
        * kernel.unsqueeze(0)
        * mask_values.unsqueeze(-1)
        * mask_values.unsqueeze(1)
    )
    identity = torch.eye(
        n_players, device=kernel.device, dtype=kernel.dtype
    ).unsqueeze(0)
    signs, coalition_values = torch.linalg.slogdet(identity + masked_kernels)
    if not torch.all(signs > 0):
        raise RuntimeError("GCPO encountered a non-positive DPP determinant")

    subset_sizes = masks.sum(dim=1)
    weights = torch.zeros(n_players + 1, device=kernel.device, dtype=kernel.dtype)
    for subset_size in range(n_players):
        weights[subset_size] = (
            math.factorial(subset_size)
            * math.factorial(n_players - subset_size - 1)
            / math.factorial(n_players)
        )

    credits = torch.zeros(n_players, device=kernel.device, dtype=kernel.dtype)
    for player in range(n_players):
        subsets_without_player = torch.where(~masks[:, player])[0]
        subsets_with_player = subsets_without_player + (1 << player)
        marginal_values = (
            coalition_values[subsets_with_player]
            - coalition_values[subsets_without_player]
        )
        credits[player] = torch.sum(
            weights[subset_sizes[subsets_without_player]] * marginal_values
        )
    return credits


def _permutation_shapley(
    kernel: torch.Tensor,
    eta: float,
    n_samples: int,
    epsilon: float,
) -> torch.Tensor:
    """Estimate Shapley values using permutation sampling and Cholesky updates."""
    n_players = kernel.shape[0]
    credits = torch.zeros(n_players, device=kernel.device, dtype=kernel.dtype)

    for _ in range(n_samples):
        permutation = torch.randperm(n_players, device=kernel.device)
        ordered_players: list[int] = []
        cholesky = torch.empty((0, 0), device=kernel.device, dtype=kernel.dtype)

        for player_tensor in permutation:
            player = int(player_tensor.item())
            diagonal = 1.0 + eta * kernel[player, player]
            if ordered_players:
                cross = eta * kernel[ordered_players, player]
                solved = torch.linalg.solve_triangular(
                    cholesky, cross.unsqueeze(-1), upper=False
                ).squeeze(-1)
                schur_complement = diagonal - torch.dot(solved, solved)
            else:
                solved = torch.empty(0, device=kernel.device, dtype=kernel.dtype)
                schur_complement = diagonal

            schur_complement = torch.clamp(schur_complement, min=epsilon)
            credits[player] += torch.log(schur_complement)

            old_size = len(ordered_players)
            updated = torch.zeros(
                (old_size + 1, old_size + 1),
                device=kernel.device,
                dtype=kernel.dtype,
            )
            if old_size:
                updated[:old_size, :old_size] = cholesky
                updated[old_size, :old_size] = solved
            updated[old_size, old_size] = torch.sqrt(schur_complement)
            cholesky = updated
            ordered_players.append(player)

    return credits / n_samples


def _validate_inputs(
    scores: torch.Tensor,
    response_embeddings: torch.Tensor,
    eta: float,
    exact_threshold: int,
    n_samples: int,
    epsilon: float,
) -> None:
    if scores.ndim != 1:
        raise ValueError(
            f"GCPO scores must be one-dimensional, got shape {tuple(scores.shape)}"
        )
    if response_embeddings.ndim != 2:
        raise ValueError(
            "GCPO response_embeddings must be a two-dimensional tensor, "
            f"got shape {tuple(response_embeddings.shape)}"
        )
    if response_embeddings.shape[0] != scores.shape[0]:
        raise ValueError(
            "GCPO scores and response_embeddings must have the same batch dimension"
        )
    if eta <= 0:
        raise ValueError(f"gcpo_eta must be positive, got {eta}")
    if exact_threshold < 0:
        raise ValueError(
            f"gcpo_exact_threshold must be non-negative, got {exact_threshold}"
        )
    if n_samples <= 0:
        raise ValueError(f"gcpo_n_samples must be positive, got {n_samples}")
    if epsilon <= 0:
        raise ValueError(f"gcpo_epsilon must be positive, got {epsilon}")


def compute_gcpo_shapley_values(
    scores: torch.Tensor,
    response_embeddings: torch.Tensor,
    eta: float = 1.0,
    exact_threshold: int = 8,
    n_samples: int = 500,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Assign DPP utility to correct rollouts with Shapley values.

    Verifier scores are binarized at zero. Only correct rollouts participate in
    the cooperative game, so incorrect responses are exact null players.
    """
    _validate_inputs(
        scores, response_embeddings, eta, exact_threshold, n_samples, epsilon
    )
    output_dtype = scores.dtype if torch.is_floating_point(scores) else torch.float32

    with torch.no_grad():
        scores = scores.detach()
        binary_scores = (scores > 0).to(dtype=output_dtype)
        embeddings = response_embeddings.detach().to(device=scores.device)
        positive_indices = torch.where(binary_scores > 0)[0]
        if positive_indices.numel() == 0:
            return torch.zeros(scores.shape, device=scores.device, dtype=output_dtype)

        computation_dtype = (
            torch.float64
            if output_dtype == torch.float64 or embeddings.dtype == torch.float64
            else torch.float32
        )
        positive_embeddings = embeddings[positive_indices].to(dtype=computation_dtype)
        if not torch.all(torch.isfinite(positive_embeddings)):
            raise ValueError("GCPO response_embeddings must contain only finite values")

        norms = torch.linalg.vector_norm(positive_embeddings, dim=-1, keepdim=True)
        if torch.any(norms <= epsilon):
            raise ValueError("GCPO response_embeddings must be non-zero")
        normalized_embeddings = positive_embeddings / norms
        weighted_embeddings = (
            binary_scores[positive_indices].to(computation_dtype).unsqueeze(-1)
            * normalized_embeddings
        )
        kernel = weighted_embeddings @ weighted_embeddings.T
        kernel = (kernel + kernel.T) / 2.0

        if positive_indices.numel() <= exact_threshold:
            positive_credits = _exact_shapley(kernel, eta)
        else:
            positive_credits = _permutation_shapley(
                kernel, eta=eta, n_samples=n_samples, epsilon=epsilon
            )

        credits = torch.zeros(
            scores.shape, device=scores.device, dtype=computation_dtype
        )
        credits[positive_indices] = positive_credits
        return credits.to(dtype=output_dtype)


def compute_gcpo_redistributed_rewards(
    scores: torch.Tensor,
    response_embeddings: torch.Tensor,
    eta: float = 1.0,
    exact_threshold: int = 8,
    n_samples: int = 500,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Redistribute a prompt group's binary reward budget by GCPO credit."""
    output_dtype = scores.dtype if torch.is_floating_point(scores) else torch.float32
    binary_scores = (scores > 0).to(dtype=output_dtype)
    positive_mask = binary_scores > 0
    if not torch.any(positive_mask):
        return torch.zeros(scores.shape, device=scores.device, dtype=output_dtype)

    credits = compute_gcpo_shapley_values(
        scores=binary_scores,
        response_embeddings=response_embeddings,
        eta=eta,
        exact_threshold=exact_threshold,
        n_samples=n_samples,
        epsilon=epsilon,
    )
    total_credit = credits[positive_mask].sum()
    redistributed = torch.zeros_like(binary_scores)
    if not torch.isfinite(total_credit) or total_credit <= 0:
        raise RuntimeError("GCPO produced a non-positive total cooperative credit")
    redistributed[positive_mask] = (
        binary_scores.sum() * credits[positive_mask] / total_credit
    )
    return redistributed


@register_adv_est("gcpo")
def compute_gcpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    response_embeddings: torch.Tensor | None = None,
    eta: float = 1.0,
    exact_threshold: int = 8,
    n_samples: int = 500,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token-level GCPO advantages from external response embeddings."""
    if response_embeddings is None:
        raise ValueError(
            "GCPO requires response_embeddings from an external sentence-transformer"
        )
    if not norm_adv_by_std_in_grpo:
        raise ValueError(
            "Paper-faithful GCPO requires algorithm.norm_adv_by_std_in_grpo=true"
        )
    if index is None:
        raise ValueError("GCPO requires a group index for every rollout")
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            "GCPO token_level_rewards and response_mask must have equal shapes"
        )
    if response_embeddings.shape[0] != token_level_rewards.shape[0]:
        raise ValueError(
            "GCPO response_embeddings and token_level_rewards must have equal batch sizes"
        )
    if len(index) != token_level_rewards.shape[0]:
        raise ValueError("GCPO index and token_level_rewards must have equal batch sizes")

    if config is not None:
        eta = config.get("gcpo_eta", eta)
        exact_threshold = config.get("gcpo_exact_threshold", exact_threshold)
        n_samples = config.get("gcpo_n_samples", n_samples)
        epsilon = config.get("gcpo_epsilon", epsilon)

    response_length = token_level_rewards.shape[-1]
    with torch.no_grad():
        scores = (token_level_rewards * response_mask).sum(dim=-1)
        group_to_indices: dict[object, list[int]] = defaultdict(list)
        for batch_index, uid in enumerate(index):
            if isinstance(uid, torch.Tensor):
                uid = uid.item()
            elif isinstance(uid, np.generic):
                uid = uid.item()
            group_to_indices[uid].append(batch_index)

        sequence_advantages = torch.zeros_like(scores)
        for group_indices in group_to_indices.values():
            if len(group_indices) <= 1:
                continue
            redistributed = compute_gcpo_redistributed_rewards(
                scores=scores[group_indices],
                response_embeddings=response_embeddings[group_indices],
                eta=eta,
                exact_threshold=exact_threshold,
                n_samples=n_samples,
                epsilon=epsilon,
            )
            centered = redistributed - redistributed.mean()
            reward_std = redistributed.std()
            if torch.isfinite(reward_std) and reward_std > epsilon:
                centered = centered / (reward_std + epsilon)
            sequence_advantages[group_indices] = centered

        advantages = (
            sequence_advantages.unsqueeze(-1).tile([1, response_length])
            * response_mask
        )
    return advantages, advantages
