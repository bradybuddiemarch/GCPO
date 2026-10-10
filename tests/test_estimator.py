import itertools
import math

import numpy as np
import pytest
import torch

from gcpo.estimator import (
    compute_gcpo_advantage,
    compute_gcpo_redistributed_rewards,
    compute_gcpo_shapley_values,
)
from verl.trainer.ppo.core_algos import compute_policy_loss, get_adv_estimator_fn


def _coalition_value(kernel, coalition, eta=1.0):
    if not coalition:
        return torch.zeros((), dtype=kernel.dtype)
    indices = torch.tensor(coalition, dtype=torch.long)
    submatrix = kernel[indices][:, indices]
    identity = torch.eye(len(coalition), dtype=kernel.dtype)
    return torch.linalg.slogdet(identity + eta * submatrix).logabsdet


def _brute_force_permutation_shapley(embeddings, eta=1.0):
    normalized = embeddings / torch.linalg.vector_norm(
        embeddings, dim=-1, keepdim=True
    )
    kernel = normalized @ normalized.T
    result = torch.zeros(embeddings.shape[0], dtype=embeddings.dtype)
    permutations = list(itertools.permutations(range(embeddings.shape[0])))

    for permutation in permutations:
        coalition = []
        previous = torch.zeros((), dtype=embeddings.dtype)
        for player in permutation:
            coalition.append(player)
            current = _coalition_value(kernel, coalition, eta)
            result[player] += current - previous
            previous = current
    return result / len(permutations)


@pytest.mark.parametrize("n_players", range(1, 7))
def test_exact_shapley_matches_brute_force_permutations(n_players):
    generator = torch.Generator().manual_seed(100 + n_players)
    embeddings = torch.randn(n_players, 4, generator=generator, dtype=torch.float64)
    scores = torch.ones(n_players, dtype=torch.float64)

    actual = compute_gcpo_shapley_values(scores, embeddings, exact_threshold=8)
    expected = _brute_force_permutation_shapley(embeddings)

    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)


def test_redistribution_rewards_a_novel_correct_rollout_more():
    scores = torch.ones(3)
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    redistributed = compute_gcpo_redistributed_rewards(scores, embeddings)

    expected = torch.tensor([0.9197208, 1.1605585, 0.9197208])
    torch.testing.assert_close(redistributed, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(redistributed.sum(), scores.sum())


def test_zero_reward_rollouts_are_null_players():
    scores = torch.tensor([1.0, 0.0, 1.0])
    embeddings = torch.tensor([[1.0, 0.0], [50.0, -100.0], [0.0, 1.0]])

    redistributed = compute_gcpo_redistributed_rewards(scores, embeddings)

    torch.testing.assert_close(redistributed, torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(redistributed.sum(), torch.tensor(2.0))


def test_all_non_positive_rewards_receive_zero_credit():
    scores = torch.tensor([0.0, -1.0, 0.0])
    embeddings = torch.randn(3, 4)

    redistributed = compute_gcpo_redistributed_rewards(scores, embeddings)

    torch.testing.assert_close(redistributed, torch.zeros_like(scores))


def test_signed_verifier_scores_are_mapped_to_the_binary_paper_reward():
    scores = torch.tensor([1.0, -1.0, -1.0, -1.0])

    redistributed = compute_gcpo_redistributed_rewards(scores, torch.eye(4))

    torch.testing.assert_close(redistributed, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_embedding_scale_does_not_change_credits():
    scores = torch.ones(3)
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    scaled = embeddings * torch.tensor([[3.0], [7.0], [11.0]])

    actual = compute_gcpo_redistributed_rewards(scores, scaled)
    expected = compute_gcpo_redistributed_rewards(scores, embeddings)

    torch.testing.assert_close(actual, expected)


def test_permutation_estimator_preserves_the_grand_coalition_value():
    scores = torch.ones(3)

    credits = compute_gcpo_shapley_values(
        scores,
        torch.eye(3),
        exact_threshold=2,
        n_samples=5,
    )

    torch.testing.assert_close(
        credits.sum(), torch.tensor(3.0 * math.log(2.0)), rtol=1e-5, atol=1e-6
    )
    assert torch.all(credits >= 0)


def test_advantage_groups_standardizes_and_masks_rewards():
    token_rewards = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    response_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    advantages, returns = compute_gcpo_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.array(["prompt", "prompt", "prompt"], dtype=object),
        response_embeddings=embeddings,
    )

    sequence_advantages = torch.tensor([-0.5773461, 1.1546922, -0.5773461])
    expected = sequence_advantages.unsqueeze(1) * response_mask
    torch.testing.assert_close(advantages, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(returns, expected, rtol=1e-5, atol=2e-6)


def test_singleton_groups_receive_zero_advantage():
    advantages, _ = compute_gcpo_advantage(
        token_level_rewards=torch.ones(2, 1),
        response_mask=torch.ones(2, 1),
        index=np.array(["first", "second"], dtype=object),
        response_embeddings=torch.eye(2),
    )

    torch.testing.assert_close(advantages, torch.zeros_like(advantages))


def test_advantage_requires_external_embeddings_and_paper_normalization():
    kwargs = {
        "token_level_rewards": torch.ones(2, 1),
        "response_mask": torch.ones(2, 1),
        "index": np.array([0, 0]),
    }
    with pytest.raises(ValueError, match="external sentence-transformer"):
        compute_gcpo_advantage(**kwargs)
    with pytest.raises(ValueError, match="norm_adv_by_std_in_grpo=true"):
        compute_gcpo_advantage(
            **kwargs,
            response_embeddings=torch.eye(2),
            norm_adv_by_std_in_grpo=False,
        )


def test_gcpo_is_registered_by_string_name():
    assert get_adv_estimator_fn("gcpo") is compute_gcpo_advantage


def test_mixed_binary_rewards_produce_a_nonzero_policy_gradient():
    token_rewards = torch.tensor([[1.0], [-1.0], [-1.0], [-1.0]])
    response_mask = torch.ones_like(token_rewards)
    advantages, _ = compute_gcpo_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=np.array(["prompt"] * 4, dtype=object),
        response_embeddings=torch.eye(4),
    )

    assert advantages[0, 0] > 0
    assert torch.all(advantages[1:, 0] < 0)
    log_prob = torch.full((4, 1), -1.0, requires_grad=True)
    before = log_prob.detach().clone()
    optimizer = torch.optim.SGD([log_prob], lr=0.1)
    loss, *_ = compute_policy_loss(
        old_log_prob=before,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
        loss_agg_mode="token-mean",
    )
    loss.backward()

    assert torch.linalg.vector_norm(log_prob.grad) > 0
    optimizer.step()
    assert log_prob[0, 0] > before[0, 0]
    assert torch.all(log_prob[1:, 0] < before[1:, 0])


@pytest.mark.parametrize(
    ("scores", "embeddings", "error"),
    [
        (torch.ones(2, 1), torch.eye(2), "one-dimensional"),
        (torch.ones(2), torch.ones(2), "two-dimensional"),
        (torch.ones(2), torch.ones(3, 2), "same batch dimension"),
        (torch.ones(2), torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]), "finite"),
        (torch.ones(2), torch.tensor([[0.0, 0.0], [0.0, 1.0]]), "non-zero"),
    ],
)
def test_invalid_estimator_inputs_fail_loudly(scores, embeddings, error):
    with pytest.raises(ValueError, match=error):
        compute_gcpo_shapley_values(scores, embeddings)
