import pytest

from gcpo.reward import compute_score


@pytest.mark.parametrize(
    ("completion", "ground_truth"),
    [
        ("The calculation is complete.\nAnswer: 42", "42"),
        (r"Answer: 1/2", r"\frac{1}{2}"),
        (r"Therefore the result is \boxed{\frac{3}{4}}.", r"3/4"),
        (r"Answer: $42$", r"\boxed{42}"),
    ],
)
def test_equivalent_math_answers_receive_binary_one(completion, ground_truth):
    result = compute_score("math_dapo", completion, ground_truth)

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["pred"] != "[INVALID]"


def test_rightmost_explicit_answer_is_used():
    result = compute_score(
        "math_dapo",
        "A draft said Answer: 41.\nChecking again.\nAnswer: \\boxed{42}",
        "42",
    )

    assert result["score"] == 1.0
    assert result["acc"] is True


def test_rightmost_explicit_answer_is_used_when_answers_share_a_line():
    result = compute_score(
        "math_dapo",
        "A draft said Answer: 41. Final Answer: 42",
        "42",
    )

    assert result["score"] == 1.0


def test_later_boxed_answer_takes_precedence_over_earlier_answer_line():
    result = compute_score(
        "math_dapo",
        "Answer: 42\nThis was revised to \\boxed{41}.",
        "42",
    )

    assert result["score"] == 0.0
    assert result["acc"] is False


def test_correct_answer_does_not_need_to_be_the_terminal_text():
    result = compute_score(
        "math_dapo",
        "Answer: 42\nI will now double-check the derivation.",
        "42",
    )

    assert result["score"] == 1.0


@pytest.mark.parametrize(
    "completion",
    [
        "Answer: 41",
        "The reasoning mentions 42 but has no explicit answer anchor.",
        r"The final expression is \boxed{41}.",
        r"The malformed result is \boxed{42.",
    ],
)
def test_wrong_or_unparseable_answers_receive_binary_zero(completion):
    result = compute_score("math_dapo", completion, "42")

    assert result["score"] == 0.0
    assert result["acc"] is False


def test_scores_are_exactly_binary():
    correct = compute_score("math_dapo", "Answer: 42", "42")
    wrong = compute_score("math_dapo", "Answer: 41", "42")

    assert {correct["score"], wrong["score"]} == {0.0, 1.0}


def test_aime_validation_source_is_supported():
    assert compute_score("aime24", "Answer: 123", "123")["score"] == 1.0


def test_unsupported_data_source_fails_loudly():
    with pytest.raises(ValueError, match="Unsupported data source"):
        compute_score("gsm8k", "Answer: 42", "42")
