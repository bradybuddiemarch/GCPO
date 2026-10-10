# Copyright 2026 GCPO Authors
# Licensed under the Apache License, Version 2.0 (the "License").

"""Binary Math-Verify reward used by the GCPO training recipe."""

from functools import lru_cache
import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse
from math_verify import verify as verify_math


_ANSWER_PATTERN = re.compile(
    r"(?i)\bAnswer\s*:\s*(.*?)(?=\bAnswer\s*:|[\r\n]|$)"
)
_PREDICTION_EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False,
            malformed_operators=False,
            basic_latex=True,
            equations=False,
            boxed="all",
            units=True,
        ),
        boxed_match_priority=0,
        try_extract_without_anchor=True,
    ),
    ExprExtractionConfig(),
]


def _last_boxed_content(text: str) -> tuple[str | None, int]:
    """Return the content and position of the last balanced ``\\boxed{...}``."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None, -1

    depth = 0
    content_start = start + len("\\boxed{")
    for position in range(content_start - 1, len(text)):
        character = text[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:position], start
    return None, -1


def _extract_candidate(solution: str) -> str | None:
    answer_matches = list(_ANSWER_PATTERN.finditer(solution))
    last_answer = answer_matches[-1] if answer_matches else None
    boxed_content, boxed_position = _last_boxed_content(solution)

    if last_answer is not None and last_answer.start() > boxed_position:
        return last_answer.group(1).strip()
    return boxed_content.strip() if boxed_content is not None else None


def _display_prediction(candidate: str | None) -> str:
    if candidate is None:
        return "[INVALID]"
    prediction = candidate.strip()
    if prediction.startswith("$") and prediction.endswith("$"):
        prediction = prediction[1:-1].strip()
    boxed_content, boxed_position = _last_boxed_content(prediction)
    if boxed_position == 0 and boxed_content is not None:
        prediction = boxed_content.strip()
    return prediction


@lru_cache(maxsize=4096)
def _parse_ground_truth(ground_truth: str):
    boxed_content, _ = _last_boxed_content(ground_truth)
    target = boxed_content if boxed_content is not None else ground_truth
    return parse(f"\\boxed{{{target}}}", extraction_mode="first_match")


def _is_equivalent(candidate: str, ground_truth: str) -> bool:
    try:
        gold = _parse_ground_truth(str(ground_truth))
        prediction = parse(
            candidate,
            extraction_config=_PREDICTION_EXTRACTION_CONFIG,
            extraction_mode="first_match",
        )
        return bool(gold and prediction and verify_math(gold, prediction))
    except Exception:
        return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
) -> dict:
    """Return an exact 0/1 reward without EOS or terminal-answer gating."""
    if data_source != "math_dapo" and not data_source.startswith("aime"):
        raise ValueError(f"Unsupported data source for GCPO reward: {data_source}")

    candidate = _extract_candidate(str(solution_str))
    correct = candidate is not None and _is_equivalent(candidate, str(ground_truth))
    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "pred": _display_prediction(candidate),
    }
