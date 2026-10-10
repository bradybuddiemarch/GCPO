# Copyright 2026 GCPO Authors
# Licensed under the Apache License, Version 2.0 (the "License").

"""Frozen, local-only sentence encoder adapter used by GCPO."""

from pathlib import Path

import torch


def load_frozen_encoder(model_path: str, device: str):
    """Load a Sentence Transformer from a local directory and freeze it."""
    local_path = Path(model_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(
            f"Local semantic encoder directory does not exist: {local_path}"
        )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(local_path), device=device, local_files_only=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_responses(
    model,
    tokenizer,
    response_ids: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Decode generated responses and return detached, unnormalized embeddings."""
    if batch_size <= 0:
        raise ValueError(f"sentence-transformer batch_size must be positive, got {batch_size}")
    if response_ids.ndim != 2:
        raise ValueError(
            "response_ids must be a two-dimensional tensor, "
            f"got shape {tuple(response_ids.shape)}"
        )

    texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=batch_size,
            normalize_embeddings=False,
        )
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.as_tensor(embeddings)
    return embeddings.detach().to(device=response_ids.device)
