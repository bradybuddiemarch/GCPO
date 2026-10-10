import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "verl-v0.4.1-gcpo.patch"
PINNED_VERL = ROOT / "third_party" / "verl"


def _patch_text():
    return PATCH.read_text()


def test_patch_applies_cleanly_to_pinned_verl_checkout():
    result = subprocess.run(
        ["git", "apply", "--check", str(PATCH)],
        cwd=PINNED_VERL,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_patch_only_modifies_ray_trainer():
    changed_files = set(re.findall(r"^\+\+\+ b/(.+)$", _patch_text(), re.MULTILINE))

    assert changed_files == {"verl/trainer/ppo/ray_trainer.py"}


def test_patch_registers_and_forwards_external_gcpo_embeddings():
    text = _patch_text()

    assert "from gcpo import estimator as _gcpo_estimator" in text
    assert "from gcpo.encoder import encode_responses, load_frozen_encoder" in text
    assert '== "gcpo"' in text
    assert 'adv_kwargs["response_embeddings"]' in text
    assert "load_frozen_encoder(" in text
    assert "encode_responses(" in text
    assert "use_kl_in_reward=false" in text


def test_patch_does_not_add_policy_hidden_state_or_worker_changes():
    text = _patch_text()

    assert "fsdp_workers.py" not in text
    assert "extract_response_embeddings" not in text
    assert "hidden_states" not in text
    assert "output_hidden_states" not in text
