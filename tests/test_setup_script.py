import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup_verl.sh"
COMMIT = "8d9e350ea58c7ad4b50dd14d9dcb50577242c55f"


def _fake_commands(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        """#!/usr/bin/env bash
printf 'git' >> "$COMMAND_LOG"
printf ' %q' "$@" >> "$COMMAND_LOG"
printf '\n' >> "$COMMAND_LOG"
if [[ "$1" == "clone" ]]; then
  target="${@: -1}"
  mkdir -p "$target/.git"
elif [[ "$1" == "-C" && "$3" == "rev-parse" ]]; then
  printf '%s\n' "$EXPECTED_COMMIT"
elif [[ "$1" == "-C" && "$3" == "apply" && "$4" == "--reverse" ]]; then
  exit 1
fi
"""
    )
    git.chmod(0o755)
    python = bin_dir / "python3"
    python.write_text(
        """#!/usr/bin/env bash
printf 'python3' >> "$COMMAND_LOG"
printf ' %q' "$@" >> "$COMMAND_LOG"
printf '\n' >> "$COMMAND_LOG"
"""
    )
    python.chmod(0o755)
    return bin_dir


def test_setup_checks_exact_commit_before_applying_patch(tmp_path):
    bin_dir = _fake_commands(tmp_path)
    command_log = tmp_path / "commands.log"
    verl_root = tmp_path / "verl"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "COMMAND_LOG": str(command_log),
            "EXPECTED_COMMIT": COMMIT,
            "VERL_ROOT": str(verl_root),
        }
    )

    result = subprocess.run(
        ["bash", str(SETUP)], text=True, capture_output=True, env=env
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text().splitlines()
    verify_index = next(i for i, line in enumerate(commands) if "rev-parse HEAD" in line)
    check_index = next(
        i
        for i, line in enumerate(commands)
        if "apply --check" in line and "--reverse" not in line
    )
    assert verify_index < check_index
    assert any(f"checkout --detach {COMMIT}" in line for line in commands)
    assert any(
        "python3 -m pip install --no-build-isolation -e" in line
        for line in commands
    )


def test_setup_does_not_use_eval_or_overwrite_unknown_directory():
    text = SETUP.read_text()

    assert "eval " not in text
    assert "rm -" not in text
    assert "Refusing to use existing non-Git directory" in text
    assert "apply --reverse --check" in text
    assert "pip install --no-build-isolation -e" in text
