from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_verl_pin_and_dependencies():
    pin = (ROOT / "REQUIRED_VERL.txt").read_text()
    requirements = (ROOT / "requirements.txt").read_text()

    assert "VERL_REPOSITORY=https://github.com/volcengine/verl.git" in pin
    assert "VERL_TAG=v0.4.1" in pin
    assert "VERL_COMMIT=8d9e350ea58c7ad4b50dd14d9dcb50577242c55f" in pin
    assert "sentence-transformers==5.3.0" in requirements
    assert "math-verify[antlr4_9_3]==0.9.0" in requirements


def test_runtime_artifacts_are_ignored():
    ignore = (ROOT / ".gitignore").read_text()

    for path in ("third_party/", "data/", "checkpoints/", "outputs/", "wandb/"):
        assert path in ignore
    assert "model/*" in ignore
    assert "!model/README.md" in ignore


def test_public_documentation_and_license_files_exist():
    for relative_path in ("README.md", "LICENSE", "NOTICE", "model/README.md"):
        path = ROOT / relative_path
        assert path.is_file()
        assert path.stat().st_size > 0


def test_public_release_contains_no_machine_paths_or_credentials():
    excluded = {".git", "third_party", ".pytest_cache", "__pycache__"}
    text_suffixes = {"", ".md", ".py", ".sh", ".txt", ".toml", ".patch"}

    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if "docs" in path.parts and "superpowers" in path.parts:
            continue
        if path.suffix not in text_suffixes:
            continue
        text = path.read_text(errors="ignore")
        lowered = text.lower()
        assert "/" + "data2/" not in text, path
        assert "hf_" + "token=" not in lowered, path
        assert "api_" + "key=" not in lowered, path
