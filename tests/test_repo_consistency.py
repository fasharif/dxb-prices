"""Version pins that live in several files must agree.

Dependabot updates the uv image in the Dockerfile but not the workflows, so the
workflows read the uv version from the Dockerfile and the tests check they still do.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SHA_PIN = re.compile(r"^[0-9a-f]{40}$")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _env(workflow: str, name: str) -> str:
    match = re.search(rf'^\s+{name}: "([^"]+)"$', _read(WORKFLOWS / workflow), re.MULTILINE)
    assert match, f"{workflow} has no {name} in its env block"
    return match.group(1)


def test_the_workflows_take_the_uv_version_from_the_dockerfile() -> None:
    dockerfile = _read(ROOT / "Dockerfile")
    image = re.search(r"^FROM ghcr\.io/astral-sh/uv:(\S+) AS uv$", dockerfile, re.MULTILINE)
    assert image, "the workflows read the uv version from this FROM line"
    for workflow in ("ci.yml", "retrain.yml"):
        text = _read(WORKFLOWS / workflow)
        assert "UV_VERSION:" not in text, f"{workflow} must not pin uv itself"
        setups = text.count("uses: astral-sh/setup-uv@")
        reads = text.count("name: Read the uv version from the Dockerfile")
        assert setups and reads == setups, f"{workflow}: each uv setup needs the read step"
        assert text.count("version: ${{ env.UV_VERSION }}") == setups


def test_python_version_is_the_same_in_the_workflows_and_the_dockerfile() -> None:
    images = re.findall(r"^FROM \S+/python:(\d+\.\d+)-", _read(ROOT / "Dockerfile"), re.MULTILINE)
    assert images and set(images) == {_env("ci.yml", "PYTHON_VERSION")}
    assert _env("ci.yml", "PYTHON_VERSION") == _env("retrain.yml", "PYTHON_VERSION")


def test_the_test_matrix_starts_at_the_minimum_python_and_includes_the_images() -> None:
    matrix = re.search(r"^\s+python: \[([^\]]+)\]$", _read(WORKFLOWS / "ci.yml"), re.MULTILINE)
    assert matrix
    versions = [v.strip().strip('"') for v in matrix.group(1).split(",")]
    assert _env("ci.yml", "PYTHON_VERSION") in versions
    minimum = re.search(r'^requires-python = ">=(\d+\.\d+)"$', _read(ROOT / "pyproject.toml"), re.M)
    assert minimum
    oldest = min(versions, key=lambda v: tuple(int(part) for part in v.split(".")))
    assert oldest == minimum.group(1)


def test_action_refs_are_major_tags_or_commit_pins() -> None:
    """setup-uv publishes no moving major tags (only exact releases), so it must be pinned."""
    refs = [
        (workflow.name, match.group(1), match.group(2), match.group(3))
        for workflow in sorted(WORKFLOWS.glob("*.yml"))
        for match in re.finditer(r"uses: ([\w.-]+/[\w.-]+)@(\S+)(?:\s+#\s*(\S+))?", _read(workflow))
    ]
    assert refs
    for workflow, action, ref, comment in refs:
        if SHA_PIN.match(ref):
            assert comment and re.fullmatch(r"v\d+\.\d+\.\d+", comment), (workflow, action)
        else:
            assert action != "astral-sh/setup-uv", f"{workflow}: pin setup-uv to a commit"
            assert re.fullmatch(r"v\d+", ref), (workflow, action, ref)
