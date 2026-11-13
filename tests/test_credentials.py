"""copula_inter.credentials: lookup order, the private-file check, and the ssh-agent helper (subprocess stubbed)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

import copula_inter.credentials as cred


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """No real environment, .env, config dir or home leaks into a test."""
    for var in ("TABPFN_TOKEN", "TABPFN_TOKEN_FILE", "COPULA_SSH_KEY", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COPULA_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(cred, "DOTENV_PATH", tmp_path / "repo" / ".env")
    for d in ("home/.config", "cfg", "repo"):
        (tmp_path / d).mkdir(parents=True)
    return tmp_path


def _private(path: Path, text: str, mode: int = 0o600) -> Path:
    path.write_text(text)
    path.chmod(mode)
    return path


def test_lookup_order_env_then_dotenv_then_config_then_legacy(isolated: Path, monkeypatch: MonkeyPatch) -> None:
    legacy = _private(isolated / "home" / ".config" / "tabpfn_token", "from-legacy\n")
    assert cred.get_secret(cred.TABPFN_TOKEN) == "from-legacy"
    _private(isolated / "cfg" / "tabpfn_token", "from-config\n")
    assert cred.get_secret(cred.TABPFN_TOKEN) == "from-config"
    _private(isolated / "repo" / ".env", "TABPFN_TOKEN='from-dotenv'\n")
    assert cred.get_secret(cred.TABPFN_TOKEN) == "from-dotenv"
    monkeypatch.setenv("TABPFN_TOKEN", "from-env")
    assert cred.get_secret(cred.TABPFN_TOKEN) == "from-env"
    assert legacy.exists()


def test_tabpfn_token_file_override(isolated: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TABPFN_TOKEN_FILE", str(_private(isolated / "elsewhere", "k")))
    assert cred.get_secret(cred.TABPFN_TOKEN) == "k"


def test_empty_values_fall_through(isolated: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TABPFN_TOKEN", "  ")
    _private(isolated / "repo" / ".env", "TABPFN_TOKEN=\n")
    _private(isolated / "cfg" / "tabpfn_token", "real")
    assert cred.get_secret(cred.TABPFN_TOKEN) == "real"


def test_missing_raises_with_sources_but_optional_returns_none(isolated: Path) -> None:
    assert cred.get_secret(cred.TABPFN_TOKEN, required=False) is None
    with pytest.raises(cred.MissingSecretError, match="ux.priorlabs.ai") as err:
        cred.get_secret(cred.TABPFN_TOKEN)
    assert str(isolated / "cfg" / "tabpfn_token") in str(err.value)


@pytest.mark.parametrize("where", ["dotenv", "config"])
def test_group_readable_file_is_refused_without_leaking_it(isolated: Path, where: str) -> None:
    path = isolated / "repo" / ".env" if where == "dotenv" else isolated / "cfg" / "tabpfn_token"
    _private(path, "TABPFN_TOKEN=s3cret\n" if where == "dotenv" else "s3cret", mode=0o644)
    with pytest.raises(cred.InsecureSecretFileError, match="chmod 600") as err:
        cred.get_secret(cred.TABPFN_TOKEN)
    assert "s3cret" not in str(err.value)


def test_export_tabpfn_token_sets_environ(isolated: Path, monkeypatch: MonkeyPatch) -> None:
    import os

    _private(isolated / "cfg" / "tabpfn_token", "tok")
    monkeypatch.setenv("TABPFN_TOKEN", "")  # monkeypatch restores it after the test
    assert cred.export_tabpfn_token() == "tok"
    assert os.environ["TABPFN_TOKEN"] == "tok"


def test_ssh_key_path_expands_and_checks(isolated: Path, monkeypatch: MonkeyPatch) -> None:
    key = _private(isolated / "home" / "id_test", "PRIVATE")
    monkeypatch.setenv("COPULA_SSH_KEY", "~/id_test")
    assert cred.ssh_key_path() == key
    key.chmod(0o640)
    with pytest.raises(cred.InsecureSecretFileError):
        cred.ssh_key_path()
    monkeypatch.setenv("COPULA_SSH_KEY", "~/absent")
    with pytest.raises(cred.MissingSecretError, match="not a file"):
        cred.ssh_key_path()


class _FakeSsh:
    """Records ssh-keygen / ssh-add calls; the agent initially holds `held`."""

    def __init__(self, held: list[str]) -> None:
        self.held = held
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        if cmd[0] == "ssh-keygen":
            out = "256 SHA256:abc user@host (ED25519)\n"
        elif cmd[:2] == ["ssh-add", "-l"]:
            out = "".join(f"256 {fp} x (ED25519)\n" for fp in self.held)
        else:
            out = ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


@pytest.mark.parametrize("held,added", [([], True), (["SHA256:abc"], False)])
def test_ssh_add_skips_a_key_the_agent_holds(
    isolated: Path, monkeypatch: MonkeyPatch, held: list[str], added: bool
) -> None:
    key = _private(isolated / "home" / "id_test", "PRIVATE")
    monkeypatch.setenv("COPULA_SSH_KEY", str(key))
    monkeypatch.setenv("SSH_AUTH_SOCK", str(isolated / "agent.sock"))
    monkeypatch.setattr(cred.shutil, "which", lambda name: f"/usr/bin/{name}")
    fake = _FakeSsh(held)
    monkeypatch.setattr(cred.subprocess, "run", fake)
    assert cred.ssh_add(lifetime_s=3600) is added
    adds = [c for c in fake.calls if c[0] == "ssh-add" and c[1] != "-l"]
    assert adds == ([["ssh-add", "-t", "3600", str(key)]] if added else [])


def test_ssh_add_needs_an_agent(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    with pytest.raises(RuntimeError, match="SSH_AUTH_SOCK"):
        cred.ssh_add(Path("/nonexistent"))
