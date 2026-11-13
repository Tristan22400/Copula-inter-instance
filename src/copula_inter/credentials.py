"""Private credentials (the TabPFN API key, the Grid5000 SSH key): resolved at run time, never stored in git.

A secret is looked up, first hit wins, in:

1. the process environment (``spec.env_var``);
2. ``<checkout>/.env``, a git-ignored dotenv file (template: ``.env.example``);
3. ``config_dir() / spec.config_file``, a one-line file (default
   ``~/.config/copula-inter/``; ``COPULA_CONFIG_DIR`` overrides it);
4. the spec's legacy files (for TabPFN, ``$TABPFN_TOKEN_FILE`` or
   ``~/.config/tabpfn_token``).

Files 2-4 must not be readable by group or others (mode ``0600``):
``InsecureSecretFileError`` names a file that is, rather than read it. Values
never go on a command line (``ps``/``oarstat`` show job command lines to every
user of the shared cluster) and never into a log or an exception message.

The SSH key is stored as a path (the key itself stays in ``~/.ssh``);
``ssh_add`` loads it into the running ssh-agent so ``ssh``/``oarsh``/``rsync``
to Grid5000 use it.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

from copula_inter.paths import REPO_ROOT

# Read at call time, so tests can point it elsewhere.
DOTENV_PATH = REPO_ROOT / ".env"


class MissingSecretError(RuntimeError):
    """No source holds the secret."""


class InsecureSecretFileError(PermissionError):
    """A secret file is readable by group or others."""


@dataclass(frozen=True)
class SecretSpec:
    """Where one secret is looked up, and what to tell the user when it is missing."""

    env_var: str
    config_file: str
    help: str
    legacy_files: Callable[[], list[Path]] = field(default=lambda: [])


def _tabpfn_legacy_files() -> list[Path]:
    override = os.environ.get("TABPFN_TOKEN_FILE")
    return [Path(override).expanduser()] if override else [Path.home() / ".config" / "tabpfn_token"]


TABPFN_TOKEN = SecretSpec(
    env_var="TABPFN_TOKEN",
    config_file="tabpfn_token",
    help=(
        "TabPFN-3.5 requires a one-time licence acceptance: open https://ux.priorlabs.ai, log in, "
        "accept the TabPFN-3.5 licence and copy your API key from the account page."
    ),
    legacy_files=_tabpfn_legacy_files,
)

SSH_KEY = SecretSpec(
    env_var="COPULA_SSH_KEY",
    config_file="ssh_key_path",
    help="Give the path of the private key you use for Grid5000 (e.g. ~/.ssh/id_ed25519); the key stays in ~/.ssh.",
)


def config_dir() -> Path:
    """The per-user secrets directory: $COPULA_CONFIG_DIR, else $XDG_CONFIG_HOME/copula-inter, else ~/.config/copula-inter."""
    override = os.environ.get("COPULA_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "copula-inter"


def check_private(path: Path) -> None:
    """Raise InsecureSecretFileError unless group and others have no access to path."""
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise InsecureSecretFileError(
            f"{path} has mode {stat.S_IMODE(mode):04o}; secrets must be private: chmod 600 {path}"
        )


def _read_private(path: Path) -> str | None:
    if not path.is_file():
        return None
    check_private(path)
    return path.read_text().strip() or None


def _from_dotenv(env_var: str) -> str | None:
    if not DOTENV_PATH.is_file():
        return None
    check_private(DOTENV_PATH)
    value = dotenv_values(DOTENV_PATH).get(env_var)
    return value.strip() or None if value else None


def secret_sources(spec: SecretSpec) -> list[str]:
    """Every place spec is looked up, in priority order (for messages; holds no values)."""
    files = [str(DOTENV_PATH), str(config_dir() / spec.config_file), *map(str, spec.legacy_files())]
    return [f"${spec.env_var}", *files]


def get_secret(spec: SecretSpec, *, required: bool = True) -> str | None:
    """Resolve spec from the environment, .env, the config dir, then its legacy files.

    Returns None when nothing holds it and required is False; raises
    MissingSecretError when required.
    """
    value = os.environ.get(spec.env_var, "").strip() or None
    if value is None:
        value = _from_dotenv(spec.env_var)
    if value is None:
        for path in [config_dir() / spec.config_file, *spec.legacy_files()]:
            value = _read_private(path)
            if value is not None:
                break
    if value is None and required:
        where = "\n  ".join(secret_sources(spec))
        raise MissingSecretError(f"{spec.env_var} is not set. {spec.help}\nPut it in one of:\n  {where}")
    return value


def export_tabpfn_token() -> str:
    """Resolve the TabPFN key and put it in os.environ, where the tabpfn package reads it."""
    token = get_secret(TABPFN_TOKEN)
    assert token is not None  # required=True raised otherwise
    os.environ[TABPFN_TOKEN.env_var] = token
    return token


def ssh_key_path() -> Path:
    """The configured SSH private key: it must exist and be private (ssh refuses it otherwise)."""
    raw = get_secret(SSH_KEY)
    assert raw is not None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise MissingSecretError(f"{SSH_KEY.env_var} points to {path}, which is not a file.")
    check_private(path)
    return path


def _fingerprint(key: Path) -> str | None:
    """The key's fingerprint, read from its .pub when present (no passphrase needed)."""
    pub = key.with_name(key.name + ".pub")
    result = subprocess.run(
        ["ssh-keygen", "-l", "-f", str(pub if pub.is_file() else key)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    fields = result.stdout.split()
    return fields[1] if result.returncode == 0 and len(fields) > 1 else None


def ssh_add(key: Path | None = None, *, lifetime_s: int | None = None) -> bool:
    """Load the SSH key into the running ssh-agent unless it is already there.

    key defaults to ssh_key_path(); lifetime_s passes `ssh-add -t`. ssh-add
    asks for the passphrase on the terminal (or SSH_ASKPASS). Returns True if
    the key was added, False if the agent already held it.
    """
    if not os.environ.get("SSH_AUTH_SOCK"):
        raise RuntimeError("No ssh-agent: SSH_AUTH_SOCK is unset (start one with `eval $(ssh-agent)`).")
    if shutil.which("ssh-add") is None:
        raise RuntimeError("ssh-add is not on PATH.")
    key = ssh_key_path() if key is None else key.expanduser()
    check_private(key)
    fingerprint = _fingerprint(key)
    listed = subprocess.run(["ssh-add", "-l"], capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False)
    if fingerprint is not None and fingerprint in listed.stdout.split():
        return False
    cmd = ["ssh-add"] + (["-t", str(int(lifetime_s))] if lifetime_s is not None else []) + [str(key)]
    subprocess.run(cmd, check=True)
    return True
