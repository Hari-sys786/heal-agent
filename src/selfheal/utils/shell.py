from __future__ import annotations

import shlex
import subprocess
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CommandResult:
    """Outcome of a shell command invocation."""

    cmd: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    def succeeded(self) -> bool:
        """True when the command exited with code 0."""
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the command result for logging or JSON responses."""
        return {
            "cmd": list(self.cmd),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "succeeded": self.succeeded(),
        }


def run_command(
    command: Iterable[str] | str,
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    check: bool = False,
    timeout: float | None = None,
    dry_run: bool = False,
) -> CommandResult:
    """Execute a command and capture stdout/stderr.

    Args:
        command: Sequence of arguments or shell string.
        env: Optional environment overrides.
        cwd: Optional working directory.
        check: Raise an exception if the command fails.
        timeout: Maximum number of seconds to wait.
        dry_run: If True, do not execute; return a zero exit code placeholder.
    """
    if isinstance(command, str):
        args: Sequence[str] = tuple(shlex.split(command))
    else:
        args = tuple(command)

    logger.debug("Executing command: %s (cwd=%s dry_run=%s timeout=%s)", args, cwd, dry_run, timeout)

    if dry_run:
        logger.debug("Dry-run enabled; skipping execution for command: %s", args)
        return CommandResult(cmd=args, returncode=0, stdout="(dry-run)", stderr="")

    env_dict: MutableMapping[str, str] | None = None
    if env is not None:
        env_dict = {**env}

    completed = subprocess.run(  # nosec B603 - caller controls command
        args,
        capture_output=True,
        text=True,
        check=check,
        cwd=cwd,
        timeout=timeout,
        env=env_dict,
    )
    result = CommandResult(
        cmd=args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if result.succeeded():
        logger.debug("Command succeeded: %s (rc=%s)", args, result.returncode)
    else:
        logger.warning("Command failed: %s (rc=%s stderr=%s)", args, result.returncode, result.stderr.strip())
    return result


__all__ = ["CommandResult", "run_command"]
