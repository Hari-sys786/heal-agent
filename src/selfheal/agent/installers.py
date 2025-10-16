from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from selfheal.utils.servicenow_client import ServiceNowClient
from selfheal.utils.shell import CommandResult, run_command


@dataclass(slots=True)
class InstallerConfig:
    """Configuration for package installation routines."""

    package_manager: str = "apt-get"
    dry_run: bool = True
    sudo: bool = True


class PackageInstaller:
    """High-level helper that executes package installations."""

    def __init__(self, sn_client: ServiceNowClient, config: InstallerConfig | None = None) -> None:
        self.sn_client = sn_client
        self.config = config or InstallerConfig()
        self._logger = logging.getLogger(__name__)

    def extract_packages(self, text: str) -> list[str]:
        """Infer package names from user-provided text."""
        tokens = re.findall(r"[A-Za-z0-9._+-]+", text.lower())
        packages: list[str] = []
        skip_words = {"install", "setup", "please", "run", "apt-get", "yum", "package", "packages"}
        for idx, token in enumerate(tokens):
            if token in {"install", "setup"} and idx + 1 < len(tokens):
                candidate = tokens[idx + 1]
                if candidate not in skip_words:
                    packages.append(candidate)
            elif token not in skip_words and len(token) > 2:
                if any(prefix in tokens[max(0, idx - 3) : idx] for prefix in ("install", "setup", "package")):
                    packages.append(token)
        unique = sorted(set(packages))
        self._logger.debug("Extracted candidate packages: text=%s packages=%s", text, unique)
        return unique

    def install(self, sys_id: str, packages: Iterable[str]) -> list[CommandResult]:
        """Install each requested package and log results to ServiceNow."""
        results: list[CommandResult] = []
        packages = list(packages)
        if not packages:
            self.sn_client.append_work_note(sys_id, "No packages inferred from the request.")
            return results

        for package in packages:
            cmd = self._build_command(package)
            self._logger.info("Installing package %s via command %s", package, cmd)
            self.sn_client.append_work_note(sys_id, f"Installing {package} using `{self._format_command(cmd)}`.")
            result = run_command(cmd, dry_run=self.config.dry_run)
            results.append(result)
            note = f"{package} installation {'succeeded' if result.succeeded() else 'failed'}."
            self.sn_client.append_work_note(sys_id, note)
            if result.succeeded():
                self._logger.info("Package %s installed successfully (dry_run=%s)", package, self.config.dry_run)
            else:
                self._logger.error(
                    "Package %s installation failed (rc=%s stdout=%s stderr=%s)",
                    package,
                    result.returncode,
                    result.stdout.strip(),
                    result.stderr.strip(),
                )
        return results

    def _build_command(self, package: str) -> Sequence[str]:
        manager = self.config.package_manager
        args: list[str] = []
        if self.config.sudo:
            args.append("sudo")

        if manager in {"apt", "apt-get"}:
            args.extend([manager, "install"])
            if self.config.dry_run:
                args.append("--dry-run")
            else:
                args.append("-y")
        elif manager in {"yum", "dnf"}:
            args.append(manager)
            args.append("install")
            if self.config.dry_run:
                args.append("--assumeno")
            else:
                args.append("-y")
        else:
            raise ValueError(f"Unsupported package manager: {manager}")

        args.append(package)
        return tuple(args)

    @staticmethod
    def _format_command(cmd: Sequence[str]) -> str:
        return " ".join(cmd)


__all__ = ["InstallerConfig", "PackageInstaller"]
