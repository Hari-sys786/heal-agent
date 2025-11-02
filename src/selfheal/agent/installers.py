from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from selfheal.agent.classifier import PackageRequest
from selfheal.utils.servicenow_client import ServiceNowClient
from selfheal.utils.shell import CommandResult, run_command


@dataclass(slots=True)
class InstallerConfig:
    """Configuration for package installation routines."""

    package_manager: str = "apt-get"
    dry_run: bool = True
    sudo: bool = True
    sudo_password: str | None = None


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
        skip_words = {
            "install",
            "installing",
            "setup",
            "please",
            "run",
            "apt-get",
            "yum",
            "package",
            "packages",
            "latest",
            "version",
        }
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

    def install(self, sys_id: str, packages: Iterable[PackageRequest | Mapping[str, object] | str]) -> list[CommandResult]:
        """Install each requested package and log results to ServiceNow."""
        results: list[CommandResult] = []
        normalized_packages = [self._coerce_package(pkg) for pkg in packages if pkg]
        normalized_packages = [pkg for pkg in normalized_packages if pkg is not None]
        if not normalized_packages:
            self.sn_client.append_work_note(sys_id, "No packages inferred from the request.")
            return results

        use_password = bool(self.config.sudo_password and self.config.sudo and not self.config.dry_run)

        for package in normalized_packages:
            package_success = True
            last_result: CommandResult | None = None

            if package.install_steps:
                total_steps = len(package.install_steps)
                for index, step in enumerate(package.install_steps, start=1):
                    self._logger.info(
                        "Executing install step %s/%s for %s: %s",
                        index,
                        total_steps,
                        package.name,
                        step,
                    )
                    self.sn_client.append_work_note(
                        sys_id,
                        f"Installing {package.name} step {index}/{total_steps} using `{step}`.",
                    )
                    step_result = run_command(("bash", "-lc", step), dry_run=self.config.dry_run)
                    results.append(step_result)
                    last_result = step_result
                    if not step_result.succeeded():
                        package_success = False
                        self._logger.warning(
                            "Install step %s/%s for %s failed (rc=%s stderr=%s)",
                            index,
                            total_steps,
                            package.name,
                            step_result.returncode,
                            step_result.stderr.strip(),
                        )
                        break
            elif package.install_command:
                self._logger.info("Executing custom install command for %s", package.name)
                self.sn_client.append_work_note(
                    sys_id,
                    f"Installing {package.name} using custom command `{package.install_command}`.",
                )
                result = run_command(("bash", "-lc", package.install_command), dry_run=self.config.dry_run)
                results.append(result)
                last_result = result
                package_success = result.succeeded()
            else:
                cmd = self._build_command(package, use_password=use_password)
                self._logger.info("Installing package %s via command %s", package.name, cmd)
                display_name = package.name if not package.version else f"{package.name} ({package.version})"
                self.sn_client.append_work_note(
                    sys_id,
                    f"Installing {display_name} using `{self._format_command(cmd)}`.",
                )
                password_input = f"{self.config.sudo_password}\n" if use_password else None
                result = run_command(cmd, dry_run=self.config.dry_run, input_text=password_input)
                results.append(result)
                last_result = result
                package_success = result.succeeded()

            note = f"{package.name} installation {'succeeded' if package_success else 'failed'}."
            self.sn_client.append_work_note(sys_id, note)
            if package_success:
                self._logger.info(
                    "Package %s installed successfully (dry_run=%s)", package.name, self.config.dry_run
                )
            else:
                if last_result is not None:
                    self._logger.error(
                        "Package %s installation failed (rc=%s stdout=%s stderr=%s)",
                        package.name,
                        last_result.returncode,
                        last_result.stdout.strip(),
                        last_result.stderr.strip(),
                    )
                else:
                    self._logger.error("Package %s installation failed", package.name)
        return results

    def _build_command(self, package: PackageRequest, *, use_password: bool) -> Sequence[str]:
        manager = package.manager or self.config.package_manager
        args: list[str] = []
        if self.config.sudo:
            args.append("sudo")
            if use_password:
                args.append("-S")

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

        if package.version and package.version.lower() != "latest":
            args.append(f"{package.name}={package.version}")
        else:
            args.append(package.name)
        return tuple(args)

    @staticmethod
    def _coerce_package(package: PackageRequest | Mapping[str, object] | str) -> PackageRequest | None:
        if isinstance(package, PackageRequest):
            return package
        if isinstance(package, Mapping):
            name = str(package.get("name", "")).strip()
            if not name:
                return None
            version_raw = package.get("version")
            version = str(version_raw).strip() or None if version_raw is not None else None
            manager_raw = package.get("manager")
            manager = str(manager_raw).strip() or None if manager_raw is not None else None
            command_raw = package.get("install_command")
            install_command = str(command_raw).strip() or None if command_raw is not None else None
            steps_raw = package.get("install_steps")
            steps: List[str] = []
            if isinstance(steps_raw, str) and steps_raw.strip():
                steps = [steps_raw.strip()]
            elif isinstance(steps_raw, Iterable) and not isinstance(steps_raw, (str, bytes)):
                for item in steps_raw:
                    if isinstance(item, str) and item.strip():
                        steps.append(item.strip())
            return PackageRequest(name=name, version=version, manager=manager, install_command=install_command, install_steps=steps)
        coerced_name = str(package).strip()
        if not coerced_name:
            return None
        return PackageRequest(name=coerced_name)

    @staticmethod
    def _format_command(cmd: Sequence[str]) -> str:
        return " ".join(cmd)


__all__ = ["InstallerConfig", "PackageInstaller"]
