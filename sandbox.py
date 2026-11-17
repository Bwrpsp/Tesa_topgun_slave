import os
import sys
import time
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List
from hardware import drop_os_caches


def run_command_stream(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    prefix: str = "   [pip] ",
    verbose: bool = False
) -> Tuple[int, str]:
    """
    Executes a subprocess, capturing output while optionally streaming lines
    to stdout in real time. Enforces timeout.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    logs: List[str] = []
    start_time = time.time()

    for line in iter(proc.stdout.readline, ""):
        logs.append(line)
        if verbose:
            sys.stdout.write(f"{prefix}{line}")
            sys.stdout.flush()
        if timeout and (time.time() - start_time > timeout):
            proc.kill()
            raise subprocess.TimeoutExpired(cmd, timeout, output="".join(logs))

    proc.wait()
    return proc.returncode, "".join(logs)


class SandboxManager:
    """
    Manages isolated virtual environments, dependency installation,
    and complete post-run cleanup to ensure zero inter-team contamination.
    """

    def get_weights_dir(self, sandbox_dir: Path) -> Optional[Path]:
        """
        Locates the model weights folder.
        Verifies that weights/ exists, is a directory, and contains files (any extension or format).
        """
        weights_dir = sandbox_dir / "weights"
        if weights_dir.exists() and weights_dir.is_dir():
            files = [f for f in weights_dir.rglob("*") if f.is_file()]
            if files:
                return weights_dir

        return None

    @staticmethod
    def get_venv_python(sandbox_dir: Path) -> Path:
        if sys.platform == "win32":
            return sandbox_dir / ".venv" / "Scripts" / "python.exe"
        return sandbox_dir / ".venv" / "bin" / "python"

    @staticmethod
    def get_venv_pip(sandbox_dir: Path) -> Path:
        if sys.platform == "win32":
            return sandbox_dir / ".venv" / "Scripts" / "pip.exe"
        return sandbox_dir / ".venv" / "bin" / "pip"

    def setup_sandbox(
        self,
        job_id: str,
        bundle_path: Path,
        base_dir: Path,
        verbose: bool = False
    ) -> Path:
        sandbox_dir = base_dir / job_id
        if sandbox_dir.exists():
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"   [Sandbox] Extracting {bundle_path.name} to {sandbox_dir} ...")

        # Unpack bundle
        with zipfile.ZipFile(bundle_path, "r") as zf:
            zf.extractall(sandbox_dir)

        if verbose:
            extracted_items = [f.name for f in sandbox_dir.iterdir()]
            print(f"   [Sandbox] Extracted {len(extracted_items)} items: {', '.join(extracted_items)}")

        # Create isolated virtual environment
        sys_python = sys.executable
        venv_dir = sandbox_dir / ".venv"
        if verbose:
            print(f"   [Sandbox] Creating isolated virtual environment at {venv_dir.name} ...")

        cmd_venv = [sys_python, "-m", "venv", str(venv_dir)]
        res = subprocess.run(cmd_venv, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to create virtual environment: {res.stderr}")

        if verbose:
            print(f"   [Sandbox] Virtual environment initialized ({self.get_venv_python(sandbox_dir).name})")

        return sandbox_dir

    def install_dependencies(
        self,
        sandbox_dir: Path,
        wheelhouse_dir: Optional[Path] = None,
        timeout_seconds: int = 600,
        verbose: bool = False
    ) -> Tuple[bool, str]:
        req_file = sandbox_dir / "requirements.txt"
        if not req_file.exists():
            if verbose:
                print("   [Dependencies] No requirements.txt found, skipping dependency installation.")
            return True, "No requirements.txt found, skipping install."

        if verbose:
            print("   [Dependencies] Found requirements.txt with contents:")
            try:
                with open(req_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line_s = line.strip()
                        if line_s and not line_s.startswith("#"):
                            print(f"      * {line_s}")
            except Exception:
                pass

        pip_bin = self.get_venv_pip(sandbox_dir)
        if not pip_bin.exists():
            return False, f"Pip binary not found at {pip_bin}"

        # Upgrade pip tooling
        if verbose:
            print("   [Dependencies] Checking pip tooling (pip, setuptools, wheel) ...")
        try:
            run_command_stream(
                [str(pip_bin), "install", "--upgrade", "pip", "setuptools", "wheel"],
                timeout=120,
                prefix="   [pip-setup] ",
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"   [Dependencies Warning] Tooling upgrade warning: {e}")

        logs: List[str] = []

        # Attempt 1: Fast local wheelhouse installation if wheelhouse exists
        if wheelhouse_dir and wheelhouse_dir.exists() and any(wheelhouse_dir.glob("*.whl")):
            if verbose:
                print(f"   [Dependencies] Attempting installation from local wheelhouse ({wheelhouse_dir}) ...")
            cmd_fast = [
                str(pip_bin), "install",
                "--no-index",
                f"--find-links={str(wheelhouse_dir)}",
                "-r", str(req_file)
            ]
            try:
                code, out = run_command_stream(
                    cmd_fast,
                    timeout=timeout_seconds,
                    prefix="   [pip-wheelhouse] ",
                    verbose=verbose
                )
                logs.append(f"Wheelhouse install output:\n{out}")
                if code == 0:
                    if verbose:
                        print("   [Dependencies] Successfully installed all packages from local wheelhouse!")
                    return True, "\n".join(logs)
                if verbose:
                    print("   [Dependencies] Local wheelhouse had missing packages. Falling back to online PyPI ...")
                logs.append("Wheelhouse install failed, falling back to online PyPI with wheelhouse link.")
            except subprocess.TimeoutExpired:
                return False, f"Pip wheelhouse install timed out after {timeout_seconds} seconds."

        # Attempt 2: PyPI fallback (with wheelhouse caching if available)
        cmd_online = [str(pip_bin), "install"]
        if wheelhouse_dir and wheelhouse_dir.exists():
            cmd_online.append(f"--find-links={str(wheelhouse_dir)}")
        cmd_online.extend(["-r", str(req_file)])

        if verbose:
            print(f"   [Dependencies] Running online pip install (timeout: {timeout_seconds}s) ...")

        try:
            code, out = run_command_stream(
                cmd_online,
                timeout=timeout_seconds,
                prefix="   [pip] ",
                verbose=verbose
            )
            logs.append(f"Online pip install output:\n{out}")
            if code == 0:
                if verbose:
                    print("   [Dependencies] All dependencies installed successfully!")
                return True, "\n".join(logs)
            else:
                if verbose:
                    print(f"   [Dependencies] Pip install failed with exit code {code}!")
                return False, "\n".join(logs)
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"   [Dependencies] Pip install timed out after {timeout_seconds} seconds!")
            return False, f"Pip install timed out after {timeout_seconds} seconds."

    def find_weight_file(self, sandbox_dir: Path) -> Optional[Path]:
        """
        Locates a primary weight file inside weights/ (for contestants whose infer.py takes a file).
        """
        weights_dir = sandbox_dir / "weights"
        if weights_dir.exists() and weights_dir.is_dir():
            files = [f for f in weights_dir.rglob("*") if f.is_file()]
            if files:
                return next((f for f in files if "best" in f.name.lower()), files[0])
        return None

    def find_weights(self, sandbox_dir: Path) -> Optional[Path]:
        """Backward-compatible alias: returns the weights directory."""
        return self.get_weights_dir(sandbox_dir)

    def purge_sandbox(self, sandbox_dir: Path):
        """
        Deep cleanup: deletes code, venv, outputs, pip caches, and flushes memory.
        """
        if sandbox_dir.exists():
            shutil.rmtree(sandbox_dir, ignore_errors=True)

        # Clear pip cache
        try:
            subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], capture_output=True, timeout=10)
        except Exception:
            pass

        # Flush OS buffer caches
        drop_os_caches()
