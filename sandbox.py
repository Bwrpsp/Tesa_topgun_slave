import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional
from hardware import drop_os_caches


class SandboxManager:
    """
    Manages isolated virtual environments, dependency installation,
    and complete post-run cleanup to ensure zero inter-team contamination.
    """

    ALLOWED_WEIGHT_EXTS = {".joblib", ".pt", ".pth", ".onnx", ".bin", ".h5", ".pkl", ".tflite", ".engine"}

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

    def setup_sandbox(self, job_id: str, bundle_path: Path, base_dir: Path) -> Path:
        sandbox_dir = base_dir / job_id
        if sandbox_dir.exists():
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Unpack bundle
        with zipfile.ZipFile(bundle_path, "r") as zf:
            zf.extractall(sandbox_dir)

        # Create isolated virtual environment
        sys_python = sys.executable
        cmd_venv = [sys_python, "-m", "venv", str(sandbox_dir / ".venv")]
        res = subprocess.run(cmd_venv, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to create virtual environment: {res.stderr}")

        return sandbox_dir

    def install_dependencies(
        self,
        sandbox_dir: Path,
        wheelhouse_dir: Optional[Path] = None,
        timeout_seconds: int = 240
    ) -> Tuple[bool, str]:
        req_file = sandbox_dir / "requirements.txt"
        if not req_file.exists():
            return True, "No requirements.txt found, skipping install."

        pip_bin = self.get_venv_pip(sandbox_dir)
        if not pip_bin.exists():
            return False, f"Pip binary not found at {pip_bin}"

        # Upgrade pip tooling quietly
        subprocess.run([str(pip_bin), "install", "--upgrade", "pip", "setuptools", "wheel"],
                       capture_output=True, timeout=60)

        logs = []

        # Attempt 1: Fast local wheelhouse installation if wheelhouse exists
        if wheelhouse_dir and wheelhouse_dir.exists() and any(wheelhouse_dir.glob("*.whl")):
            cmd_fast = [
                str(pip_bin), "install",
                "--no-index",
                f"--find-links={str(wheelhouse_dir)}",
                "-r", str(req_file)
            ]
            res = subprocess.run(cmd_fast, capture_output=True, text=True, timeout=timeout_seconds)
            logs.append(f"Wheelhouse install stdout:\n{res.stdout}")
            if res.returncode == 0:
                return True, "\n".join(logs)
            logs.append(f"Wheelhouse install failed, falling back to online PyPI with wheelhouse link:\n{res.stderr}")

        # Attempt 2: PyPI fallback (with wheelhouse caching if available)
        cmd_online = [str(pip_bin), "install"]
        if wheelhouse_dir and wheelhouse_dir.exists():
            cmd_online.append(f"--find-links={str(wheelhouse_dir)}")
        cmd_online.extend(["-r", str(req_file)])

        try:
            res = subprocess.run(cmd_online, capture_output=True, text=True, timeout=timeout_seconds)
            logs.append(f"Online pip install stdout:\n{res.stdout}")
            if res.returncode == 0:
                return True, "\n".join(logs)
            else:
                logs.append(f"Online pip install stderr:\n{res.stderr}")
                return False, "\n".join(logs)
        except subprocess.TimeoutExpired:
            return False, f"Pip install timed out after {timeout_seconds} seconds."

    def find_weights(self, sandbox_dir: Path) -> Optional[Path]:
        """Locates the model weights file."""
        weights_dir = sandbox_dir / "weights"
        if weights_dir.exists():
            for f in weights_dir.iterdir():
                if f.is_file() and f.suffix.lower() in self.ALLOWED_WEIGHT_EXTS:
                    return f

        # Fallback: check root
        for f in sandbox_dir.iterdir():
            if f.is_file() and f.suffix.lower() in self.ALLOWED_WEIGHT_EXTS:
                return f

        return None

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
