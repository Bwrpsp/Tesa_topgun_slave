import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class SlaveConfig:
    MASTER_URL: str = os.getenv("MASTER_URL", "http://127.0.0.1:8000").rstrip("/")
    SECRET_TOKEN: str = os.getenv("SECRET_TOKEN", "tesa_topgun_secret_key_2026")
    SLAVE_ID: str = os.getenv("SLAVE_ID", "pi-node-01")

    POLL_INTERVAL_SECONDS: int = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
    MAX_TEMPERATURE_CELSIUS: float = float(os.getenv("MAX_TEMPERATURE_CELSIUS", "50.0"))
    COOLDOWN_SLEEP_SECONDS: int = int(os.getenv("COOLDOWN_SLEEP_SECONDS", "10"))

    # Always resolve paths to absolute so they work regardless of subprocess cwd
    _raw_datasets = os.getenv("LOCAL_DATASETS_BASE", "test_datasets")
    LOCAL_DATASETS_BASE: Path = (
        Path(_raw_datasets) if Path(_raw_datasets).is_absolute()
        else (BASE_DIR / _raw_datasets)
    ).resolve()

    _raw_wheelhouse = os.getenv("LOCAL_WHEELHOUSE", "/opt/pi_wheels")
    LOCAL_WHEELHOUSE: Path = Path(_raw_wheelhouse).resolve()

    _raw_sandbox = os.getenv("SANDBOX_BASE", "/tmp/eval_sandbox")
    if sys.platform == "win32" and _raw_sandbox.startswith(("/tmp", "\\tmp")):
        SANDBOX_BASE: Path = (Path(tempfile.gettempdir()) / "eval_sandbox").resolve()
    else:
        SANDBOX_BASE: Path = (
            Path(_raw_sandbox) if Path(_raw_sandbox).is_absolute()
            else (BASE_DIR / _raw_sandbox)
        ).resolve()

    INFERENCE_THREADS: int = int(os.getenv("INFERENCE_THREADS", "2"))
    DEPENDENCY_TIMEOUT_SECONDS: int = int(os.getenv("DEPENDENCY_TIMEOUT_SECONDS", "600"))
    VERBOSE: bool = os.getenv("VERBOSE", "false").lower() in ("true", "1", "yes")


config = SlaveConfig()
