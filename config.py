import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()


class SlaveConfig:
    MASTER_URL: str = os.getenv("MASTER_URL", "http://127.0.0.1:8000").rstrip("/")
    SECRET_TOKEN: str = os.getenv("SECRET_TOKEN", "tesa_topgun_secret_key_2026")
    SLAVE_ID: str = os.getenv("SLAVE_ID", "pi-node-01")

    POLL_INTERVAL_SECONDS: int = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
    MAX_TEMPERATURE_CELSIUS: float = float(os.getenv("MAX_TEMPERATURE_CELSIUS", "50.0"))
    COOLDOWN_SLEEP_SECONDS: int = int(os.getenv("COOLDOWN_SLEEP_SECONDS", "10"))

    LOCAL_DATASETS_BASE: Path = Path(os.getenv("LOCAL_DATASETS_BASE", "test_datasets"))
    LOCAL_WHEELHOUSE: Path = Path(os.getenv("LOCAL_WHEELHOUSE", "/opt/pi_wheels"))
    SANDBOX_BASE: Path = Path(os.getenv("SANDBOX_BASE", "/tmp/eval_sandbox"))

    INFERENCE_THREADS: int = int(os.getenv("INFERENCE_THREADS", "2"))


config = SlaveConfig()
