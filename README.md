# TESA TopGun 2026 - Raspberry Pi Slave Worker

Evaluation daemon designed to run on Raspberry Pi boards. Pulls inference tasks from the Master server, prepares an isolated virtual environment, enforces hardware fairness gates (temperature and throttling flags), executes contestant code, reports timings and outputs, and performs complete workspace cleanup.

---

## Features

- **Fairness & Consistency Gate:**
  - Checks Raspberry Pi thermal sensor (`vcgencmd measure_temp` / sysfs).
  - Verifies throttling bitmask (`vcgencmd get_throttled == 0x0`).
  - Cooldown loop pauses execution if temperature exceeds threshold (e.g., 50°C).
  - CPU governor set to `performance` mode to eliminate frequency ramping latency jitter.
- **Strict Sandbox Isolation:**
  - Dedicated virtual environment (`python3 -m venv .venv`) per contestant.
  - Dependencies installed from contestant `requirements.txt`.
  - Wheelhouse caching (`/opt/pi_wheels`) for rapid 5-second package installation.
- **Deep Cleanup:**
  - Completely purges contestant code, `.venv`, model weights, and outputs after every run.
  - Purges pip cache (`pip cache purge`).
  - Flushes Linux dirty buffers and page cache (`drop_caches`).
- **Dynamic Multi-Day Test Execution:**
  - Points to local pre-stored test images (`test_datasets/day_{N}/`) according to the active day assigned by Master.
  - Automatically captures `predictions.csv` (Day 1) or `predictions.json` (Day 2/3), along with `timings.csv` and `benchmark.json`.

---

## Directory Structure

```text
slave/
├── .env                 # MASTER_URL, SLAVE_ID, TEMP_THRESHOLD, paths
├── requirements.txt     # requests, psutil, python-dotenv
├── config.py            # Loads .env
├── hardware.py          # Pi thermal & throttling gates, CPU governor
├── sandbox.py           # Virtualenv lifecycle, pip wheelhouse, cleanup
├── runner.py            # Main worker daemon loop
└── test_datasets/       # Local test images
    ├── day_1/           # Day 1 test images
    ├── day_2/           # Day 2 test images
    └── day_3/           # Day 3 test images
```

---

## Raspberry Pi Setup Instructions

### 1. System Requirements & Optimization
On 64-bit Raspberry Pi OS:
```bash
# Optional: Set CPU governor to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Optional: Build or place pre-compiled wheels in /opt/pi_wheels to speed up pip installs
sudo mkdir -p /opt/pi_wheels
```

### 2. Install Runner Requirements
```bash
cd slave
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure `.env`
Edit `slave/.env`:
```ini
MASTER_URL=http://<MASTER_IP>:8000
SLAVE_ID=pi-node-01
MAX_TEMPERATURE_CELSIUS=50.0
LOCAL_DATASETS_BASE=test_datasets
```

### 4. Place Test Images
Store your test images locally in `slave/test_datasets/`:
```text
slave/test_datasets/
├── day_1/ (Day 1 test images)
├── day_2/ (Day 2 test images)
└── day_3/ (Day 3 test images)
```

### 5. Start the Slave Runner
```bash
python runner.py
```
The slave will begin polling the Master, awaiting assigned jobs.
