import os
import sys
import csv
import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import config
from hardware import (
    get_temperature,
    get_throttled_flag,
    set_performance_governor,
    wait_for_fair_conditions,
    drop_os_caches
)
from sandbox import SandboxManager


def parse_csv_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def parse_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class SlaveRunner:
    def __init__(self):
        self.sandbox_mgr = SandboxManager()

    def run(self):
        print("=" * 60)
        print(f"TESA Slave Worker [{config.SLAVE_ID}] Initializing")
        print(f"Master URL: {config.MASTER_URL}")
        print(f"Max Temp Gate: {config.MAX_TEMPERATURE_CELSIUS}°C")
        print("=" * 60)

        # Set CPU governor to performance mode for consistent benchmarking
        set_performance_governor()

        while True:
            try:
                self.poll_and_execute()
            except KeyboardInterrupt:
                print("\nSlave interrupted by user. Exiting cleanly...")
                break
            except Exception as e:
                print(f"[Worker Error] Unexpected error in worker loop: {e}")
                time.sleep(config.POLL_INTERVAL_SECONDS)

    def poll_and_execute(self):
        current_temp = get_temperature()
        throttled_flag = get_throttled_flag()

        poll_url = f"{config.MASTER_URL}/api/v1/jobs/poll"
        params = {
            "slave_id": config.SLAVE_ID,
            "current_temp": current_temp,
            "throttled_flag": throttled_flag
        }

        try:
            resp = requests.get(poll_url, params=params, timeout=10)
        except requests.RequestException as e:
            print(f"[Master Unreachable] Could not connect to {config.MASTER_URL}: {e}. Retrying in {config.POLL_INTERVAL_SECONDS}s...")
            time.sleep(config.POLL_INTERVAL_SECONDS)
            return

        if resp.status_code == 204:
            # No jobs ready
            time.sleep(config.POLL_INTERVAL_SECONDS)
            return

        if resp.status_code != 200:
            print(f"[Poll Error] Received HTTP {resp.status_code}: {resp.text}")
            time.sleep(config.POLL_INTERVAL_SECONDS)
            return

        job_data = resp.json()
        job_id = job_data["job_id"]
        bundle_url = job_data["bundle_url"]
        test_dir_name = job_data.get("test_data_dir_name", "day_1")
        timeout_seconds = job_data.get("timeout_seconds", 300)

        print("\n" + "=" * 60)
        print(f"[JOB RECEIVED] ID: {job_id} | Test Data Dir: {test_dir_name}")
        print("=" * 60)

        # Step 1: Download Bundle
        temp_bundle_path = config.SANDBOX_BASE / f"tmp_{job_id}.zip"
        temp_bundle_path.parent.mkdir(parents=True, exist_ok=True)

        full_bundle_url = f"{config.MASTER_URL}{bundle_url}"
        print(f"Downloading bundle from {full_bundle_url} ...")
        try:
            with requests.get(full_bundle_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(temp_bundle_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
        except Exception as e:
            self._send_result(job_id, status="FAILED", error_message=f"Bundle download failed: {e}")
            return

        # Step 2: Setup Sandbox and Virtual Environment
        print(f"Setting up isolated sandbox environment for {job_id} ...")
        sandbox_dir = None
        try:
            sandbox_dir = self.sandbox_mgr.setup_sandbox(job_id, temp_bundle_path, config.SANDBOX_BASE)
        except Exception as e:
            self._send_result(job_id, status="FAILED", error_message=f"Sandbox creation failed: {e}")
            if temp_bundle_path.exists():
                temp_bundle_path.unlink()
            return

        # Step 3: Install Dependencies
        print("Installing contestant dependencies ...")
        install_ok, install_logs = self.sandbox_mgr.install_dependencies(
            sandbox_dir,
            wheelhouse_dir=config.LOCAL_WHEELHOUSE,
            timeout_seconds=240
        )
        if not install_ok:
            print(f"Dependency installation failed for {job_id}!")
            self._send_result(
                job_id,
                status="FAILED",
                logs=install_logs,
                error_message="requirements.txt installation failed or timed out."
            )
            self._cleanup(sandbox_dir, temp_bundle_path)
            return

        # Step 4: Fairness & Thermal Gate
        print(f"Checking hardware fairness gate (temp <= {config.MAX_TEMPERATURE_CELSIUS}°C, throttled == 0x0) ...")
        gate_status = wait_for_fair_conditions(
            max_temp=config.MAX_TEMPERATURE_CELSIUS,
            check_interval=config.COOLDOWN_SLEEP_SECONDS
        )
        start_temp = gate_status.get("temperature", get_temperature())
        start_throttled = gate_status.get("throttled_flag", get_throttled_flag())

        # Step 5: Execute Inference
        venv_python = self.sandbox_mgr.get_venv_python(sandbox_dir)
        infer_script = sandbox_dir / "infer.py"
        test_dataset_path = config.LOCAL_DATASETS_BASE / test_dir_name
        output_dir = sandbox_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        weight_path = self.sandbox_mgr.find_weights(sandbox_dir)
        if not weight_path:
            self._send_result(
                job_id,
                status="FAILED",
                error_message="Model weight file not found in sandbox weights/ directory."
            )
            self._cleanup(sandbox_dir, temp_bundle_path)
            return

        cmd = [
            str(venv_python),
            str(infer_script),
            "--input", str(test_dataset_path),
            "--weights", str(weight_path),
            "--out", str(output_dir),
            "--threads", str(config.INFERENCE_THREADS)
        ]

        print(f"Executing: {' '.join(cmd)}")
        wall_start = time.perf_counter()
        try:
            res = subprocess.run(
                cmd,
                cwd=str(sandbox_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            wall_elapsed = time.perf_counter() - wall_start
            combined_logs = f"=== STDOUT ===\n{res.stdout}\n=== STDERR ===\n{res.stderr}"

            end_temp = get_temperature()
            end_throttled = get_throttled_flag()

            if res.returncode != 0:
                print(f"infer.py failed with exit code {res.returncode}!")
                self._send_result(
                    job_id,
                    status="FAILED",
                    logs=combined_logs,
                    error_message=f"infer.py exited with code {res.returncode}",
                    wall_time_s=wall_elapsed,
                    start_temp=start_temp,
                    end_temp=end_temp,
                    throttled_flag=end_throttled
                )
            else:
                print(f"Inference completed successfully in {wall_elapsed:.2f}s!")
                pred_json_file = output_dir / "predictions.json"
                pred_csv_file = output_dir / "predictions.csv"
                if pred_json_file.exists():
                    predictions = parse_json_file(pred_json_file)
                elif pred_csv_file.exists():
                    predictions = parse_csv_file(pred_csv_file)
                else:
                    predictions = None

                timings = parse_csv_file(output_dir / "timings.csv")
                benchmark = parse_json_file(output_dir / "benchmark.json")

                self._send_result(
                    job_id,
                    status="SUCCESS",
                    predictions=predictions,
                    timings=timings,
                    benchmark=benchmark,
                    logs=combined_logs,
                    wall_time_s=wall_elapsed,
                    start_temp=start_temp,
                    end_temp=end_temp,
                    throttled_flag=end_throttled
                )
        except subprocess.TimeoutExpired:
            print(f"Inference timed out after {timeout_seconds}s!")
            self._send_result(
                job_id,
                status="FAILED",
                error_message=f"Execution timed out after {timeout_seconds} seconds."
            )
        except Exception as e:
            print(f"Execution error: {e}")
            self._send_result(job_id, status="FAILED", error_message=str(e))
        finally:
            # Step 6: Deep Cleanup
            self._cleanup(sandbox_dir, temp_bundle_path)

    def _send_result(
        self,
        job_id: str,
        status: str,
        predictions: Optional[Any] = None,
        timings: Optional[List[Dict[str, Any]]] = None,
        benchmark: Optional[Dict[str, Any]] = None,
        logs: str = "",
        error_message: Optional[str] = None,
        wall_time_s: Optional[float] = None,
        start_temp: Optional[float] = None,
        end_temp: Optional[float] = None,
        throttled_flag: Optional[str] = None
    ):
        return_url = f"{config.MASTER_URL}/api/v1/jobs/{job_id}/return"
        payload = {
            "slave_id": config.SLAVE_ID,
            "status": status,
            "predictions": predictions,
            "timings": timings,
            "benchmark": benchmark,
            "total_wall_time_s": wall_time_s,
            "logs": logs,
            "hardware_report": {
                "start_temp": start_temp,
                "end_temp": end_temp,
                "throttled_flag": throttled_flag or "0x0"
            },
            "error_message": error_message
        }

        try:
            print(f"Returning evaluation results for {job_id} to Master ...")
            r = requests.post(return_url, json=payload, timeout=20)
            if r.status_code == 200:
                print(f"[Result Recorded] Job {job_id} returned successfully.")
            else:
                print(f"[Return Error] Master returned {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[Return Failed] Could not deliver results for {job_id} to Master: {e}")

    def _cleanup(self, sandbox_dir: Optional[Path], temp_bundle_path: Optional[Path]):
        print("Cleaning up sandbox, virtual environment, and cache ...")
        if sandbox_dir:
            self.sandbox_mgr.purge_sandbox(sandbox_dir)
        if temp_bundle_path and temp_bundle_path.exists():
            try:
                temp_bundle_path.unlink()
            except Exception:
                pass
        drop_os_caches()
        time.sleep(config.COOLDOWN_SLEEP_SECONDS)


if __name__ == "__main__":
    runner = SlaveRunner()
    runner.run()
