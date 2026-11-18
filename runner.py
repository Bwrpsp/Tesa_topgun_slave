import os
import sys
import csv
import json
import time
import argparse
import threading
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from config import config
from hardware import (
    get_temperature,
    get_throttled_flag,
    set_performance_governor,
    wait_for_fair_conditions,
    drop_os_caches
)
from sandbox import SandboxManager


def format_bytes(size: int) -> str:
    """Formats raw byte counts into human-readable strings."""
    s = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if s < 1024.0:
            return f"{s:.2f} {unit}"
        s /= 1024.0
    return f"{s:.2f} TB"


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


def run_command_dual_stream(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    verbose: bool = False
) -> Tuple[int, str, str]:
    """
    Executes a subprocess, capturing stdout and stderr separately while
    streaming them in real time to the console when verbose=True.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    def reader(pipe, container: List[str], prefix: str):
        for line in iter(pipe.readline, ""):
            container.append(line)
            if verbose:
                sys.stdout.write(f"{prefix}{line}")
                sys.stdout.flush()
        pipe.close()

    t_out = threading.Thread(target=reader, args=(proc.stdout, stdout_lines, "   [infer.py stdout] "))
    t_err = threading.Thread(target=reader, args=(proc.stderr, stderr_lines, "   [infer.py stderr] "))
    t_out.daemon = True
    t_err.daemon = True
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)
        raise subprocess.TimeoutExpired(
            cmd,
            timeout,
            output="".join(stdout_lines),
            stderr="".join(stderr_lines)
        )

    t_out.join()
    t_err.join()
    return proc.returncode, "".join(stdout_lines), "".join(stderr_lines)


class SlaveRunner:
    def __init__(
        self,
        verbose: Optional[bool] = None,
        slave_id: Optional[str] = None,
        master_url: Optional[str] = None,
        day: Optional[int] = None
    ):
        self.sandbox_mgr = SandboxManager()
        self.verbose = verbose if verbose is not None else config.VERBOSE
        self.slave_id = slave_id or config.SLAVE_ID
        self.master_url = (master_url or config.MASTER_URL).rstrip("/")
        self.day = day

    def run(self):
        print("=" * 65)
        print(f"TESA Slave Worker [{self.slave_id}] Initializing")
        print(f"Master URL:          {self.master_url}")
        print(f"Max Temp Gate:       {config.MAX_TEMPERATURE_CELSIUS}°C")
        print(f"Dependency Timeout:  {config.DEPENDENCY_TIMEOUT_SECONDS}s")
        print(f"Verbose Mode:        {'ON (Full real-time streaming)' if self.verbose else 'OFF'}")
        print("=" * 65)

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

        poll_url = f"{self.master_url}/api/v1/jobs/poll"
        params = {
            "slave_id": self.slave_id,
            "current_temp": current_temp,
            "throttled_flag": throttled_flag
        }
        if self.day:
            params["day"] = self.day

        if self.verbose:
            sys.stdout.write(f"\r[Polling] Checking Master for jobs (Temp: {current_temp:.1f}°C, Throttled: {throttled_flag}) ...")
            sys.stdout.flush()

        try:
            resp = requests.get(poll_url, params=params, timeout=10)
        except requests.RequestException as e:
            if self.verbose:
                print()
            print(f"[Master Unreachable] Could not connect to {self.master_url}: {e}. Retrying in {config.POLL_INTERVAL_SECONDS}s...")
            time.sleep(config.POLL_INTERVAL_SECONDS)
            return

        if resp.status_code == 204:
            # No jobs ready
            time.sleep(config.POLL_INTERVAL_SECONDS)
            return

        if self.verbose:
            print()  # Break line from polling status

        if resp.status_code != 200:
            print(f"[Poll Error] Received HTTP {resp.status_code}: {resp.text}")
            time.sleep(config.POLL_INTERVAL_SECONDS)
            return

        job_data = resp.json()
        job_id = job_data["job_id"]
        team_name = job_data.get("team_name", job_id)
        bundle_url = job_data["bundle_url"]
        test_dir_name = job_data.get("test_data_dir_name", "day_1")
        timeout_seconds = job_data.get("timeout_seconds", 300)

        print("\n" + "=" * 65)
        print(f"[JOB RECEIVED] ID: {job_id} | Team: {team_name}")
        print(f"Test Data Dir: {test_dir_name} | Max Inference Timeout: {timeout_seconds}s")
        print("=" * 65)

        # Step 1: Download Bundle
        temp_bundle_path = config.SANDBOX_BASE / f"tmp_{job_id}.zip"
        temp_bundle_path.parent.mkdir(parents=True, exist_ok=True)

        full_bundle_url = f"{self.master_url}{bundle_url}"
        print(f"Downloading bundle from {full_bundle_url} ...")
        t_dl_start = time.perf_counter()
        try:
            with requests.get(full_bundle_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(temp_bundle_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            dl_size = temp_bundle_path.stat().st_size
            print(f"Downloaded bundle ({format_bytes(dl_size)}) in {time.perf_counter() - t_dl_start:.2f}s.")
        except Exception as e:
            self._send_result(job_id, status="FAILED", error_message=f"Bundle download failed: {e}")
            return

        # Step 2: Setup Sandbox and Virtual Environment
        print(f"Setting up isolated sandbox environment for {job_id} ...")
        sandbox_dir = None
        try:
            sandbox_dir = self.sandbox_mgr.setup_sandbox(
                job_id,
                temp_bundle_path,
                config.SANDBOX_BASE,
                verbose=self.verbose
            )
        except Exception as e:
            self._send_result(job_id, status="FAILED", error_message=f"Sandbox creation failed: {e}")
            if temp_bundle_path.exists():
                temp_bundle_path.unlink()
            return

        # Step 3: Install Dependencies
        print(f"Installing contestant dependencies (timeout: {config.DEPENDENCY_TIMEOUT_SECONDS}s) ...")
        t_install_start = time.perf_counter()
        install_ok, install_logs = self.sandbox_mgr.install_dependencies(
            sandbox_dir,
            wheelhouse_dir=config.LOCAL_WHEELHOUSE,
            timeout_seconds=config.DEPENDENCY_TIMEOUT_SECONDS,
            verbose=self.verbose
        )
        if not install_ok:
            print(f"Dependency installation failed for {job_id} in {time.perf_counter() - t_install_start:.2f}s!")
            self._send_result(
                job_id,
                status="FAILED",
                logs=install_logs,
                error_message="requirements.txt installation failed or timed out."
            )
            self._cleanup(sandbox_dir, temp_bundle_path)
            return

        print(f"Dependencies installed successfully in {time.perf_counter() - t_install_start:.2f}s.")

        # Step 4: Fairness & Thermal Gate
        print(f"Checking hardware fairness gate (temp <= {config.MAX_TEMPERATURE_CELSIUS}°C, throttled == 0x0) ...")
        gate_status = wait_for_fair_conditions(
            max_temp=config.MAX_TEMPERATURE_CELSIUS,
            check_interval=config.COOLDOWN_SLEEP_SECONDS
        )
        start_temp = gate_status.get("temperature", get_temperature())
        start_throttled = gate_status.get("throttled_flag", get_throttled_flag())
        if self.verbose:
            print(f"   [Fairness Gate] Ready. Starting Temp: {start_temp:.1f}°C, Throttled: {start_throttled}")

        # Step 5: Execute Inference
        venv_python = self.sandbox_mgr.get_venv_python(sandbox_dir)
        infer_script = sandbox_dir / "infer.py"

        # Resolve test dataset path (handles both day_1 and day1)
        test_dataset_path = (config.LOCAL_DATASETS_BASE / test_dir_name).resolve()
        if not test_dataset_path.exists():
            alt_name = test_dir_name.replace("_", "")
            alt_path = (config.LOCAL_DATASETS_BASE / alt_name).resolve()
            if alt_path.exists():
                test_dataset_path = alt_path

        # Verify test dataset directory exists and has images
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        if not test_dataset_path.exists():
            msg = f"Test dataset directory does not exist: {test_dataset_path}"
            print(f"[INPUT ERROR] {msg}")
            self._send_result(job_id, status="FAILED", error_message=msg)
            self._cleanup(sandbox_dir, temp_bundle_path)
            return

        image_files = [p for p in test_dataset_path.rglob('*') if p.is_file() and p.suffix.lower() in valid_exts]
        if not image_files:
            msg = f"No test images found in {test_dataset_path}! Please copy test images (.jpg, .png, etc.) into this folder."
            print(f"[INPUT ERROR] {msg}")
            self._send_result(job_id, status="FAILED", error_message=msg)
            self._cleanup(sandbox_dir, temp_bundle_path)
            return

        print(f"Found {len(image_files)} test images in {test_dataset_path}")

        output_dir = sandbox_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        weights_dir = self.sandbox_mgr.get_weights_dir(sandbox_dir)
        if not weights_dir:
            self._send_result(
                job_id,
                status="FAILED",
                error_message="Weights folder not found or empty in sandbox weights/ directory."
            )
            self._cleanup(sandbox_dir, temp_bundle_path)
            return

        cmd = [
            str(venv_python),
            str(infer_script),
            "--input", str(test_dataset_path),
            "--weights", str(weights_dir),
            "--out", str(output_dir)
        ]

        print(f"Executing: {' '.join(cmd)}")
        if self.verbose:
            print("   --- [infer.py Live Output Stream Start] ---")

        wall_start = time.perf_counter()
        try:
            returncode, stdout_str, stderr_str = run_command_dual_stream(
                cmd,
                cwd=sandbox_dir,
                timeout=timeout_seconds,
                verbose=self.verbose
            )
            wall_elapsed = time.perf_counter() - wall_start
            combined_logs = f"=== STDOUT ===\n{stdout_str}\n=== STDERR ===\n{stderr_str}"

            # Smart Fallback: if infer.py expects a single weight file rather than a directory
            if returncode != 0 and ("IsADirectoryError" in stderr_str or "Errno 21" in stderr_str or "PermissionError" in stderr_str):
                weight_file = self.sandbox_mgr.find_weight_file(sandbox_dir)
                if weight_file:
                    print(f"\n[Smart Fallback] infer.py expects a single weight file rather than folder.")
                    print(f"[Smart Fallback] Retrying with file: {weight_file.name} ...")
                    weights_idx = cmd.index("--weights") + 1
                    cmd[weights_idx] = str(weight_file)
                    wall_start = time.perf_counter()
                    returncode, stdout_str, stderr_str = run_command_dual_stream(
                        cmd,
                        cwd=sandbox_dir,
                        timeout=timeout_seconds,
                        verbose=self.verbose
                    )
                    wall_elapsed = time.perf_counter() - wall_start
                    combined_logs = f"=== STDOUT ===\n{stdout_str}\n=== STDERR ===\n{stderr_str}"

            if self.verbose:
                print("   --- [infer.py Live Output Stream End] ---")

            end_temp = get_temperature()
            end_throttled = get_throttled_flag()

            if returncode != 0:
                print(f"infer.py failed with exit code {returncode}!")
                if not self.verbose:
                    if stderr_str.strip():
                        print(f"[infer.py STDERR]:\n{stderr_str.strip()}")
                    if stdout_str.strip():
                        print(f"[infer.py STDOUT]:\n{stdout_str.strip()}")
                self._send_result(
                    job_id,
                    status="FAILED",
                    logs=combined_logs,
                    error_message=f"infer.py exited with code {returncode}: {stderr_str.strip()}",
                    wall_time_s=wall_elapsed,
                    start_temp=start_temp,
                    end_temp=end_temp,
                    throttled_flag=end_throttled
                )
            else:
                print(f"Inference completed successfully in {wall_elapsed:.3f}s!")
                pred_json_file = output_dir / "predictions.json"
                pred_csv_file = output_dir / "predictions.csv"
                if pred_json_file.exists():
                    predictions = parse_json_file(pred_json_file)
                    if self.verbose:
                        print(f"   [Output] predictions.json detected ({type(predictions).__name__})")
                elif pred_csv_file.exists():
                    predictions = parse_csv_file(pred_csv_file)
                    if self.verbose:
                        print(f"   [Output] predictions.csv detected ({len(predictions)} rows)")
                else:
                    predictions = None
                    if self.verbose:
                        print("   [Output Warning] Neither predictions.json nor predictions.csv was generated in outputs/!")

                timings = parse_csv_file(output_dir / "timings.csv")
                benchmark = parse_json_file(output_dir / "benchmark.json")

                if self.verbose:
                    if timings:
                        print(f"   [Output] timings.csv detected ({len(timings)} records)")
                    if benchmark:
                        print(f"   [Output] benchmark.json detected: {benchmark}")

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
        return_url = f"{self.master_url}/api/v1/jobs/{job_id}/return"
        payload = {
            "slave_id": self.slave_id,
            "status": status,
            "predictions": predictions,
            "timings": timings,
            "benchmark": benchmark,
            "total_wall_time_s": wall_time_s,
            "logs": logs[-50000:] if logs else "",
            "hardware_report": {
                "start_temp": start_temp,
                "end_temp": end_temp,
                "throttled_flag": throttled_flag or "0x0"
            },
            "error_message": error_message
        }

        try:
            print(f"Returning evaluation results for {job_id} to Master ...")
            r = requests.post(return_url, json=payload, timeout=30)
            if r.status_code == 200:
                print(f"[Result Recorded] Job {job_id} returned successfully.")
                res_data = r.json()
                eval_data = res_data.get("evaluation")
                if eval_data:
                    eval_day = eval_data.get("day", 1)
                    if eval_day == 2:
                        print("============================================================")
                        print(f" [Day 2 Evaluation Summary for {job_id}]")
                        print(f"  * Latency:       {eval_data.get('latency', 0.0):.3f}s")
                        print(f"  * Count Acc:     {eval_data.get('count_accuracy', 0.0):.2f}%")
                        print(f"  * IoU >= 95%:    {eval_data.get('iou95', 0)} / {eval_data.get('total_gt_objects', 0)}")
                        print(f"  * IoU >= 90%:    {eval_data.get('iou90', 0)} / {eval_data.get('total_gt_objects', 0)}")
                        print(f"  * IoU Avg:       {eval_data.get('iou_average', 0.0):.4f}")
                        print(f"  * Composite:     {eval_data.get('composite_score', 0.0):.2f}")
                        print("============================================================")
                    elif eval_day == 1:
                        print("============================================================")
                        print(f" [Day 1 Evaluation Summary for {job_id}]")
                        print(f"  * Accuracy:      {eval_data.get('accuracy', 0.0):.2f}%")
                        print(f"  * Correct Pics:  {eval_data.get('correct_predictions', 0)} / {eval_data.get('total_images', 0)}")
                        print(f"  * Total Wall:    {eval_data.get('total_wall_time_s', 0.0):.3f}s")
                        print("============================================================")
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


def main():
    parser = argparse.ArgumentParser(description="TESA Slave Worker Node Daemon")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose mode: streams dependency installation and infer.py execution in real time"
    )
    parser.add_argument(
        "--slave-id",
        type=str,
        default=None,
        help=f"Override SLAVE_ID (default: {config.SLAVE_ID})"
    )
    parser.add_argument(
        "--day",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Target competition day (default: whatever active day Master server runs)"
    )
    parser.add_argument(
        "--master-url",
        type=str,
        default=None,
        help=f"Override MASTER_URL (default: {config.MASTER_URL})"
    )

    args = parser.parse_args()
    is_verbose = args.verbose or config.VERBOSE

    runner = SlaveRunner(
        verbose=is_verbose,
        slave_id=args.slave_id,
        master_url=args.master_url,
        day=args.day
    )
    runner.run()


if __name__ == "__main__":
    main()
