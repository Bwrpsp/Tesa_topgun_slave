import os
import re
import subprocess
import time
from typing import Tuple, List, Dict, Any


def get_temperature() -> float:
    """
    Reads CPU temperature from Raspberry Pi hardware.
    Tries vcgencmd first, then sysfs thermal zone, with fallback.
    """
    # 1. Try vcgencmd
    try:
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            match = re.search(r"temp=([\d\.]+)'C", res.stdout)
            if match:
                return float(match.group(1))
    except Exception:
        pass

    # 2. Try Linux sysfs thermal zone
    thermal_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(thermal_path):
        try:
            with open(thermal_path, "r") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            pass

    # Fallback for dev / mock environments
    return 40.0


def get_throttled_flag() -> str:
    """Reads the vcgencmd get_throttled hex status."""
    try:
        res = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0:
            match = re.search(r"throttled=(0x[0-9a-fA-F]+)", res.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass
    return "0x0"


def decode_throttled(hex_str: str) -> Tuple[bool, List[str]]:
    """
    Decodes Raspberry Pi throttled bitmask.
    Returns (is_active_throttle, list_of_reasons).
    """
    try:
        val = int(hex_str, 16)
    except ValueError:
        return False, []

    reasons = []
    # Active conditions (bits 0-3)
    if val & 0x1:
        reasons.append("Under-voltage detected")
    if val & 0x2:
        reasons.append("Arm frequency capped")
    if val & 0x4:
        reasons.append("Currently throttled")
    if val & 0x8:
        reasons.append("Soft temperature limit active")

    # Past conditions (bits 16-19)
    if val & 0x10000:
        reasons.append("Under-voltage has previously occurred")
    if val & 0x20000:
        reasons.append("Arm frequency capping has previously occurred")
    if val & 0x40000:
        reasons.append("Throttling has previously occurred")
    if val & 0x80000:
        reasons.append("Soft temperature limit has previously occurred")

    is_currently_throttled = bool(val & 0xF)
    return is_currently_throttled, reasons


def set_performance_governor() -> bool:
    """
    Sets CPU governor to 'performance' to avoid dynamic scaling latency jitter.
    """
    try:
        cpu_path = "/sys/devices/system/cpu"
        if not os.path.exists(cpu_path):
            return False

        # Attempt to set via shell
        cmd = "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
        subprocess.run(cmd, shell=True, capture_output=True, timeout=3)
        return True
    except Exception:
        return False


def drop_os_caches():
    """Flushes dirty pages and clears memory page cache for consistent RAM state."""
    try:
        subprocess.run(["sync"], timeout=3)
        subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True, capture_output=True, timeout=3)
    except Exception:
        pass


def wait_for_fair_conditions(
    max_temp: float = 50.0,
    check_interval: int = 10,
    max_wait_seconds: int = 300
) -> Dict[str, Any]:
    """
    Fairness Gate:
    Ensures Raspberry Pi is sufficiently cool and not throttled before execution begins.
    """
    start_wait = time.time()
    while True:
        temp = get_temperature()
        throttled_hex = get_throttled_flag()
        is_active_throttled, reasons = decode_throttled(throttled_hex)

        if temp <= max_temp and not is_active_throttled:
            return {
                "ready": True,
                "temperature": temp,
                "throttled_flag": throttled_hex,
                "wait_seconds": round(time.time() - start_wait, 2)
            }

        elapsed = time.time() - start_wait
        if elapsed >= max_wait_seconds:
            return {
                "ready": False,
                "temperature": temp,
                "throttled_flag": throttled_hex,
                "warning": f"Cooldown timeout exceeded after {elapsed:.1f}s. Proceeding with current temp {temp:.1f}C"
            }

        print(f"[Fairness Gate] Cooling down... Current temp: {temp:.1f}°C (Threshold: {max_temp}°C, Throttled: {throttled_hex}). Sleeping {check_interval}s...")
        time.sleep(check_interval)
