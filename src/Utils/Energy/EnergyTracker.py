import csv
import threading
import time
from datetime import datetime
from pathlib import Path

from codecarbon import EmissionsTracker

try:
    import pynvml
    _PYNVML_IMPORT_ERROR = None
except ImportError as e:
    pynvml = None
    _PYNVML_IMPORT_ERROR = e


class _NVMLSampler:
    """
    Polls NVML directly (independent of CodeCarbon) to record per-GPU energy and
    peak power draw for a run. Uses the hardware energy counter
    (nvmlDeviceGetTotalEnergyConsumption) when the GPU supports it (Volta+), and
    falls back to trapezoidal integration of sampled power draw otherwise.
    """

    def __init__(self, interval_sec=1.0):
        self.interval_sec = interval_sec
        self.available = False
        self._stop_event = threading.Event()
        self._thread = None
        self._handles = []
        self._gpu_names = []
        self._power_energy_joules = []
        self._peak_power_w = []
        self._start_energy_counters = []

        if pynvml is None:
            print(f"[EnergyTracker] pynvml not installed ({_PYNVML_IMPORT_ERROR}). "
                  f"Skipping direct NVML GPU sampling.")
            return

        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            print(f"[EnergyTracker] NVML unavailable ({e}). Skipping direct NVML GPU sampling.")
            return

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            self._handles.append(handle)
            self._gpu_names.append(name)
            self._power_energy_joules.append(0.0)
            self._peak_power_w.append(0.0)

        self.available = count > 0

    def _sample_loop(self):
        last_time = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now
            for idx, handle in enumerate(self._handles):
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    self._power_energy_joules[idx] += power_w * dt
                    self._peak_power_w[idx] = max(self._peak_power_w[idx], power_w)
                except Exception:
                    pass
            self._stop_event.wait(self.interval_sec)

    def start(self):
        if not self.available:
            return

        self._start_energy_counters = []
        for handle in self._handles:
            try:
                self._start_energy_counters.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
            except Exception:
                self._start_energy_counters.append(None)

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.available:
            return {"gpu_available": False, "per_gpu": [], "total_energy_joules": None, "total_energy_kwh": None}

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec * 2)

        per_gpu = []
        total_energy_j = 0.0
        for idx, handle in enumerate(self._handles):
            start_counter = self._start_energy_counters[idx]
            try:
                end_counter = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            except Exception:
                end_counter = None

            if start_counter is not None and end_counter is not None and end_counter >= start_counter:
                # Hardware energy counter is reported in millijoules.
                energy_j = (end_counter - start_counter) / 1000.0
                source = "nvml_energy_counter"
            else:
                energy_j = self._power_energy_joules[idx]
                source = "power_integration"

            total_energy_j += energy_j
            per_gpu.append({
                "index": idx,
                "name": self._gpu_names[idx],
                "energy_joules": energy_j,
                "energy_source": source,
                "peak_power_w": self._peak_power_w[idx],
            })

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

        return {
            "gpu_available": True,
            "per_gpu": per_gpu,
            "total_energy_joules": total_energy_j,
            "total_energy_kwh": total_energy_j / 3_600_000.0,
        }


class EnergyTracker:
    """
    Context manager that records energy consumption for one pipeline stage:
      - CodeCarbon: CPU/GPU/RAM energy (kWh) and estimated CO2eq, appended to
        '<output_dir>/codecarbon_emissions.csv' (one row per run).
      - Direct NVML sampling: per-GPU energy (Joules) and peak power draw,
        recorded independently of CodeCarbon as a cross-check.

    Both are summarized into one row per run in '<output_dir>/energy_summary.csv'.
    If no NVIDIA GPU/driver is present (e.g. running on CPU-only or non-NVIDIA
    hardware), GPU fields are recorded as unavailable rather than raising.

    Usage:
        with EnergyTracker("a_Cache_initial_search_files", output_dir=".../Output/Energy_Logs"):
            ... stage code ...
    """

    def __init__(self, stage_name, output_dir, nvml_interval_sec=1.0):
        self.stage_name = stage_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_csv = self.output_dir / "energy_summary.csv"
        self._codecarbon_tracker = None
        self._nvml_sampler = _NVMLSampler(interval_sec=nvml_interval_sec)
        self._start_time = None

    def __enter__(self):
        self._start_time = datetime.now()
        self._nvml_sampler.start()

        try:
            self._codecarbon_tracker = EmissionsTracker(
                project_name=self.stage_name,
                output_dir=str(self.output_dir),
                output_file="codecarbon_emissions.csv",
                save_to_file=True,
                log_level="error",
                tracking_mode="process",
            )
            self._codecarbon_tracker.start()
        except Exception as e:
            print(f"[EnergyTracker] CodeCarbon failed to start ({e}). Continuing without it.")
            self._codecarbon_tracker = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration_sec = (end_time - self._start_time).total_seconds()

        codecarbon_kwh = None
        codecarbon_co2_kg = None
        if self._codecarbon_tracker is not None:
            try:
                self._codecarbon_tracker.stop()
                emissions_data = self._codecarbon_tracker.final_emissions_data
                codecarbon_kwh = emissions_data.energy_consumed
                codecarbon_co2_kg = emissions_data.emissions
            except Exception as e:
                print(f"[EnergyTracker] CodeCarbon failed to stop cleanly ({e}).")

        nvml_result = self._nvml_sampler.stop()
        status = "error" if exc_type else "success"
        self._write_summary_row(self._start_time, end_time, duration_sec,
                                 codecarbon_kwh, codecarbon_co2_kg, nvml_result, status)

        gpu_summary = f"{nvml_result['total_energy_kwh']:.6f} kWh" if nvml_result["gpu_available"] else "N/A (no GPU)"
        print(f"[EnergyTracker] Stage '{self.stage_name}' finished in {duration_sec:.1f}s "
              f"[{status}]. CodeCarbon: "
              f"{codecarbon_kwh if codecarbon_kwh is not None else 'N/A'} kWh, "
              f"{codecarbon_co2_kg if codecarbon_co2_kg is not None else 'N/A'} kgCO2eq. "
              f"NVML GPU energy: {gpu_summary}. Log: {self.summary_csv}")

        # Never swallow the caller's exception.
        return False

    def _write_summary_row(self, start_time, end_time, duration_sec,
                            codecarbon_kwh, codecarbon_co2_kg, nvml_result, status):
        per_gpu = nvml_result.get("per_gpu", [])
        row = {
            "stage": self.stage_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_sec": round(duration_sec, 3),
            "status": status,
            "codecarbon_energy_kwh": codecarbon_kwh,
            "codecarbon_co2eq_kg": codecarbon_co2_kg,
            "nvml_gpu_available": nvml_result.get("gpu_available", False),
            "nvml_gpu_names": "; ".join(g["name"] for g in per_gpu),
            "nvml_gpu_energy_joules": nvml_result.get("total_energy_joules"),
            "nvml_gpu_energy_kwh": nvml_result.get("total_energy_kwh"),
            "nvml_gpu_peak_power_w": max((g["peak_power_w"] for g in per_gpu), default=None),
            "nvml_gpu_energy_source": "; ".join(g["energy_source"] for g in per_gpu),
        }

        file_exists = self.summary_csv.exists()
        with open(self.summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
