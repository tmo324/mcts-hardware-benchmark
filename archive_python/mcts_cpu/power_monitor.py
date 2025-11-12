#!/usr/bin/env python3
"""
Power Monitoring for Multi-Platform Benchmarking
================================================

Intelligent power measurement using best available method for each platform:
  - Linux Intel CPU: RAPL (accurate)
  - Linux AMD CPU: psutil estimation
  - NVIDIA GPU: nvidia-smi
  - Fallback: CPU usage estimation

Author: MCTS Hardware Benchmark Project
"""

import os
import platform
import subprocess
import time
from typing import Dict, Optional, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PowerMonitor:
    """Cross-platform power monitoring"""

    def __init__(self, force_cpu=False):
        """
        Initialize PowerMonitor

        Args:
            force_cpu: If True, force CPU-only monitoring (skip GPU detection)
        """
        self.platform = platform.system()
        self.force_cpu = force_cpu
        self.method = self._detect_best_method()
        self.baseline_power = None

        # For RAPL
        self.rapl_domains = []
        if self.method == 'rapl':
            self._init_rapl()

        # For GPU
        self.gpu_available = False
        if self.method == 'nvidia-smi':
            self.gpu_available = self._check_nvidia_gpu()

        print(f"[PowerMonitor] Using method: {self.method}")

    def _detect_best_method(self) -> str:
        """Detect best available power measurement method"""

        # If force_cpu is True, skip GPU detection
        if not self.force_cpu:
            # Check for NVIDIA GPU first
            if self._check_nvidia_gpu():
                return 'nvidia-smi'

        # Check for Intel RAPL (Linux only)
        if self.platform == 'Linux':
            if os.path.exists('/sys/class/powercap/intel-rapl'):
                # Check if we can actually read RAPL files
                if self._can_read_rapl():
                    return 'rapl'

        # Fallback to psutil estimation
        if PSUTIL_AVAILABLE:
            return 'psutil'

        return 'none'

    def _check_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'],
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def _can_read_rapl(self) -> bool:
        """Check if we have permission to read RAPL energy files"""
        rapl_base = '/sys/class/powercap/intel-rapl'
        try:
            for entry in os.listdir(rapl_base):
                if entry.startswith('intel-rapl:'):
                    energy_file = os.path.join(rapl_base, entry, 'energy_uj')
                    if os.path.exists(energy_file):
                        # Try to read the file
                        with open(energy_file) as f:
                            f.read()
                        return True  # If we can read at least one, RAPL is accessible
        except (OSError, PermissionError):
            return False
        return False

    def _init_rapl(self):
        """Initialize RAPL energy counters"""
        rapl_base = '/sys/class/powercap/intel-rapl'
        if not os.path.exists(rapl_base):
            return

        for entry in os.listdir(rapl_base):
            if entry.startswith('intel-rapl:'):
                domain_path = os.path.join(rapl_base, entry)
                name_file = os.path.join(domain_path, 'name')
                energy_file = os.path.join(domain_path, 'energy_uj')

                if os.path.exists(name_file) and os.path.exists(energy_file):
                    with open(name_file) as f:
                        domain_name = f.read().strip()
                    self.rapl_domains.append({
                        'name': domain_name,
                        'energy_file': energy_file
                    })

    def start_measurement(self) -> Dict:
        """Start power measurement, return initial state"""
        if self.method == 'rapl':
            return self._read_rapl_energy()
        elif self.method == 'nvidia-smi':
            return self._read_gpu_power()
        elif self.method == 'psutil':
            return {'cpu_percent': psutil.cpu_percent(interval=0.1)}
        return {}

    def stop_measurement(self, start_state: Dict, elapsed_time: float) -> Dict:
        """
        Stop measurement and calculate energy

        Args:
            start_state: State from start_measurement()
            elapsed_time: Time elapsed in seconds

        Returns:
            dict with power_mw and energy_uj
        """
        if self.method == 'rapl':
            end_state = self._read_rapl_energy()
            return self._calculate_rapl_energy(start_state, end_state, elapsed_time)

        elif self.method == 'nvidia-smi':
            end_state = self._read_gpu_power()
            return self._calculate_gpu_energy(start_state, end_state, elapsed_time)

        elif self.method == 'psutil':
            end_cpu = psutil.cpu_percent(interval=0.1)
            avg_cpu = (start_state['cpu_percent'] + end_cpu) / 2
            return self._estimate_cpu_energy(avg_cpu, elapsed_time)

        return {'power_mw': 0, 'energy_uj': 0, 'method': 'none'}

    def _read_rapl_energy(self) -> Dict:
        """Read RAPL energy counters"""
        energy = {}
        for domain in self.rapl_domains:
            try:
                with open(domain['energy_file']) as f:
                    energy[domain['name']] = float(f.read().strip())
            except:
                pass
        return energy

    def _calculate_rapl_energy(self, start: Dict, end: Dict, elapsed: float) -> Dict:
        """Calculate energy from RAPL readings"""
        total_energy_uj = 0
        for domain_name in start:
            if domain_name in end:
                delta = end[domain_name] - start[domain_name]
                # Handle counter wraparound (rare)
                if delta < 0:
                    delta += 2**32
                total_energy_uj += delta

        power_mw = (total_energy_uj / elapsed) / 1000 if elapsed > 0 else 0

        return {
            'power_mw': power_mw,
            'energy_uj': total_energy_uj,
            'method': 'rapl',
            'domains': len(start)
        }

    def _read_gpu_power(self) -> Dict:
        """Read GPU power via nvidia-smi"""
        try:
            # Determine which GPU to query based on CUDA_VISIBLE_DEVICES
            gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            # If multiple GPUs specified, use the first one
            if ',' in gpu_id:
                gpu_id = gpu_id.split(',')[0]

            cmd = ['nvidia-smi', f'-i', gpu_id, '--query-gpu=power.draw',
                   '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                power_w = float(result.stdout.strip())
                return {'power_w': power_w, 'timestamp': time.time()}
        except Exception as e:
            # If CUDA_VISIBLE_DEVICES is set but fails, try default GPU 0
            try:
                cmd = ['nvidia-smi', '-i', '0', '--query-gpu=power.draw',
                       '--format=csv,noheader,nounits']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    power_w = float(result.stdout.strip())
                    return {'power_w': power_w, 'timestamp': time.time()}
            except:
                pass
        return {'power_w': 0, 'timestamp': time.time()}

    def _calculate_gpu_energy(self, start: Dict, end: Dict, elapsed: float) -> Dict:
        """Calculate GPU energy"""
        avg_power_w = (start.get('power_w', 0) + end.get('power_w', 0)) / 2
        avg_power_mw = avg_power_w * 1000
        energy_uj = avg_power_mw * elapsed * 1000  # mW * s = mJ, * 1000 = µJ

        return {
            'power_mw': avg_power_mw,
            'energy_uj': energy_uj,
            'method': 'nvidia-smi'
        }

    def _estimate_cpu_energy(self, cpu_percent: float, elapsed: float) -> Dict:
        """Estimate CPU energy from utilization"""
        # Get CPU info for TDP estimation
        cpu_info = self._get_cpu_info()

        # Estimate TDP based on CPU model
        tdp_w = self._estimate_tdp(cpu_info)

        # Power = Idle + (TDP - Idle) * CPU%
        idle_power_w = tdp_w * 0.2  # Assume 20% idle
        active_power_w = tdp_w - idle_power_w

        avg_power_w = idle_power_w + (active_power_w * cpu_percent / 100)
        avg_power_mw = avg_power_w * 1000
        energy_uj = avg_power_mw * elapsed * 1000

        return {
            'power_mw': avg_power_mw,
            'energy_uj': energy_uj,
            'method': 'psutil_estimation',
            'cpu_percent': cpu_percent,
            'estimated_tdp_w': tdp_w
        }

    def _get_cpu_info(self) -> Dict:
        """Get CPU information"""
        info = {'model': 'Unknown', 'cores': 1}

        if PSUTIL_AVAILABLE:
            info['cores'] = psutil.cpu_count(logical=False) or 1

        try:
            if self.platform == 'Linux':
                with open('/proc/cpuinfo') as f:
                    for line in f:
                        if 'model name' in line:
                            info['model'] = line.split(':')[1].strip()
                            break
            else:
                info['model'] = platform.processor()
        except:
            pass

        return info

    def _estimate_tdp(self, cpu_info: Dict) -> float:
        """Estimate CPU TDP from model name"""
        model = cpu_info['model'].lower()
        cores = cpu_info['cores']

        # AMD Threadripper estimation
        if 'threadripper' in model:
            if '5945wx' in model or '5955wx' in model:
                return 280.0  # TDP for Threadripper PRO 5000 series
            return 280.0

        # Intel Xeon estimation
        if 'xeon' in model:
            if 'platinum' in model:
                return 270.0
            elif 'gold' in model:
                return 205.0
            return 150.0

        # Intel Core estimation
        if 'i9' in model:
            return 125.0 if cores > 8 else 65.0
        elif 'i7' in model:
            return 65.0 if cores > 4 else 45.0
        elif 'i5' in model:
            return 65.0

        # AMD Ryzen estimation
        if 'ryzen' in model:
            if '9' in model:
                return 105.0
            elif '7' in model:
                return 65.0
            return 65.0

        # Default estimate
        return 65.0 * (cores / 4)


if __name__ == "__main__":
    # Test power monitoring
    print("Power Monitor Test")
    print("=" * 60)

    monitor = PowerMonitor()

    # Run a short test
    print(f"\nRunning 2-second test...")
    start_state = monitor.start_measurement()

    # Simulate some work
    time.sleep(2.0)

    result = monitor.stop_measurement(start_state, 2.0)

    print(f"\nResults:")
    print(f"  Method: {result.get('method', 'unknown')}")
    print(f"  Power: {result.get('power_mw', 0):.1f} mW")
    print(f"  Energy: {result.get('energy_uj', 0):.1f} µJ")
    if 'cpu_percent' in result:
        print(f"  CPU%: {result['cpu_percent']:.1f}%")
