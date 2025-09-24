"""
Raspberry Pi Zero Performance Monitor for BitGen
Comprehensive monitoring of FLOPS, energy consumption, memory, power, and thermal profile
"""

import torch
import psutil
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import subprocess
import os

# Monitoring imports
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: CodeCarbon not available. Install with: pip install codecarbon")

try:
    from ptflops import get_model_complexity_info
    from thop import profile, clever_format
    FLOPS_AVAILABLE = True
except ImportError:
    FLOPS_AVAILABLE = False
    print("Warning: FLOPS calculation tools not available")

try:
    import RPi.GPIO as GPIO
    from gpiozero import CPUTemperature
    from w1thermsensor import W1ThermSensor
    RPI_SENSORS_AVAILABLE = True
except ImportError:
    RPI_SENSORS_AVAILABLE = False
    print("Warning: Raspberry Pi sensors not available (not running on Pi?)")

@dataclass
class PerformanceMetrics:
    """Performance metrics for Raspberry Pi Zero deployment"""
    timestamp: float
    # Model performance
    tokens_per_second: float
    latency_ms_per_token: float
    response_latency_ms: float

    # Memory metrics
    ram_usage_mb: float
    ram_peak_mb: float
    swap_usage_mb: float

    # Compute metrics
    cpu_usage_percent: float
    flops_count: Optional[int]

    # Energy metrics
    power_consumption_mw: float
    energy_consumed_mj: float
    carbon_emissions_g: float

    # Thermal metrics
    cpu_temperature_c: float
    gpu_temperature_c: Optional[float]

    # System metrics
    disk_usage_mb: float
    network_bytes: int

class RaspberryPiMonitor:
    """Comprehensive monitoring for Raspberry Pi Zero"""

    def __init__(self, output_dir: str = "monitoring_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Monitoring state
        self.monitoring_active = False
        self.metrics_history = deque(maxlen=10000)
        self.start_time = None

        # Carbon tracking
        self.carbon_tracker = None
        if CODECARBON_AVAILABLE:
            self.carbon_tracker = OfflineEmissionsTracker(
                country_iso_code="US",  # Adjust based on location
                output_dir=str(self.output_dir)
            )

        # Temperature sensors
        self.cpu_temp = None
        self.thermal_sensors = []
        if RPI_SENSORS_AVAILABLE:
            try:
                self.cpu_temp = CPUTemperature()
                self.thermal_sensors = W1ThermSensor.get_available_sensors()
            except Exception as e:
                print(f"Warning: Could not initialize temperature sensors: {e}")

        # Baseline measurements
        self.baseline_power = self._measure_baseline_power()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for monitoring"""
        log_file = self.output_dir / "monitoring.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _measure_baseline_power(self) -> float:
        """Measure baseline power consumption"""
        # Simple baseline measurement (would need actual power meter)
        try:
            # Read from /sys/class/thermal for basic power estimation
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000.0
                # Basic power estimation based on temperature
                baseline = 200 + (temp - 40) * 5  # Rough estimate in mW
                return max(150, min(500, baseline))
        except:
            return 200.0  # Default baseline for Pi Zero

    def calculate_model_flops(self, model: torch.nn.Module, input_shape: Tuple) -> Dict:
        """Calculate model FLOPS for given input shape"""
        if not FLOPS_AVAILABLE:
            self.logger.warning("FLOPS calculation tools not available")
            return {"flops": 0, "params": 0}

        try:
            # Method 1: ptflops
            flops, params = get_model_complexity_info(
                model,
                input_shape,
                print_per_layer_stat=False,
                verbose=False
            )

            # Parse the string output
            if isinstance(flops, str):
                flops_num = float(flops.split()[0])
                if 'G' in flops:
                    flops_num *= 1e9
                elif 'M' in flops:
                    flops_num *= 1e6
                elif 'K' in flops:
                    flops_num *= 1e3
            else:
                flops_num = flops

            if isinstance(params, str):
                params_num = float(params.split()[0])
                if 'M' in params:
                    params_num *= 1e6
                elif 'K' in params:
                    params_num *= 1e3
            else:
                params_num = params

            return {
                "flops": int(flops_num),
                "params": int(params_num),
                "flops_human": flops,
                "params_human": params
            }

        except Exception as e:
            self.logger.error(f"FLOPS calculation failed: {e}")

            # Fallback: rough estimation
            param_count = sum(p.numel() for p in model.parameters())
            # Very rough FLOPS estimation (2 * params for forward pass)
            estimated_flops = param_count * 2

            return {
                "flops": estimated_flops,
                "params": param_count,
                "flops_human": f"{estimated_flops/1e6:.1f}M",
                "params_human": f"{param_count/1e6:.1f}M"
            }

    def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring_active = True
        self.start_time = time.time()

        if self.carbon_tracker:
            self.carbon_tracker.start()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Started comprehensive monitoring for Raspberry Pi Zero")

    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return summary"""
        self.monitoring_active = False

        if self.carbon_tracker:
            self.carbon_tracker.stop()

        # Generate summary
        summary = self._generate_summary()

        # Save detailed results
        self._save_monitoring_results()

        self.logger.info("Stopped monitoring and saved results")
        return summary

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(1.0)  # Monitor every second
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        timestamp = time.time()

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Disk usage
        disk = psutil.disk_usage('/')

        # Network I/O
        net_io = psutil.net_io_counters()

        # Temperature
        cpu_temp = 0.0
        gpu_temp = None

        if self.cpu_temp:
            try:
                cpu_temp = self.cpu_temp.temperature
            except:
                pass

        # Try to get temperature from system files
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = int(f.read().strip()) / 1000.0
        except:
            pass

        # Power estimation (simplified)
        power_mw = self._estimate_power_consumption(cpu_percent, cpu_temp)

        # Energy calculation
        energy_mj = 0.0
        if hasattr(self, 'last_power_timestamp'):
            time_delta = timestamp - self.last_power_timestamp
            energy_mj = (power_mw / 1000.0) * time_delta / 1000.0  # Convert to millijoules

        self.last_power_timestamp = timestamp

        # Carbon emissions (from CodeCarbon if available)
        carbon_g = 0.0
        if self.carbon_tracker:
            try:
                # CodeCarbon tracks cumulative emissions
                carbon_g = getattr(self.carbon_tracker, 'final_emissions', 0.0) * 1000  # Convert to grams
            except:
                pass

        return PerformanceMetrics(
            timestamp=timestamp,
            tokens_per_second=0.0,  # Will be updated during inference
            latency_ms_per_token=0.0,
            response_latency_ms=0.0,
            ram_usage_mb=memory.used / 1024 / 1024,
            ram_peak_mb=memory.used / 1024 / 1024,  # Peak tracking would need separate logic
            swap_usage_mb=swap.used / 1024 / 1024,
            cpu_usage_percent=cpu_percent,
            flops_count=None,
            power_consumption_mw=power_mw,
            energy_consumed_mj=energy_mj,
            carbon_emissions_g=carbon_g,
            cpu_temperature_c=cpu_temp,
            gpu_temperature_c=gpu_temp,
            disk_usage_mb=disk.used / 1024 / 1024,
            network_bytes=net_io.bytes_sent + net_io.bytes_recv
        )

    def _estimate_power_consumption(self, cpu_percent: float, temperature_c: float) -> float:
        """Estimate power consumption based on CPU load and temperature"""
        # Base power consumption for Pi Zero
        base_power = 150.0  # mW

        # CPU load contribution
        cpu_power = (cpu_percent / 100.0) * 200.0  # Up to 200mW for CPU

        # Temperature contribution (rough approximation)
        temp_power = max(0, (temperature_c - 40.0) * 2.0)  # 2mW per degree above 40C

        total_power = base_power + cpu_power + temp_power
        return min(total_power, 800.0)  # Cap at reasonable maximum

    @contextmanager
    def measure_inference(self, input_text: str, model_name: str = "BitGen"):
        """Context manager for measuring inference performance"""
        # Pre-inference measurements
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_tokens = len(input_text.split())

        # Create inference metrics collector
        inference_data = {
            'model_name': model_name,
            'input_text': input_text,
            'input_tokens': start_tokens,
            'start_time': start_time,
            'start_memory_mb': start_memory / 1024 / 1024
        }

        try:
            yield inference_data
        finally:
            # Post-inference measurements
            end_time = time.time()
            end_memory = psutil.virtual_memory().used

            total_time_ms = (end_time - start_time) * 1000
            memory_delta_mb = (end_memory - start_memory) / 1024 / 1024

            # Calculate output tokens (if available)
            output_tokens = inference_data.get('output_tokens', 0)
            total_tokens = start_tokens + output_tokens

            # Calculate metrics
            tokens_per_second = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
            ms_per_token = total_time_ms / total_tokens if total_tokens > 0 else total_time_ms

            # Update latest metrics in history
            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                latest_metrics.tokens_per_second = tokens_per_second
                latest_metrics.latency_ms_per_token = ms_per_token
                latest_metrics.response_latency_ms = total_time_ms

            # Log inference results
            self.logger.info(f"Inference completed:")
            self.logger.info(f"  Total time: {total_time_ms:.2f}ms")
            self.logger.info(f"  Tokens/sec: {tokens_per_second:.2f}")
            self.logger.info(f"  Latency per token: {ms_per_token:.2f}ms")
            self.logger.info(f"  Memory delta: {memory_delta_mb:.2f}MB")

            # Store inference data
            inference_data.update({
                'end_time': end_time,
                'total_time_ms': total_time_ms,
                'output_tokens': output_tokens,
                'tokens_per_second': tokens_per_second,
                'latency_ms_per_token': ms_per_token,
                'memory_delta_mb': memory_delta_mb
            })

            # Save individual inference result
            self._save_inference_result(inference_data)

    def _save_inference_result(self, inference_data: Dict):
        """Save individual inference result"""
        inference_file = self.output_dir / "inference_results.jsonl"

        with open(inference_file, 'a') as f:
            json.dump(inference_data, f)
            f.write('\n')

    def _generate_summary(self) -> Dict:
        """Generate comprehensive monitoring summary"""
        if not self.metrics_history:
            return {}

        metrics_list = list(self.metrics_history)

        # Calculate averages and statistics
        summary = {
            'monitoring_duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'total_samples': len(metrics_list),

            # Performance averages
            'avg_tokens_per_second': np.mean([m.tokens_per_second for m in metrics_list if m.tokens_per_second > 0]),
            'avg_latency_ms_per_token': np.mean([m.latency_ms_per_token for m in metrics_list if m.latency_ms_per_token > 0]),
            'avg_response_latency_ms': np.mean([m.response_latency_ms for m in metrics_list if m.response_latency_ms > 0]),

            # Memory statistics
            'avg_ram_usage_mb': np.mean([m.ram_usage_mb for m in metrics_list]),
            'peak_ram_usage_mb': np.max([m.ram_usage_mb for m in metrics_list]),
            'avg_swap_usage_mb': np.mean([m.swap_usage_mb for m in metrics_list]),

            # CPU statistics
            'avg_cpu_usage_percent': np.mean([m.cpu_usage_percent for m in metrics_list]),
            'peak_cpu_usage_percent': np.max([m.cpu_usage_percent for m in metrics_list]),

            # Power and energy
            'avg_power_consumption_mw': np.mean([m.power_consumption_mw for m in metrics_list]),
            'peak_power_consumption_mw': np.max([m.power_consumption_mw for m in metrics_list]),
            'total_energy_consumed_mj': np.sum([m.energy_consumed_mj for m in metrics_list]),
            'total_carbon_emissions_g': np.max([m.carbon_emissions_g for m in metrics_list]),  # Cumulative

            # Thermal statistics
            'avg_cpu_temperature_c': np.mean([m.cpu_temperature_c for m in metrics_list]),
            'peak_cpu_temperature_c': np.max([m.cpu_temperature_c for m in metrics_list]),

            # System statistics
            'final_disk_usage_mb': metrics_list[-1].disk_usage_mb if metrics_list else 0,
            'network_bytes_total': metrics_list[-1].network_bytes if metrics_list else 0,
        }

        # Calculate energy efficiency
        if summary['total_energy_consumed_mj'] > 0 and summary['avg_tokens_per_second'] > 0:
            summary['energy_per_token_mj'] = summary['total_energy_consumed_mj'] / (
                summary['avg_tokens_per_second'] * summary['monitoring_duration_seconds']
            )

        return summary

    def _save_monitoring_results(self):
        """Save detailed monitoring results to files"""
        # Save metrics history
        metrics_file = self.output_dir / "detailed_metrics.json"
        metrics_data = [asdict(m) for m in self.metrics_history]

        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # Save summary
        summary = self._generate_summary()
        summary_file = self.output_dir / "monitoring_summary.json"

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Generate visualization if matplotlib available
        try:
            self._generate_monitoring_plots()
        except ImportError:
            self.logger.info("Matplotlib not available, skipping plots")

    def _generate_monitoring_plots(self):
        """Generate monitoring visualization plots"""
        import matplotlib.pyplot as plt

        if not self.metrics_history:
            return

        metrics_list = list(self.metrics_history)
        timestamps = [m.timestamp - self.start_time for m in metrics_list]

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Raspberry Pi Zero BitGen Monitoring Results', fontsize=16)

        # Memory usage
        ram_usage = [m.ram_usage_mb for m in metrics_list]
        axes[0, 0].plot(timestamps, ram_usage, 'b-', label='RAM Usage')
        axes[0, 0].set_ylabel('Memory (MB)')
        axes[0, 0].set_title('Memory Usage Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # CPU usage
        cpu_usage = [m.cpu_usage_percent for m in metrics_list]
        axes[0, 1].plot(timestamps, cpu_usage, 'r-', label='CPU Usage')
        axes[0, 1].set_ylabel('CPU Usage (%)')
        axes[0, 1].set_title('CPU Usage Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Power consumption
        power_usage = [m.power_consumption_mw for m in metrics_list]
        axes[1, 0].plot(timestamps, power_usage, 'g-', label='Power')
        axes[1, 0].set_ylabel('Power (mW)')
        axes[1, 0].set_title('Power Consumption Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Temperature
        temperatures = [m.cpu_temperature_c for m in metrics_list]
        axes[1, 1].plot(timestamps, temperatures, 'orange', label='CPU Temp')
        axes[1, 1].set_ylabel('Temperature (°C)')
        axes[1, 1].set_title('CPU Temperature Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Tokens per second (when available)
        token_rates = [m.tokens_per_second for m in metrics_list if m.tokens_per_second > 0]
        if token_rates:
            token_times = [timestamps[i] for i, m in enumerate(metrics_list) if m.tokens_per_second > 0]
            axes[2, 0].plot(token_times, token_rates, 'purple', marker='o', label='Tokens/sec')
            axes[2, 0].set_ylabel('Tokens/Second')
            axes[2, 0].set_title('Inference Performance')
            axes[2, 0].legend()
            axes[2, 0].grid(True)

        # Latency per token
        latencies = [m.latency_ms_per_token for m in metrics_list if m.latency_ms_per_token > 0]
        if latencies:
            latency_times = [timestamps[i] for i, m in enumerate(metrics_list) if m.latency_ms_per_token > 0]
            axes[2, 1].plot(latency_times, latencies, 'brown', marker='s', label='Latency')
            axes[2, 1].set_ylabel('Latency (ms/token)')
            axes[2, 1].set_title('Response Latency')
            axes[2, 1].legend()
            axes[2, 1].grid(True)

        # Set x-axis labels
        for ax in axes.flat:
            ax.set_xlabel('Time (seconds)')

        plt.tight_layout()
        plt.savefig(self.output_dir / "monitoring_plots.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info("Monitoring plots saved")

    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        info = {
            'platform': {
                'system': os.uname().sysname,
                'machine': os.uname().machine,
                'processor': os.uname().machine,
            },
            'cpu': {
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
            },
            'memory': {
                'total_mb': psutil.virtual_memory().total / 1024 / 1024,
                'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            },
            'disk': {
                'total_gb': psutil.disk_usage('/').total / 1024 / 1024 / 1024,
                'free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024,
            }
        }

        # Try to get Raspberry Pi specific info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo:
                    info['platform']['device'] = 'Raspberry Pi'
                    # Extract model info
                    for line in cpuinfo.split('\n'):
                        if line.startswith('Model'):
                            info['platform']['model'] = line.split(':', 1)[1].strip()
                            break
        except:
            pass

        return info

# Global monitor instance for easy access
_global_monitor = None

def get_monitor() -> RaspberryPiMonitor:
    """Get global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RaspberryPiMonitor()
    return _global_monitor

def start_monitoring():
    """Start global monitoring"""
    monitor = get_monitor()
    monitor.start_monitoring()
    return monitor

def stop_monitoring():
    """Stop global monitoring and return results"""
    monitor = get_monitor()
    return monitor.stop_monitoring()

@contextmanager
def monitor_inference(input_text: str, model_name: str = "BitGen"):
    """Context manager for monitoring inference"""
    monitor = get_monitor()
    with monitor.measure_inference(input_text, model_name) as data:
        yield data
