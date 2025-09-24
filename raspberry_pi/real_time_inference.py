"""
Real-time BitGen Inference for Raspberry Pi Zero
Comprehensive monitoring of throughput, latency, energy, memory, power, and thermal profile
"""

import torch
import torch.nn as nn
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque, defaultdict
from contextlib import contextmanager
import threading
import psutil
import os

# Import BitGen components
from bitgen_model import BitGenModel, BitGenConfig, create_bitgen_model
from data_loader import BitGenTokenizer
from raspberry_pi.rpi_monitor import RaspberryPiMonitor, monitor_inference

# Real-time display imports
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

class RealTimeInferenceMonitor:
    """Real-time inference monitoring with live display for Raspberry Pi Zero"""

    def __init__(self, model: BitGenModel, config: BitGenConfig, output_dir: str = "inference_monitoring"):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = BitGenTokenizer(config.vocab_size)

        # Initialize monitoring
        self.rpi_monitor = RaspberryPiMonitor(str(self.output_dir))
        self.rpi_monitor.start_monitoring()

        # Performance tracking
        self.inference_history = deque(maxlen=100)  # Keep last 100 inferences
        self.real_time_metrics = {}
        self.session_start_time = time.time()

        # Baseline measurements
        self.baseline_power_mw = self._measure_baseline_power()
        self.baseline_memory_mb = psutil.virtual_memory().used / 1024 / 1024

        # Threading for continuous monitoring
        self.monitoring_active = True
        self.display_thread = None

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for inference monitoring"""
        log_file = self.output_dir / "inference_monitoring.log"
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
        try:
            # Get CPU frequency and temperature for power estimation
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_c = int(f.read().strip()) / 1000.0

            # Basic power model for Pi Zero
            base_idle = 120.0  # Base idle power in mW
            temp_factor = max(0, (temp_c - 35.0) * 2.0)  # Temperature contribution

            return base_idle + temp_factor
        except:
            return 150.0  # Default baseline

    def measure_single_inference(self, input_text: str, max_length: int = 50) -> Dict:
        """Measure comprehensive metrics for a single inference"""

        # Pre-inference state
        pre_memory = psutil.virtual_memory()
        pre_cpu_times = psutil.cpu_times()
        pre_temp = self._get_cpu_temperature()
        pre_time = time.time()

        # Tokenize input
        input_ids = torch.tensor([self.tokenizer.encode(input_text)])
        input_tokens = len(self.tokenizer.tokenize(input_text))

        # Inference with detailed timing
        inference_metrics = {}

        with monitor_inference(input_text, "BitGen-PiZero") as inference_data:
            # Forward pass timing
            forward_start = time.time()

            with torch.no_grad():
                # Generate response
                generated_ids, attention_cache = self.model.generate_embedded(
                    input_ids,
                    max_length=max_length,
                    temperature=0.7,
                    cache=None
                )

            forward_time = time.time() - forward_start

            # Decode output
            decode_start = time.time()
            output_text = self.tokenizer.decode(generated_ids[0].tolist())
            decode_time = time.time() - decode_start

            # Calculate tokens
            output_tokens = len(self.tokenizer.tokenize(output_text))
            total_tokens = input_tokens + output_tokens

            # Update inference data for monitoring
            inference_data['output_tokens'] = output_tokens
            inference_data['output_text'] = output_text

        # Post-inference state
        post_memory = psutil.virtual_memory()
        post_cpu_times = psutil.cpu_times()
        post_temp = self._get_cpu_temperature()
        post_time = time.time()

        # Calculate comprehensive metrics
        total_inference_time = post_time - pre_time

        # Throughput and Latency
        tokens_per_second = total_tokens / total_inference_time if total_inference_time > 0 else 0
        ms_per_token = (total_inference_time * 1000) / total_tokens if total_tokens > 0 else 0
        response_latency_ms = total_inference_time * 1000

        # Memory Footprint
        memory_delta_mb = (post_memory.used - pre_memory.used) / 1024 / 1024
        memory_peak_mb = post_memory.used / 1024 / 1024

        # CPU Usage
        cpu_user_delta = post_cpu_times.user - pre_cpu_times.user
        cpu_system_delta = post_cpu_times.system - pre_cpu_times.system
        cpu_usage_percent = ((cpu_user_delta + cpu_system_delta) / total_inference_time) * 100

        # Power and Energy
        avg_temp = (pre_temp + post_temp) / 2
        estimated_power_mw = self._estimate_inference_power(cpu_usage_percent, avg_temp, total_inference_time)
        energy_consumed_mj = (estimated_power_mw * total_inference_time * 1000) / 1000  # mJ

        # Thermal Profile
        thermal_delta = post_temp - pre_temp

        # Compile comprehensive metrics
        metrics = {
            'timestamp': post_time,
            'session_time_s': post_time - self.session_start_time,

            # Input/Output
            'input_text': input_text,
            'output_text': output_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,

            # Performance Metrics
            'total_inference_time_ms': total_inference_time * 1000,
            'forward_time_ms': forward_time * 1000,
            'decode_time_ms': decode_time * 1000,
            'tokens_per_second': tokens_per_second,
            'latency_ms_per_token': ms_per_token,
            'response_latency_ms': response_latency_ms,

            # Memory Footprint
            'memory_delta_mb': memory_delta_mb,
            'memory_peak_mb': memory_peak_mb,
            'memory_baseline_mb': self.baseline_memory_mb,
            'swap_usage_mb': post_memory.swap / 1024 / 1024,

            # CPU and Processing
            'cpu_usage_percent': cpu_usage_percent,
            'cpu_user_time_delta': cpu_user_delta,
            'cpu_system_time_delta': cpu_system_delta,

            # Power and Energy
            'estimated_power_mw': estimated_power_mw,
            'baseline_power_mw': self.baseline_power_mw,
            'power_overhead_mw': estimated_power_mw - self.baseline_power_mw,
            'energy_consumed_mj': energy_consumed_mj,
            'energy_per_token_mj': energy_consumed_mj / total_tokens if total_tokens > 0 else 0,

            # Thermal Profile
            'cpu_temp_pre_c': pre_temp,
            'cpu_temp_post_c': post_temp,
            'thermal_delta_c': thermal_delta,
            'thermal_load_factor': self._calculate_thermal_load_factor(pre_temp, post_temp, total_inference_time),

            # Efficiency Metrics
            'tokens_per_mj': total_tokens / energy_consumed_mj if energy_consumed_mj > 0 else 0,
            'tokens_per_mb': total_tokens / max(memory_delta_mb, 0.1),
            'performance_score': self._calculate_performance_score(tokens_per_second, energy_consumed_mj, memory_delta_mb)
        }

        # Store in history
        self.inference_history.append(metrics)

        # Update real-time metrics
        self._update_real_time_metrics(metrics)

        # Save individual result
        self._save_inference_result(metrics)

        return metrics

    def _get_cpu_temperature(self) -> float:
        """Get current CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return int(f.read().strip()) / 1000.0
        except:
            return 50.0  # Default temperature

    def _estimate_inference_power(self, cpu_usage: float, temperature: float, duration: float) -> float:
        """Estimate power consumption during inference"""
        # Base power
        base_power = self.baseline_power_mw

        # CPU load contribution (Pi Zero specific)
        cpu_power = (cpu_usage / 100.0) * 250.0  # Up to 250mW for full CPU load

        # Temperature contribution
        temp_power = max(0, (temperature - 40.0) * 3.0)  # 3mW per degree above 40C

        # Duration factor (longer operations may be more efficient)
        duration_factor = 1.0 + max(0, (duration - 1.0) * 0.1)  # Slight increase for longer operations

        total_power = (base_power + cpu_power + temp_power) * duration_factor
        return min(total_power, 600.0)  # Cap at reasonable maximum for Pi Zero

    def _calculate_thermal_load_factor(self, pre_temp: float, post_temp: float, duration: float) -> float:
        """Calculate thermal load factor"""
        thermal_delta = post_temp - pre_temp
        if duration > 0:
            thermal_rate = thermal_delta / duration  # Temperature rise per second
            # Normalize to 0-1 scale (0.5°C/s = factor of 1.0)
            return min(thermal_rate / 0.5, 2.0)
        return 0.0

    def _calculate_performance_score(self, tokens_per_second: float, energy_mj: float, memory_mb: float) -> float:
        """Calculate overall performance score (higher is better)"""
        # Normalize metrics to 0-1 scale
        throughput_score = min(tokens_per_second / 10.0, 1.0)  # 10 tokens/s = perfect
        energy_score = max(0, 1.0 - (energy_mj / 1000.0))  # Penalize high energy
        memory_score = max(0, 1.0 - (abs(memory_mb) / 100.0))  # Penalize high memory usage

        # Weighted combination
        return (throughput_score * 0.5 + energy_score * 0.3 + memory_score * 0.2)

    def _update_real_time_metrics(self, metrics: Dict):
        """Update real-time aggregated metrics"""
        if len(self.inference_history) == 0:
            return

        recent_metrics = list(self.inference_history)[-20:]  # Last 20 inferences

        self.real_time_metrics = {
            'current': metrics,
            'session_stats': {
                'total_inferences': len(self.inference_history),
                'session_duration_s': time.time() - self.session_start_time,
                'total_tokens_generated': sum(m['total_tokens'] for m in self.inference_history),
                'total_energy_consumed_mj': sum(m['energy_consumed_mj'] for m in self.inference_history),
            },
            'recent_averages': {
                'avg_tokens_per_second': np.mean([m['tokens_per_second'] for m in recent_metrics]),
                'avg_latency_ms_per_token': np.mean([m['latency_ms_per_token'] for m in recent_metrics]),
                'avg_response_latency_ms': np.mean([m['response_latency_ms'] for m in recent_metrics]),
                'avg_memory_usage_mb': np.mean([m['memory_peak_mb'] for m in recent_metrics]),
                'avg_power_mw': np.mean([m['estimated_power_mw'] for m in recent_metrics]),
                'avg_cpu_temp_c': np.mean([m['cpu_temp_post_c'] for m in recent_metrics]),
                'avg_performance_score': np.mean([m['performance_score'] for m in recent_metrics]),
            },
            'peaks': {
                'peak_tokens_per_second': max(m['tokens_per_second'] for m in self.inference_history),
                'peak_latency_ms': max(m['response_latency_ms'] for m in self.inference_history),
                'peak_memory_mb': max(m['memory_peak_mb'] for m in self.inference_history),
                'peak_power_mw': max(m['estimated_power_mw'] for m in self.inference_history),
                'peak_temperature_c': max(m['cpu_temp_post_c'] for m in self.inference_history),
            }
        }

    def _save_inference_result(self, metrics: Dict):
        """Save individual inference result"""
        results_file = self.output_dir / "inference_results.jsonl"

        with open(results_file, 'a') as f:
            json.dump(metrics, f)
            f.write('\n')

    def start_real_time_display(self):
        """Start real-time display of metrics"""
        if CURSES_AVAILABLE:
            self.display_thread = threading.Thread(target=self._curses_display, daemon=True)
            self.display_thread.start()
        else:
            self.display_thread = threading.Thread(target=self._simple_display, daemon=True)
            self.display_thread.start()

    def _curses_display(self):
        """Real-time display using curses"""
        curses.wrapper(self._curses_main)

    def _curses_main(self, stdscr):
        """Main curses display loop"""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(1000)  # Refresh every second

        while self.monitoring_active:
            stdscr.clear()

            # Display header
            stdscr.addstr(0, 0, "BitGen Real-Time Inference Monitor - Raspberry Pi Zero", curses.A_BOLD)
            stdscr.addstr(1, 0, "=" * 80)

            if self.real_time_metrics:
                self._draw_metrics_display(stdscr)
            else:
                stdscr.addstr(3, 0, "Waiting for inference data...")

            stdscr.addstr(25, 0, "Press 'q' to quit, any other key to refresh")
            stdscr.refresh()

            # Check for quit
            key = stdscr.getch()
            if key == ord('q'):
                break

    def _draw_metrics_display(self, stdscr):
        """Draw the metrics display"""
        metrics = self.real_time_metrics
        current = metrics['current']
        session = metrics['session_stats']
        recent = metrics['recent_averages']
        peaks = metrics['peaks']

        row = 3

        # Current inference
        stdscr.addstr(row, 0, "CURRENT INFERENCE:", curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, f"Input: {current['input_text'][:50]}...")
        row += 1
        stdscr.addstr(row, 2, f"Output: {current['output_text'][:50]}...")
        row += 2

        # Performance metrics
        stdscr.addstr(row, 0, "PERFORMANCE METRICS:", curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, f"Throughput: {current['tokens_per_second']:6.2f} tokens/sec (avg: {recent['avg_tokens_per_second']:6.2f})")
        row += 1
        stdscr.addstr(row, 2, f"Latency: {current['latency_ms_per_token']:8.2f} ms/token (avg: {recent['avg_latency_ms_per_token']:8.2f})")
        row += 1
        stdscr.addstr(row, 2, f"Response: {current['response_latency_ms']:7.2f} ms (avg: {recent['avg_response_latency_ms']:7.2f})")
        row += 2

        # Memory footprint
        stdscr.addstr(row, 0, "MEMORY FOOTPRINT:", curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, f"Current RAM: {current['memory_peak_mb']:7.2f} MB (avg: {recent['avg_memory_usage_mb']:7.2f})")
        row += 1
        stdscr.addstr(row, 2, f"Memory Delta: {current['memory_delta_mb']:6.2f} MB (peak: {peaks['peak_memory_mb']:7.2f})")
        row += 1
        stdscr.addstr(row, 2, f"Swap Usage: {current['swap_usage_mb']:8.2f} MB")
        row += 2

        # Power and energy
        stdscr.addstr(row, 0, "POWER & ENERGY:", curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, f"Power: {current['estimated_power_mw']:8.1f} mW (avg: {recent['avg_power_mw']:8.1f})")
        row += 1
        stdscr.addstr(row, 2, f"Energy: {current['energy_consumed_mj']:7.2f} mJ (total: {session['total_energy_consumed_mj']:8.2f})")
        row += 1
        stdscr.addstr(row, 2, f"Energy/Token: {current['energy_per_token_mj']:5.3f} mJ/token")
        row += 2

        # Thermal profile
        stdscr.addstr(row, 0, "THERMAL PROFILE:", curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, f"CPU Temp: {current['cpu_temp_post_c']:6.1f}°C (avg: {recent['avg_cpu_temp_c']:6.1f}°C)")
        row += 1
        stdscr.addstr(row, 2, f"Temp Delta: {current['thermal_delta_c']:5.2f}°C (peak: {peaks['peak_temperature_c']:6.1f}°C)")
        row += 1
        stdscr.addstr(row, 2, f"Thermal Load: {current['thermal_load_factor']:5.3f} (higher = more thermal stress)")
        row += 2

        # Session summary
        stdscr.addstr(row, 0, "SESSION SUMMARY:", curses.A_BOLD)
        row += 1
        stdscr.addstr(row, 2, f"Total Inferences: {session['total_inferences']}")
        row += 1
        stdscr.addstr(row, 2, f"Session Duration: {session['session_duration_s']:7.1f} seconds")
        row += 1
        stdscr.addstr(row, 2, f"Total Tokens: {session['total_tokens_generated']}")
        row += 1
        stdscr.addstr(row, 2, f"Performance Score: {current['performance_score']:5.3f}/1.000")

    def _simple_display(self):
        """Simple console display without curses"""
        while self.monitoring_active:
            if self.real_time_metrics:
                os.system('clear' if os.name == 'posix' else 'cls')
                self._print_metrics_simple()
            time.sleep(2)  # Update every 2 seconds

    def _print_metrics_simple(self):
        """Print metrics in simple format"""
        metrics = self.real_time_metrics
        current = metrics['current']
        recent = metrics['recent_averages']
        session = metrics['session_stats']

        print("=" * 80)
        print("BitGen Real-Time Inference Monitor - Raspberry Pi Zero")
        print("=" * 80)

        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Throughput:     {current['tokens_per_second']:8.2f} tokens/sec (avg: {recent['avg_tokens_per_second']:6.2f})")
        print(f"  Latency/Token:  {current['latency_ms_per_token']:8.2f} ms/token (avg: {recent['avg_latency_ms_per_token']:6.2f})")
        print(f"  Response Time:  {current['response_latency_ms']:8.2f} ms (avg: {recent['avg_response_latency_ms']:6.2f})")

        print(f"\n💾 MEMORY FOOTPRINT:")
        print(f"  RAM Usage:      {current['memory_peak_mb']:8.2f} MB (avg: {recent['avg_memory_usage_mb']:6.2f})")
        print(f"  Memory Delta:   {current['memory_delta_mb']:8.2f} MB")
        print(f"  Swap Usage:     {current['swap_usage_mb']:8.2f} MB")

        print(f"\n⚡ POWER & ENERGY:")
        print(f"  Power:          {current['estimated_power_mw']:8.1f} mW (avg: {recent['avg_power_mw']:6.1f})")
        print(f"  Energy:         {current['energy_consumed_mj']:8.2f} mJ")
        print(f"  Energy/Token:   {current['energy_per_token_mj']:8.3f} mJ/token")

        print(f"\n🌡️  THERMAL PROFILE:")
        print(f"  CPU Temp:       {current['cpu_temp_post_c']:8.1f}°C (avg: {recent['avg_cpu_temp_c']:6.1f}°C)")
        print(f"  Temp Delta:     {current['thermal_delta_c']:8.2f}°C")
        print(f"  Thermal Load:   {current['thermal_load_factor']:8.3f}")

        print(f"\n📈 SESSION STATS:")
        print(f"  Inferences:     {session['total_inferences']:8}")
        print(f"  Duration:       {session['session_duration_s']:8.1f} seconds")
        print(f"  Total Tokens:   {session['total_tokens_generated']:8}")
        print(f"  Performance:    {current['performance_score']:8.3f}/1.000")

        print(f"\nLast inference: {current['input_text'][:40]}...")
        print(f"Response: {current['output_text'][:40]}...")

    def stop_monitoring(self):
        """Stop monitoring and generate final report"""
        self.monitoring_active = False

        if self.display_thread:
            self.display_thread.join(timeout=2.0)

        # Stop RPI monitor
        rpi_summary = self.rpi_monitor.stop_monitoring()

        # Generate final report
        final_report = self._generate_final_report(rpi_summary)

        return final_report

    def _generate_final_report(self, rpi_summary: Dict) -> Dict:
        """Generate comprehensive final report"""
        if not self.inference_history:
            return {}

        # Calculate comprehensive statistics
        all_metrics = list(self.inference_history)

        report = {
            'session_summary': {
                'total_inferences': len(all_metrics),
                'session_duration_s': time.time() - self.session_start_time,
                'total_tokens_processed': sum(m['total_tokens'] for m in all_metrics),
                'total_energy_consumed_mj': sum(m['energy_consumed_mj'] for m in all_metrics),
            },
            'performance_statistics': {
                'throughput': {
                    'avg_tokens_per_second': np.mean([m['tokens_per_second'] for m in all_metrics]),
                    'median_tokens_per_second': np.median([m['tokens_per_second'] for m in all_metrics]),
                    'peak_tokens_per_second': np.max([m['tokens_per_second'] for m in all_metrics]),
                    'min_tokens_per_second': np.min([m['tokens_per_second'] for m in all_metrics]),
                },
                'latency': {
                    'avg_latency_ms_per_token': np.mean([m['latency_ms_per_token'] for m in all_metrics]),
                    'median_latency_ms_per_token': np.median([m['latency_ms_per_token'] for m in all_metrics]),
                    'p95_latency_ms_per_token': np.percentile([m['latency_ms_per_token'] for m in all_metrics], 95),
                    'max_response_latency_ms': np.max([m['response_latency_ms'] for m in all_metrics]),
                },
            },
            'resource_utilization': {
                'memory': {
                    'avg_memory_usage_mb': np.mean([m['memory_peak_mb'] for m in all_metrics]),
                    'peak_memory_usage_mb': np.max([m['memory_peak_mb'] for m in all_metrics]),
                    'avg_memory_delta_mb': np.mean([abs(m['memory_delta_mb']) for m in all_metrics]),
                },
                'power': {
                    'avg_power_consumption_mw': np.mean([m['estimated_power_mw'] for m in all_metrics]),
                    'peak_power_consumption_mw': np.max([m['estimated_power_mw'] for m in all_metrics]),
                    'total_energy_consumed_mj': sum(m['energy_consumed_mj'] for m in all_metrics),
                    'avg_energy_per_token_mj': np.mean([m['energy_per_token_mj'] for m in all_metrics]),
                },
                'thermal': {
                    'avg_cpu_temperature_c': np.mean([m['cpu_temp_post_c'] for m in all_metrics]),
                    'peak_cpu_temperature_c': np.max([m['cpu_temp_post_c'] for m in all_metrics]),
                    'avg_thermal_delta_c': np.mean([abs(m['thermal_delta_c']) for m in all_metrics]),
                    'thermal_throttling_risk': 'High' if np.max([m['cpu_temp_post_c'] for m in all_metrics]) > 70 else 'Low',
                }
            },
            'efficiency_metrics': {
                'tokens_per_mj': np.mean([m['tokens_per_mj'] for m in all_metrics if m['tokens_per_mj'] > 0]),
                'tokens_per_mb': np.mean([m['tokens_per_mb'] for m in all_metrics]),
                'avg_performance_score': np.mean([m['performance_score'] for m in all_metrics]),
            },
            'raspberry_pi_specific': rpi_summary
        }

        # Save comprehensive report
        report_path = self.output_dir / "comprehensive_inference_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Comprehensive inference report saved to {report_path}")

        return report

def interactive_inference_session(model_path: str):
    """Interactive inference session with real-time monitoring"""

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = BitGenConfig(**checkpoint['config'])

    model = create_bitgen_model('nano')  # Adjust based on saved model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize monitoring
    monitor = RealTimeInferenceMonitor(model, config)
    monitor.start_real_time_display()

    print("🚀 BitGen Interactive Inference - Raspberry Pi Zero")
    print("Real-time monitoring active. Type 'quit' to exit, 'stats' for session stats.")
    print("=" * 80)

    try:
        while True:
            user_input = input("\nEnter your prompt: ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                if monitor.real_time_metrics:
                    print("\n📊 Session Statistics:")
                    session = monitor.real_time_metrics['session_stats']
                    recent = monitor.real_time_metrics['recent_averages']
                    print(f"  Total inferences: {session['total_inferences']}")
                    print(f"  Avg throughput: {recent['avg_tokens_per_second']:.2f} tokens/sec")
                    print(f"  Avg power: {recent['avg_power_mw']:.1f} mW")
                    print(f"  Avg temperature: {recent['avg_cpu_temp_c']:.1f}°C")
                continue
            elif not user_input:
                continue

            # Run inference with monitoring
            print("Processing... (monitoring active)")
            metrics = monitor.measure_single_inference(user_input)

            # Display results
            print(f"\n✅ Response: {metrics['output_text']}")
            print(f"📊 Metrics:")
            print(f"   ⚡ {metrics['tokens_per_second']:.2f} tokens/sec")
            print(f"   ⏱️  {metrics['response_latency_ms']:.2f} ms total")
            print(f"   💾 {metrics['memory_peak_mb']:.2f} MB RAM")
            print(f"   🔋 {metrics['estimated_power_mw']:.1f} mW")
            print(f"   🌡️  {metrics['cpu_temp_post_c']:.1f}°C")
            print(f"   📈 {metrics['performance_score']:.3f} score")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping monitoring...")
        final_report = monitor.stop_monitoring()
        print("✅ Final report saved. Session complete.")

def main():
    """Main function for inference monitoring"""
    import argparse

    parser = argparse.ArgumentParser(description="BitGen Real-time Inference Monitor for Raspberry Pi Zero")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")
    parser.add_argument("--output_dir", type=str, default="inference_monitoring", help="Output directory")

    args = parser.parse_args()

    if args.interactive:
        interactive_inference_session(args.model_path)
    else:
        print("Use --interactive flag to start interactive monitoring session")

if __name__ == "__main__":
    main()
