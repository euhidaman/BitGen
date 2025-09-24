"""
Raspberry Pi Zero Optimized BitGen Configuration
Ultra-lightweight configuration with comprehensive monitoring capabilities
"""

from dataclasses import dataclass
from typing import Dict, Optional
import json

@dataclass
class RaspberryPiZeroConfig:
    """Ultra-optimized configuration for Raspberry Pi Zero"""
    # Model architecture (minimized for Pi Zero's 512MB RAM)
    embed_dim: int = 32
    num_layers: int = 2
    num_heads: int = 2
    head_dim: int = 16
    ffn_dim: int = 64
    vocab_size: int = 2048  # Reduced vocabulary

    # Episodic Memory (minimal)
    memory_size: int = 8
    memory_dim: int = 32
    direct_writing: bool = True

    # Attention sinks (minimal)
    attention_sinks: int = 2
    window_size: int = 32
    max_seq_len: int = 32  # Very short sequences

    # Cross-modal (ultra-minimal)
    vision_embed_dim: int = 32
    fusion_layers: int = 1

    # Reasoning (simplified)
    reasoning_dim: int = 16
    max_reasoning_steps: int = 3

    # Robot selection (reduced)
    num_robots: int = 4
    robot_embed_dim: int = 8

    # Training (optimized for Pi Zero)
    dropout: float = 0.05  # Lower dropout
    layer_norm_eps: float = 1e-5
    quantization_bits: float = 1.58

    # Pi Zero specific optimizations
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True
    use_int8_inference: bool = True
    batch_size_limit: int = 1
    memory_limit_mb: int = 128  # Conservative memory limit

    # Power management
    enable_thermal_throttling: bool = True
    max_temperature_c: float = 70.0
    power_budget_mw: float = 500.0

@dataclass
class MonitoringConfig:
    """Configuration for comprehensive monitoring"""
    # Monitoring intervals
    monitoring_interval_s: float = 1.0
    display_refresh_s: float = 2.0

    # Data retention
    max_inference_history: int = 100
    max_monitoring_samples: int = 10000

    # Thresholds for alerts
    temperature_warning_c: float = 65.0
    temperature_critical_c: float = 75.0
    memory_warning_mb: float = 400.0
    power_warning_mw: float = 600.0

    # Energy tracking
    enable_carbon_tracking: bool = True
    carbon_country_code: str = "US"

    # Performance targets
    target_tokens_per_second: float = 2.0
    target_latency_ms_per_token: float = 500.0
    target_power_budget_mw: float = 400.0

# Hardware-specific configurations for different Pi models
PI_ZERO_SPECS = {
    "cpu_cores": 1,
    "cpu_freq_mhz": 1000,
    "ram_mb": 512,
    "gpu": None,
    "thermal_limit_c": 85.0,
    "typical_idle_power_mw": 120,
    "max_safe_power_mw": 600
}

PI_ZERO_W_SPECS = {
    "cpu_cores": 1,
    "cpu_freq_mhz": 1000,
    "ram_mb": 512,
    "gpu": None,
    "thermal_limit_c": 85.0,
    "typical_idle_power_mw": 150,  # Higher due to WiFi
    "max_safe_power_mw": 650,
    "has_wifi": True
}

PI_ZERO_2W_SPECS = {
    "cpu_cores": 4,
    "cpu_freq_mhz": 1000,
    "ram_mb": 512,
    "gpu": None,
    "thermal_limit_c": 85.0,
    "typical_idle_power_mw": 180,
    "max_safe_power_mw": 800,
    "has_wifi": True
}

def get_pi_zero_config(model_variant: str = "pi_zero") -> Dict:
    """Get optimized configuration for specific Pi Zero variant"""

    base_config = RaspberryPiZeroConfig()
    monitoring_config = MonitoringConfig()

    # Adjust based on Pi model
    if model_variant == "pi_zero":
        hardware_specs = PI_ZERO_SPECS
        # Ultra-minimal config
        base_config.embed_dim = 24
        base_config.memory_size = 4

    elif model_variant == "pi_zero_w":
        hardware_specs = PI_ZERO_W_SPECS
        # Standard minimal config

    elif model_variant == "pi_zero_2w":
        hardware_specs = PI_ZERO_2W_SPECS
        # Slightly larger config for quad-core
        base_config.embed_dim = 48
        base_config.num_layers = 3
        base_config.memory_size = 16
        base_config.max_seq_len = 48

    else:
        raise ValueError(f"Unknown Pi model: {model_variant}")

    # Update monitoring thresholds based on hardware
    monitoring_config.temperature_critical_c = hardware_specs["thermal_limit_c"] - 10
    monitoring_config.memory_warning_mb = hardware_specs["ram_mb"] * 0.8
    monitoring_config.target_power_budget_mw = hardware_specs["max_safe_power_mw"] * 0.8

    return {
        "model_config": base_config,
        "monitoring_config": monitoring_config,
        "hardware_specs": hardware_specs
    }

def save_pi_zero_config(config_dict: Dict, filepath: str):
    """Save Pi Zero configuration to file"""
    # Convert dataclasses to dictionaries
    serializable_config = {}

    for key, value in config_dict.items():
        if hasattr(value, '__dict__'):
            serializable_config[key] = value.__dict__
        else:
            serializable_config[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2)

    print(f"Pi Zero configuration saved to {filepath}")

def load_pi_zero_config(filepath: str) -> Dict:
    """Load Pi Zero configuration from file"""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    # Reconstruct dataclass objects
    result = {}

    if "model_config" in config_dict:
        result["model_config"] = RaspberryPiZeroConfig(**config_dict["model_config"])

    if "monitoring_config" in config_dict:
        result["monitoring_config"] = MonitoringConfig(**config_dict["monitoring_config"])

    if "hardware_specs" in config_dict:
        result["hardware_specs"] = config_dict["hardware_specs"]

    return result
