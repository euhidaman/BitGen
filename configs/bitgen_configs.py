"""
BitGen Configuration Templates
Pre-configured setups for different embedded deployment scenarios
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import json

@dataclass
class EmbeddedConfig:
    """Base configuration for embedded deployment"""
    # Target hardware constraints
    target_memory_kb: int = 512
    target_flash_mb: int = 4
    target_cpu_mhz: int = 168
    target_board: str = "cortex-m4"

    # Performance requirements
    max_inference_time_ms: int = 500
    max_power_consumption_mw: int = 100
    min_battery_life_hours: int = 8

    # Deployment optimizations
    use_quantization: bool = True
    use_pruning: bool = True
    use_knowledge_distillation: bool = False

@dataclass
class BitGenNanoConfig:
    """Ultra-tiny configuration for smallest microcontrollers"""
    # Model architecture (minimal)
    embed_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    head_dim: int = 16
    ffn_dim: int = 128
    vocab_size: int = 2048

    # Memory constraints
    memory_size: int = 16
    memory_dim: int = 64

    # Attention sinks
    attention_sinks: int = 2
    window_size: int = 32
    max_seq_len: int = 64

    # Cross-modal (minimal)
    vision_embed_dim: int = 64
    fusion_layers: int = 1

    # Reasoning
    reasoning_dim: int = 32
    max_reasoning_steps: int = 3

    # Robot selection
    num_robots: int = 8
    robot_embed_dim: int = 16

    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    quantization_bits: float = 1.58

    # Embedded constraints
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True
    use_int8_inference: bool = True

@dataclass
class BitGenTinyConfig:
    """Default tiny configuration for typical microcontrollers"""
    # Model architecture
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 16
    ffn_dim: int = 256
    vocab_size: int = 8192

    # Episodic Memory
    memory_size: int = 64
    memory_dim: int = 128
    direct_writing: bool = True

    # Attention sinks
    attention_sinks: int = 4
    window_size: int = 128
    max_seq_len: int = 256

    # Cross-modal
    vision_embed_dim: int = 128
    fusion_layers: int = 2

    # Reasoning
    reasoning_dim: int = 64
    max_reasoning_steps: int = 8

    # Robot selection
    num_robots: int = 16
    robot_embed_dim: int = 32

    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    quantization_bits: float = 1.58

    # Embedded optimizations
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True
    use_int8_inference: bool = True

@dataclass
class BitGenSmallConfig:
    """Small configuration for more capable embedded systems"""
    # Model architecture
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 32
    ffn_dim: int = 512
    vocab_size: int = 16384

    # Episodic Memory
    memory_size: int = 128
    memory_dim: int = 256
    direct_writing: bool = True

    # Attention sinks
    attention_sinks: int = 8
    window_size: int = 256
    max_seq_len: int = 512

    # Cross-modal
    vision_embed_dim: int = 256
    fusion_layers: int = 3

    # Reasoning
    reasoning_dim: int = 128
    max_reasoning_steps: int = 12

    # Robot selection
    num_robots: int = 32
    robot_embed_dim: int = 64

    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    quantization_bits: float = 1.58

    # Embedded optimizations
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True
    use_int8_inference: bool = True

@dataclass
class TrainingConfig:
    """Training configuration for BitGen"""
    # Data
    coco_data_path: str = "data/coco/coco_train.json"
    robot_data_path: str = "data/robot_selection/robot_tasks.json"
    validation_split: float = 0.1

    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    num_epochs: int = 10
    warmup_steps: int = 1000

    # Memory optimization
    gradient_accumulation_steps: int = 4
    max_memory_mb: int = 1024
    use_mixed_precision: bool = True

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100

    # Early stopping
    patience: int = 5
    min_delta: float = 0.001

    # Distributed training
    use_ddp: bool = False
    world_size: int = 1

    # Optimization for embedded
    use_knowledge_distillation: bool = False
    teacher_model_path: Optional[str] = None
    distillation_alpha: float = 0.7
    distillation_temperature: float = 4.0

# Predefined configurations for different use cases
EMBEDDED_CONFIGS = {
    "cortex-m0": {
        "model": BitGenNanoConfig(),
        "hardware": EmbeddedConfig(
            target_memory_kb=256,
            target_flash_mb=1,
            target_cpu_mhz=48,
            target_board="cortex-m0"
        )
    },

    "cortex-m4": {
        "model": BitGenTinyConfig(),
        "hardware": EmbeddedConfig(
            target_memory_kb=512,
            target_flash_mb=4,
            target_cpu_mhz=168,
            target_board="cortex-m4"
        )
    },

    "cortex-m7": {
        "model": BitGenSmallConfig(),
        "hardware": EmbeddedConfig(
            target_memory_kb=2048,
            target_flash_mb=16,
            target_cpu_mhz=400,
            target_board="cortex-m7"
        )
    }
}

def get_config_for_target(target: str) -> Dict:
    """Get configuration for specific embedded target"""
    if target in EMBEDDED_CONFIGS:
        return EMBEDDED_CONFIGS[target]
    else:
        print(f"Unknown target '{target}', using default cortex-m4 config")
        return EMBEDDED_CONFIGS["cortex-m4"]

def save_config(config, filepath: str):
    """Save configuration to JSON file"""
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Configuration saved to {filepath}")

def load_config(filepath: str) -> Dict:
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    return config_dict

# Task-specific configurations
TASK_CONFIGS = {
    "image_captioning": {
        "vision_weight": 1.5,
        "text_weight": 1.0,
        "fusion_layers": 3,
        "max_caption_length": 50
    },

    "robot_navigation": {
        "reasoning_weight": 2.0,
        "memory_weight": 1.5,
        "robot_selection_weight": 2.0,
        "max_reasoning_steps": 15
    },

    "multimodal_qa": {
        "vision_weight": 1.2,
        "text_weight": 1.0,
        "reasoning_weight": 1.8,
        "fusion_layers": 2
    },

    "general_purpose": {
        "vision_weight": 1.0,
        "text_weight": 1.0,
        "reasoning_weight": 1.0,
        "memory_weight": 1.0,
        "robot_selection_weight": 1.0
    }
}

def create_task_specific_config(base_config, task: str):
    """Create task-specific configuration"""
    if task not in TASK_CONFIGS:
        print(f"Unknown task '{task}', using general_purpose config")
        task = "general_purpose"

    task_config = TASK_CONFIGS[task].copy()

    # Apply task-specific modifications to base config
    modified_config = base_config.__dict__.copy()

    if "max_reasoning_steps" in task_config:
        modified_config["max_reasoning_steps"] = task_config["max_reasoning_steps"]

    if "fusion_layers" in task_config:
        modified_config["fusion_layers"] = task_config["fusion_layers"]

    return modified_config, task_config
