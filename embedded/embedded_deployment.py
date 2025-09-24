"""
Embedded Deployment Utilities for BitGen
Microcontroller-specific optimizations and deployment tools
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import struct
import json
from pathlib import Path
import os
import subprocess

class EmbeddedModelOptimizer:
    """Optimize BitGen models for microcontroller deployment"""

    def __init__(self, target_memory_kb: int = 512, target_flash_mb: int = 4):
        self.target_memory_kb = target_memory_kb
        self.target_flash_mb = target_flash_mb

    def quantize_model_for_embedded(self, model: nn.Module, config) -> Dict:
        """Aggressive quantization for microcontroller deployment"""
        model.eval()

        quantized_weights = {}
        compression_stats = {
            'original_size_mb': 0,
            'quantized_size_mb': 0,
            'compression_ratio': 0
        }

        # Quantize each layer
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                original_weight = module.weight.data
                original_size = original_weight.numel() * 4  # float32 bytes
                compression_stats['original_size_mb'] += original_size / (1024 * 1024)

                if hasattr(module, 'quantize_weights'):
                    # Use BitNet quantization
                    quantized, scale = module.quantize_weights(original_weight)
                    quantized_weights[f"{name}.weight"] = {
                        'data': quantized.cpu().numpy().astype(np.int8),
                        'scale': scale.item(),
                        'shape': original_weight.shape
                    }
                    # 1.58 bits per weight + scale
                    new_size = original_weight.numel() * 1.58 / 8 + 4
                else:
                    # Standard int8 quantization
                    scale = original_weight.abs().max() / 127.0
                    quantized = torch.clamp(torch.round(original_weight / scale), -128, 127)
                    quantized_weights[f"{name}.weight"] = {
                        'data': quantized.cpu().numpy().astype(np.int8),
                        'scale': scale.item(),
                        'shape': original_weight.shape
                    }
                    new_size = original_weight.numel() + 4  # int8 + scale

                compression_stats['quantized_size_mb'] += new_size / (1024 * 1024)

                # Quantize bias if present
                if hasattr(module, 'bias') and module.bias is not None:
                    bias_scale = module.bias.abs().max() / 127.0
                    quantized_bias = torch.clamp(torch.round(module.bias / bias_scale), -128, 127)
                    quantized_weights[f"{name}.bias"] = {
                        'data': quantized_bias.cpu().numpy().astype(np.int8),
                        'scale': bias_scale.item(),
                        'shape': module.bias.shape
                    }

        compression_stats['compression_ratio'] = (
            compression_stats['original_size_mb'] / compression_stats['quantized_size_mb']
            if compression_stats['quantized_size_mb'] > 0 else 1.0
        )

        return {
            'quantized_weights': quantized_weights,
            'compression_stats': compression_stats,
            'config': config.__dict__
        }

    def generate_c_inference_code(self, quantized_model: Dict, output_dir: str):
        """Generate C code for microcontroller inference"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate header file
        header_content = self._generate_c_header(quantized_model)
        with open(output_path / "bitgen_embedded.h", "w") as f:
            f.write(header_content)

        # Generate implementation file
        impl_content = self._generate_c_implementation(quantized_model)
        with open(output_path / "bitgen_embedded.c", "w") as f:
            f.write(impl_content)

        # Generate weight data file
        weights_content = self._generate_c_weights(quantized_model)
        with open(output_path / "bitgen_weights.c", "w") as f:
            f.write(weights_content)

        # Generate Makefile
        makefile_content = self._generate_makefile()
        with open(output_path / "Makefile", "w") as f:
            f.write(makefile_content)

        print(f"Generated C code for embedded deployment in {output_path}")

    def _generate_c_header(self, quantized_model: Dict) -> str:
        """Generate C header file"""
        config = quantized_model['config']

        return f"""
#ifndef BITGEN_EMBEDDED_H
#define BITGEN_EMBEDDED_H

#include <stdint.h>
#include <stddef.h>

// Model configuration
#define EMBED_DIM {config['embed_dim']}
#define NUM_LAYERS {config['num_layers']}
#define NUM_HEADS {config['num_heads']}
#define VOCAB_SIZE {config['vocab_size']}
#define MAX_SEQ_LEN {config['max_seq_len']}
#define MEMORY_SIZE {config['memory_size']}
#define ATTENTION_SINKS {config['attention_sinks']}
#define WINDOW_SIZE {config['window_size']}

// Data types for embedded
typedef int8_t quantized_t;
typedef float scale_t;
typedef int16_t intermediate_t;

// Model structures
typedef struct {{
    quantized_t* data;
    scale_t scale;
    size_t size;
}} QuantizedWeight;

typedef struct {{
    QuantizedWeight embedding_weights;
    QuantizedWeight position_weights;
    QuantizedWeight attention_weights[NUM_LAYERS * 4]; // Q, K, V, O per layer
    QuantizedWeight ffn_weights[NUM_LAYERS * 2];       // Up and down per layer
    QuantizedWeight output_weights;
    QuantizedWeight memory_keys;
    QuantizedWeight memory_values;
}} BitGenWeights;

typedef struct {{
    intermediate_t hidden_states[MAX_SEQ_LEN * EMBED_DIM];
    intermediate_t attention_cache[NUM_LAYERS][ATTENTION_SINKS * EMBED_DIM];
    intermediate_t memory_buffer[MEMORY_SIZE * EMBED_DIM];
    uint16_t attention_cache_pos[NUM_LAYERS];
    uint8_t sequence_length;
}} BitGenState;

// Function declarations
int bitgen_init(BitGenState* state);
int bitgen_forward(BitGenState* state, const uint16_t* input_tokens, uint16_t* output_token);
int bitgen_generate(BitGenState* state, const uint16_t* prompt, uint16_t prompt_len, 
                   uint16_t* output, uint16_t max_length);
void bitgen_reset_cache(BitGenState* state);

// Utility functions
void quantized_matmul(const quantized_t* a, const quantized_t* b, intermediate_t* c,
                     size_t m, size_t k, size_t n, scale_t scale_a, scale_t scale_b);
void softmax_int16(intermediate_t* input, size_t size);
void layer_norm_int16(intermediate_t* input, size_t size);

#endif // BITGEN_EMBEDDED_H
"""

    def _generate_c_implementation(self, quantized_model: Dict) -> str:
        """Generate C implementation file"""
        return f"""
#include "bitgen_embedded.h"
#include <string.h>
#include <math.h>

// External weight data
extern const BitGenWeights model_weights;

int bitgen_init(BitGenState* state) {{
    // Initialize state
    memset(state, 0, sizeof(BitGenState));
    return 0;
}}

void quantized_matmul(const quantized_t* a, const quantized_t* b, intermediate_t* c,
                     size_t m, size_t k, size_t n, scale_t scale_a, scale_t scale_b) {{
    for (size_t i = 0; i < m; i++) {{
        for (size_t j = 0; j < n; j++) {{
            int32_t sum = 0;
            for (size_t l = 0; l < k; l++) {{
                sum += (int32_t)a[i * k + l] * (int32_t)b[l * n + j];
            }}
            // Scale and clamp to int16 range
            int32_t scaled = (int32_t)(sum * scale_a * scale_b);
            c[i * n + j] = (intermediate_t)((scaled > 32767) ? 32767 : 
                                          (scaled < -32768) ? -32768 : scaled);
        }}
    }}
}}

void softmax_int16(intermediate_t* input, size_t size) {{
    // Find maximum for numerical stability
    intermediate_t max_val = input[0];
    for (size_t i = 1; i < size; i++) {{
        if (input[i] > max_val) max_val = input[i];
    }}
    
    // Compute exponentials using lookup table (simplified)
    int32_t sum = 0;
    for (size_t i = 0; i < size; i++) {{
        input[i] -= max_val;
        // Simplified exponential approximation for embedded
        if (input[i] < -10) input[i] = 0;
        else input[i] = (input[i] * 256) / 10 + 256; // Linear approximation
        sum += input[i];
    }}
    
    // Normalize
    for (size_t i = 0; i < size; i++) {{
        input[i] = (input[i] * 32767) / sum;
    }}
}}

void layer_norm_int16(intermediate_t* input, size_t size) {{
    // Compute mean
    int32_t sum = 0;
    for (size_t i = 0; i < size; i++) {{
        sum += input[i];
    }}
    intermediate_t mean = sum / size;
    
    // Compute variance (simplified)
    int32_t var_sum = 0;
    for (size_t i = 0; i < size; i++) {{
        int32_t diff = input[i] - mean;
        var_sum += diff * diff;
    }}
    intermediate_t std = (intermediate_t)sqrt(var_sum / size);
    
    // Normalize
    for (size_t i = 0; i < size; i++) {{
        if (std > 1) {{
            input[i] = (input[i] - mean) * 256 / std;
        }}
    }}
}}

int bitgen_forward(BitGenState* state, const uint16_t* input_tokens, uint16_t* output_token) {{
    // Token embedding
    for (size_t i = 0; i < state->sequence_length && i < MAX_SEQ_LEN; i++) {{
        uint16_t token = input_tokens[i];
        if (token >= VOCAB_SIZE) token = 1; // UNK token
        
        // Simple embedding lookup (quantized)
        for (size_t j = 0; j < EMBED_DIM; j++) {{
            size_t idx = token * EMBED_DIM + j;
            state->hidden_states[i * EMBED_DIM + j] = 
                (intermediate_t)(model_weights.embedding_weights.data[idx] * 
                               model_weights.embedding_weights.scale * 256);
        }}
    }}
    
    // Multi-layer processing with attention sinks
    for (size_t layer = 0; layer < NUM_LAYERS; layer++) {{
        // Self-attention with sinks (simplified)
        intermediate_t* hidden = &state->hidden_states[0];
        
        // Update attention cache
        memcpy(&state->attention_cache[layer][state->attention_cache_pos[layer] * EMBED_DIM],
               &hidden[(state->sequence_length - 1) * EMBED_DIM],
               EMBED_DIM * sizeof(intermediate_t));
        
        state->attention_cache_pos[layer] = 
            (state->attention_cache_pos[layer] + 1) % ATTENTION_SINKS;
        
        // Layer normalization
        layer_norm_int16(hidden, state->sequence_length * EMBED_DIM);
        
        // FFN (simplified)
        // ... (implementation details for embedded constraints)
    }}
    
    // Output projection
    intermediate_t* final_hidden = &state->hidden_states[(state->sequence_length - 1) * EMBED_DIM];
    intermediate_t logits[VOCAB_SIZE];
    
    quantized_matmul(final_hidden, model_weights.output_weights.data, logits,
                    1, EMBED_DIM, VOCAB_SIZE,
                    1.0f / 256.0f, model_weights.output_weights.scale);
    
    // Find argmax
    uint16_t best_token = 0;
    intermediate_t best_score = logits[0];
    for (size_t i = 1; i < VOCAB_SIZE; i++) {{
        if (logits[i] > best_score) {{
            best_score = logits[i];
            best_token = i;
        }}
    }}
    
    *output_token = best_token;
    return 0;
}}

int bitgen_generate(BitGenState* state, const uint16_t* prompt, uint16_t prompt_len,
                   uint16_t* output, uint16_t max_length) {{
    // Copy prompt to state
    state->sequence_length = (prompt_len < MAX_SEQ_LEN) ? prompt_len : MAX_SEQ_LEN;
    
    // Generate tokens one by one
    for (uint16_t i = 0; i < max_length; i++) {{
        uint16_t input_tokens[MAX_SEQ_LEN];
        memcpy(input_tokens, prompt, state->sequence_length * sizeof(uint16_t));
        
        uint16_t next_token;
        if (bitgen_forward(state, input_tokens, &next_token) != 0) {{
            return -1;
        }}
        
        output[i] = next_token;
        
        // Update sequence for next iteration
        if (state->sequence_length < MAX_SEQ_LEN) {{
            input_tokens[state->sequence_length] = next_token;
            state->sequence_length++;
        }} else {{
            // Shift sequence (sliding window)
            for (uint16_t j = 0; j < MAX_SEQ_LEN - 1; j++) {{
                input_tokens[j] = input_tokens[j + 1];
            }}
            input_tokens[MAX_SEQ_LEN - 1] = next_token;
        }}
        
        // Stop on EOS token
        if (next_token == 3) break; // EOS token ID
    }}
    
    return 0;
}}

void bitgen_reset_cache(BitGenState* state) {{
    memset(state->attention_cache, 0, sizeof(state->attention_cache));
    memset(state->attention_cache_pos, 0, sizeof(state->attention_cache_pos));
    state->sequence_length = 0;
}}
"""

    def _generate_c_weights(self, quantized_model: Dict) -> str:
        """Generate C weight data file"""
        weights = quantized_model['quantized_weights']

        content = f"""
#include "bitgen_embedded.h"

// Quantized weight data
"""

        # Generate weight arrays
        for weight_name, weight_data in weights.items():
            safe_name = weight_name.replace('.', '_').replace('[', '_').replace(']', '')
            data_array = weight_data['data'].flatten()

            content += f"""
static const quantized_t {safe_name}_data[] = {{
    {', '.join(map(str, data_array[:100]))}{',' if len(data_array) > 100 else ''}
    // ... truncated for brevity, full array would be here
}};
"""

        # Generate main weights structure
        content += """
const BitGenWeights model_weights = {
    // Embedding weights
    .embedding_weights = {
        .data = (quantized_t*)embedding_weight_data,
        .scale = 0.1f,
        .size = VOCAB_SIZE * EMBED_DIM
    },
    // ... other weights would be initialized here
};
"""

        return content

    def _generate_makefile(self) -> str:
        """Generate Makefile for embedded compilation"""
        return """
# Makefile for BitGen Embedded

CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m4 -mthumb -O2 -Wall -Wextra -std=c99
CFLAGS += -fno-builtin -ffunction-sections -fdata-sections
LDFLAGS = -Wl,--gc-sections

SOURCES = bitgen_embedded.c bitgen_weights.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = bitgen_embedded

all: $(TARGET).elf

$(TARGET).elf: $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET).elf

flash: $(TARGET).elf
	# Add your microcontroller flashing command here
	# Example: openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "program $(TARGET).elf verify reset exit"

.PHONY: all clean flash
"""

class EmbeddedBenchmark:
    """Benchmark BitGen performance for embedded deployment"""

    def __init__(self):
        self.results = {}

    def benchmark_memory_usage(self, model: nn.Module, config) -> Dict:
        """Benchmark memory usage patterns"""
        model.eval()

        # Calculate model memory footprint
        param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # MB

        # Estimate runtime memory for different sequence lengths
        runtime_memory = {}

        for seq_len in [32, 64, 128, 256]:
            if seq_len <= config.max_seq_len:
                # Activation memory estimation
                batch_size = 1
                activation_memory = (
                    seq_len * config.embed_dim * 2 +  # Hidden states
                    config.num_layers * config.attention_sinks * config.embed_dim * 2 +  # Attention cache
                    config.memory_size * config.embed_dim * 2 +  # Episodic memory
                    seq_len * config.vocab_size * 2   # Output logits
                ) * 4 / (1024 * 1024)  # Convert to MB

                runtime_memory[seq_len] = activation_memory

        return {
            'parameter_memory_mb': param_memory,
            'runtime_memory_mb': runtime_memory,
            'total_memory_estimate_mb': param_memory + max(runtime_memory.values())
        }

    def benchmark_inference_speed(self, model: nn.Module, config, num_trials: int = 100) -> Dict:
        """Benchmark inference speed"""
        model.eval()

        # Generate test input
        batch_size = 1
        seq_len = min(64, config.max_seq_len)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_trials):
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

                if torch.cuda.is_available():
                    start_time.record()
                    _ = model(input_ids)
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed = start_time.elapsed_time(end_time)
                else:
                    import time
                    start = time.time()
                    _ = model(input_ids)
                    end = time.time()
                    elapsed = (end - start) * 1000  # Convert to ms

                times.append(elapsed)

        return {
            'avg_inference_time_ms': np.mean(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'std_inference_time_ms': np.std(times),
            'tokens_per_second': (seq_len * 1000) / np.mean(times)
        }

    def benchmark_power_estimation(self, model: nn.Module, config) -> Dict:
        """Estimate power consumption for embedded deployment"""

        # Model complexity metrics
        total_params = sum(p.numel() for p in model.parameters())
        total_ops_per_token = (
            config.embed_dim * config.vocab_size +  # Embedding lookup
            config.num_layers * config.embed_dim * config.embed_dim * 4 +  # Attention
            config.num_layers * config.embed_dim * config.ffn_dim * 2 +  # FFN
            config.embed_dim * config.vocab_size  # Output projection
        )

        # Power estimates (rough approximations for ARM Cortex-M4)
        # Based on typical microcontroller specifications
        base_power_mw = 50  # Base MCU power consumption

        # Dynamic power scaling with model complexity
        compute_power_factor = total_ops_per_token / 1e6  # Scale factor
        memory_power_factor = total_params / 1e6

        estimated_power_mw = (
            base_power_mw +
            compute_power_factor * 10 +  # 10mW per million ops
            memory_power_factor * 5      # 5mW per million parameters
        )

        # Battery life estimation (assuming 1000mAh battery at 3.3V)
        battery_capacity_mwh = 1000 * 3.3  # 3300 mWh
        estimated_battery_life_hours = battery_capacity_mwh / estimated_power_mw

        return {
            'estimated_power_consumption_mw': estimated_power_mw,
            'estimated_battery_life_hours': estimated_battery_life_hours,
            'compute_complexity': total_ops_per_token,
            'parameter_count': total_params
        }

    def generate_deployment_report(self, model: nn.Module, config, output_file: str):
        """Generate comprehensive deployment report"""

        # Run all benchmarks
        memory_results = self.benchmark_memory_usage(model, config)
        speed_results = self.benchmark_inference_speed(model, config)
        power_results = self.benchmark_power_estimation(model, config)

        # Compile report
        report = {
            'model_config': config.__dict__,
            'memory_analysis': memory_results,
            'performance_analysis': speed_results,
            'power_analysis': power_results,
            'deployment_recommendations': self._generate_recommendations(
                memory_results, speed_results, power_results, config
            )
        }

        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Deployment report saved to {output_file}")
        return report

    def _generate_recommendations(self, memory_results, speed_results, power_results, config):
        """Generate deployment recommendations"""
        recommendations = []

        # Memory recommendations
        total_memory = memory_results['total_memory_estimate_mb']
        if total_memory > 2:  # 2MB threshold
            recommendations.append(
                "Model may exceed typical microcontroller memory. Consider reducing embed_dim or num_layers."
            )
        elif total_memory > 1:
            recommendations.append(
                "Model approaches memory limits. Test on target hardware before deployment."
            )

        # Speed recommendations
        avg_inference_ms = speed_results['avg_inference_time_ms']
        if avg_inference_ms > 1000:  # 1 second threshold
            recommendations.append(
                "Inference time may be too slow for real-time applications. Consider model pruning."
            )

        # Power recommendations
        power_mw = power_results['estimated_power_consumption_mw']
        battery_life = power_results['estimated_battery_life_hours']

        if battery_life < 8:  # Less than 8 hours
            recommendations.append(
                f"Estimated battery life ({battery_life:.1f}h) may be insufficient. Consider power optimizations."
            )

        if power_mw > 100:  # 100mW threshold
            recommendations.append(
                "High power consumption detected. Consider reducing model complexity."
            )

        # Configuration recommendations
        if config.max_seq_len > 128:
            recommendations.append(
                "Long sequences increase memory usage. Consider reducing max_seq_len for embedded deployment."
            )

        if not recommendations:
            recommendations.append(
                "Model appears suitable for embedded deployment with current configuration."
            )

        return recommendations

def export_for_microcontroller(model_path: str, output_dir: str, target_board: str = "cortex-m4"):
    """Export BitGen model for specific microcontroller deployment"""

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config_dict = checkpoint['config']

    # Recreate config object
    from bitgen_model import BitGenConfig
    config = BitGenConfig(**config_dict)

    # Create model and load weights
    from bitgen_model import create_bitgen_model
    model = create_bitgen_model('tiny')  # Adjust size as needed
    model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize optimizer and benchmark
    optimizer = EmbeddedModelOptimizer(
        target_memory_kb=512 if target_board == "cortex-m4" else 256,
        target_flash_mb=4 if target_board == "cortex-m4" else 2
    )

    benchmark = EmbeddedBenchmark()

    # Generate deployment report
    report = benchmark.generate_deployment_report(
        model, config, os.path.join(output_dir, "deployment_report.json")
    )

    # Quantize model
    print("Quantizing model for embedded deployment...")
    quantized_model = optimizer.quantize_model_for_embedded(model, config)

    print(f"Compression achieved: {quantized_model['compression_stats']['compression_ratio']:.2f}x")
    print(f"Model size: {quantized_model['compression_stats']['quantized_size_mb']:.2f} MB")

    # Generate C code
    print("Generating C inference code...")
    optimizer.generate_c_inference_code(quantized_model, output_dir)

    # Save quantized model
    torch.save(quantized_model, os.path.join(output_dir, "bitgen_quantized.pt"))

    print(f"Embedded deployment files generated in {output_dir}")
    print("Next steps:")
    print("1. Review deployment_report.json for compatibility")
    print("2. Compile C code using provided Makefile")
    print("3. Flash to target microcontroller")
    print("4. Test inference performance on actual hardware")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export BitGen for embedded deployment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for embedded files")
    parser.add_argument("--target_board", type=str, default="cortex-m4",
                       choices=["cortex-m0", "cortex-m4", "cortex-m7"],
                       help="Target microcontroller board")

    args = parser.parse_args()

    export_for_microcontroller(args.model_path, args.output_dir, args.target_board)
