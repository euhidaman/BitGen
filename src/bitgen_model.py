"""
BitGen: Advanced Tiny Language Model for Embedded Microcontrollers
Core model architecture integrating all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BitGenConfig:
    """Configuration for BitGen model optimized for embedded systems"""
    # Model dimensions (optimized for microcontrollers)
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 16
    ffn_dim: int = 256
    vocab_size: int = 8192  # Reduced vocabulary for embedded

    # Episodic Memory (Larimar)
    memory_size: int = 64
    memory_dim: int = 128
    direct_writing: bool = True

    # BitNet Quantization
    quantization_bits: float = 1.58
    use_int8_inference: bool = True

    # FIBER Cross-Modal
    vision_embed_dim: int = 128
    fusion_layers: int = 2

    # Attention Sinks
    attention_sinks: int = 4
    window_size: int = 128

    # Tiny-R1 Reasoning
    reasoning_dim: int = 64
    max_reasoning_steps: int = 8

    # Robot Selection
    num_robots: int = 16
    robot_embed_dim: int = 32

    # Embedded Optimizations
    max_seq_len: int = 256  # Short sequences for embedded
    use_flash_attention: bool = False  # Not available on embedded
    gradient_checkpointing: bool = True

    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

class BitNetLinear(nn.Module):
    """1.58-bit quantized linear layer for embedded deployment"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store weights in full precision for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Quantization parameters
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('input_scale', torch.ones(1))

    def quantize_weights(self, weights):
        """Quantize weights to {-1, 0, +1} (1.58 bits)"""
        # Calculate scale
        scale = weights.abs().mean()

        # Quantize to ternary values
        quantized = torch.sign(weights) * (weights.abs() > 0.5 * scale).float()
        return quantized, scale

    def quantize_activations(self, x):
        """Quantize activations to 8-bit for embedded efficiency"""
        scale = x.abs().max() / 127.0
        quantized = torch.clamp(torch.round(x / scale), -128, 127)
        return quantized, scale

    def forward(self, x):
        if self.training:
            # Full precision during training
            return F.linear(x, self.weight, self.bias)
        else:
            # Quantized inference for embedded deployment
            q_weight, w_scale = self.quantize_weights(self.weight)
            q_input, i_scale = self.quantize_activations(x)

            # Efficient integer arithmetic
            output = F.linear(q_input, q_weight) * w_scale * i_scale

            if self.bias is not None:
                output += self.bias

            return output

class EpisodicMemory(nn.Module):
    """Larimar-inspired episodic memory for embedded systems"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.memory_size = config.memory_size
        self.memory_dim = config.memory_dim
        self.embed_dim = config.embed_dim

        # Memory parameters
        self.memory_keys = nn.Parameter(torch.randn(config.memory_size, config.memory_dim))
        self.memory_values = nn.Parameter(torch.randn(config.memory_size, config.embed_dim))

        # Memory operations
        self.key_proj = BitNetLinear(config.embed_dim, config.memory_dim)
        self.value_proj = BitNetLinear(config.embed_dim, config.embed_dim)
        self.output_proj = BitNetLinear(config.embed_dim, config.embed_dim)

        # Memory update mechanism
        self.update_gate = BitNetLinear(config.embed_dim * 2, 1)

    def forward(self, x, update_memory=True):
        """Forward pass with episodic memory retrieval and update"""
        batch_size, seq_len, embed_dim = x.shape

        # Project input to memory space
        query_keys = self.key_proj(x)  # [B, S, memory_dim]
        query_values = self.value_proj(x)  # [B, S, embed_dim]

        # Memory retrieval
        similarities = torch.matmul(query_keys, self.memory_keys.T)  # [B, S, memory_size]
        attention_weights = F.softmax(similarities / math.sqrt(self.memory_dim), dim=-1)

        retrieved_memories = torch.matmul(attention_weights, self.memory_values)  # [B, S, embed_dim]

        # Combine input with retrieved memories
        combined = x + retrieved_memories
        output = self.output_proj(combined)

        # Memory update during training
        if self.training and update_memory:
            self.update_memories(query_keys, query_values, attention_weights)

        return output

    def update_memories(self, keys, values, attention_weights):
        """Update episodic memory with new experiences"""
        # Compute memory update gates
        batch_size, seq_len, _ = keys.shape

        # Find most activated memory slots
        max_attention, max_indices = attention_weights.max(dim=1)  # [B, memory_size]

        # Update memory slots with high attention
        update_mask = (max_attention > 0.1).float()

        for b in range(batch_size):
            for m in range(self.memory_size):
                if update_mask[b, m] > 0:
                    # Weighted update of memory
                    weight = max_attention[b, m].item()
                    best_seq_idx = attention_weights[b, :, m].argmax().item()

                    self.memory_keys.data[m] = (1 - weight * 0.1) * self.memory_keys.data[m] + \
                                              weight * 0.1 * keys[b, best_seq_idx]
                    self.memory_values.data[m] = (1 - weight * 0.1) * self.memory_values.data[m] + \
                                                weight * 0.1 * values[b, best_seq_idx]

class AttentionSink(nn.Module):
    """Attention sinks for memory-efficient long sequences on embedded systems"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.attention_sinks = config.attention_sinks
        self.window_size = config.window_size

        # Linear projections
        self.q_proj = BitNetLinear(config.embed_dim, config.num_heads * config.head_dim)
        self.k_proj = BitNetLinear(config.embed_dim, config.num_heads * config.head_dim)
        self.v_proj = BitNetLinear(config.embed_dim, config.num_heads * config.head_dim)
        self.out_proj = BitNetLinear(config.num_heads * config.head_dim, config.embed_dim)

    def forward(self, x, cache=None):
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention sinks implementation
        if cache is not None and len(cache) > 0:
            # Concatenate with cached keys and values
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

            # Keep only sink tokens + recent window
            total_len = k.size(2)
            if total_len > self.attention_sinks + self.window_size:
                # Keep sink tokens (first few) + recent window
                sink_k = k[:, :, :self.attention_sinks]
                sink_v = v[:, :, :self.attention_sinks]
                recent_k = k[:, :, -(self.window_size):]
                recent_v = v[:, :, -(self.window_size):]

                k = torch.cat([sink_k, recent_k], dim=2)
                v = torch.cat([sink_v, recent_v], dim=2)

        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask for autoregressive generation
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, k.size(2)), diagonal=1).bool()
            mask = mask.to(scores.device)
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(out)

        # Update cache for next iteration
        new_cache = (k[:, :, -self.window_size:], v[:, :, -self.window_size:])

        return output, new_cache

class CrossModalFusion(nn.Module):
    """FIBER-inspired cross-modal fusion for image-text understanding"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.vision_embed_dim = config.vision_embed_dim

        # Vision encoder (simplified DinoV2-like)
        self.vision_encoder = nn.Sequential(
            BitNetLinear(3 * 14 * 14, config.vision_embed_dim),  # Patch embedding
            nn.ReLU(),
            BitNetLinear(config.vision_embed_dim, config.embed_dim)
        )

        # Cross-modal attention
        self.text_to_vision_attn = AttentionSink(config)
        self.vision_to_text_attn = AttentionSink(config)

        # Fusion layers
        self.fusion_mlp = nn.Sequential(
            BitNetLinear(config.embed_dim * 2, config.ffn_dim),
            nn.ReLU(),
            BitNetLinear(config.ffn_dim, config.embed_dim)
        )

    def encode_vision(self, images):
        """Encode images with quantized vision backbone"""
        batch_size, channels, height, width = images.shape

        # Simple patch extraction (14x14 patches for efficiency)
        patches = F.adaptive_avg_pool2d(images, (14, 14))
        patches = patches.view(batch_size, -1)

        # Vision encoding with quantization
        vision_features = self.vision_encoder(patches)
        return vision_features.unsqueeze(1)  # [B, 1, embed_dim]

    def forward(self, text_embeddings, images=None):
        """Cross-modal fusion of text and vision"""
        if images is None:
            return text_embeddings

        # Encode vision
        vision_features = self.encode_vision(images)

        # Cross-modal attention
        text_attended, _ = self.text_to_vision_attn(
            torch.cat([text_embeddings, vision_features], dim=1)
        )

        # Take only text part
        text_attended = text_attended[:, :-1, :]

        # Fusion
        fused = torch.cat([text_embeddings, text_attended], dim=-1)
        output = self.fusion_mlp(fused)

        return output

class ReasoningModule(nn.Module):
    """Tiny-R1 inspired reasoning module for embedded systems"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.reasoning_dim = config.reasoning_dim
        self.max_steps = config.max_reasoning_steps

        # Reasoning components
        self.reasoning_encoder = BitNetLinear(config.embed_dim, config.reasoning_dim)
        self.step_processor = nn.LSTM(config.reasoning_dim, config.reasoning_dim, batch_first=True)
        self.reasoning_decoder = BitNetLinear(config.reasoning_dim, config.embed_dim)

        # Step controller
        self.step_gate = BitNetLinear(config.reasoning_dim, 1)

    def forward(self, x):
        """Multi-step reasoning process"""
        batch_size, seq_len, embed_dim = x.shape

        # Encode to reasoning space
        reasoning_input = self.reasoning_encoder(x.mean(dim=1))  # [B, reasoning_dim]

        # Multi-step reasoning
        reasoning_states = []
        hidden = None

        current_state = reasoning_input.unsqueeze(1)  # [B, 1, reasoning_dim]

        for step in range(self.max_steps):
            # Process reasoning step
            output, hidden = self.step_processor(current_state, hidden)
            reasoning_states.append(output)

            # Check if reasoning should continue
            gate_score = torch.sigmoid(self.step_gate(output.squeeze(1)))
            if (gate_score < 0.5).all() and step > 2:  # Minimum 3 steps
                break

            current_state = output

        # Aggregate reasoning steps
        final_reasoning = torch.stack(reasoning_states, dim=1).mean(dim=1)  # [B, 1, reasoning_dim]

        # Decode back to embedding space
        reasoning_output = self.reasoning_decoder(final_reasoning)  # [B, embed_dim]

        # Broadcast to sequence length
        reasoning_output = reasoning_output.unsqueeze(1).expand(-1, seq_len, -1)

        return x + reasoning_output

class RobotSelector(nn.Module):
    """Robot selection module based on task requirements"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.num_robots = config.num_robots
        self.robot_embed_dim = config.robot_embed_dim

        # Robot embeddings
        self.robot_embeddings = nn.Parameter(torch.randn(config.num_robots, config.robot_embed_dim))

        # Selection network
        self.task_encoder = BitNetLinear(config.embed_dim, config.robot_embed_dim)
        self.selector = BitNetLinear(config.robot_embed_dim * 2, config.num_robots)

    def forward(self, task_representation):
        """Select best robot for given task"""
        batch_size = task_representation.size(0)

        # Encode task
        task_encoded = self.task_encoder(task_representation.mean(dim=1))  # [B, robot_embed_dim]

        # Compute similarity with all robots
        similarities = []
        for robot_emb in self.robot_embeddings:
            combined = torch.cat([task_encoded, robot_emb.unsqueeze(0).expand(batch_size, -1)], dim=-1)
            sim = self.selector(combined)
            similarities.append(sim)

        robot_scores = torch.stack(similarities, dim=-1).mean(dim=1)  # [B, num_robots]

        return F.softmax(robot_scores, dim=-1)

class BitGenModel(nn.Module):
    """Complete BitGen model for embedded microcontrollers"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # Core components
        self.episodic_memory = EpisodicMemory(config)
        self.attention_layers = nn.ModuleList([
            AttentionSink(config) for _ in range(config.num_layers)
        ])
        self.cross_modal_fusion = CrossModalFusion(config)
        self.reasoning_module = ReasoningModule(config)
        self.robot_selector = RobotSelector(config)

        # Output layers
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.output_projection = BitNetLinear(config.embed_dim, config.vocab_size)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for embedded optimization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, images=None, return_robot_selection=False, attention_cache=None):
        """Forward pass through BitGen model"""
        batch_size, seq_len = input_ids.shape

        # Token and position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)

        x = self.dropout(token_emb + pos_emb)

        # Episodic memory integration
        x = self.episodic_memory(x)

        # Cross-modal fusion with images
        x = self.cross_modal_fusion(x, images)

        # Multi-layer attention with sinks
        new_cache = []
        cache_idx = 0

        for attention_layer in self.attention_layers:
            layer_cache = attention_cache[cache_idx] if attention_cache else None
            x, cache = attention_layer(x, layer_cache)
            new_cache.append(cache)
            cache_idx += 1

        # Reasoning module
        x = self.reasoning_module(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Output projection
        logits = self.output_projection(x)

        # Robot selection
        robot_probs = None
        if return_robot_selection:
            robot_probs = self.robot_selector(x)

        return {
            'logits': logits,
            'robot_selection': robot_probs,
            'attention_cache': new_cache
        }

    def generate_embedded(self, input_ids, max_length=50, temperature=0.7, cache=None):
        """Optimized generation for embedded deployment"""
        self.eval()

        generated = input_ids.clone()
        current_cache = cache

        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(
                    generated[:, -1:] if current_cache else generated,
                    attention_cache=current_cache
                )

                # Sample next token
                logits = outputs['logits'][:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                generated = torch.cat([generated, next_token], dim=-1)
                current_cache = outputs['attention_cache']

                # Stop on EOS token
                if next_token.item() == self.config.vocab_size - 1:  # Assuming last token is EOS
                    break

        return generated, current_cache

    def export_for_embedded(self, path: str):
        """Export model for embedded deployment with quantization"""
        self.eval()

        # Quantize all BitNetLinear layers
        for module in self.modules():
            if isinstance(module, BitNetLinear):
                # Pre-quantize weights
                q_weight, w_scale = module.quantize_weights(module.weight)
                module.weight.data = q_weight
                module.weight_scale.data = w_scale

        # Save quantized model
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'quantized': True
        }, path)

        print(f"Quantized model exported to {path}")

        # Calculate model size
        param_count = sum(p.numel() for p in self.parameters())
        size_mb = param_count * 4 / (1024 * 1024)  # Assuming fp32
        quantized_size_mb = param_count * 1.58 / 8 / (1024 * 1024)  # 1.58 bits per parameter

        print(f"Original size: {size_mb:.2f} MB")
        print(f"Quantized size: {quantized_size_mb:.2f} MB")
        print(f"Compression ratio: {size_mb / quantized_size_mb:.2f}x")

# Factory function for different model sizes
def create_bitgen_model(size='tiny'):
    """Create BitGen model optimized for different embedded systems"""

    if size == 'nano':
        # Ultra-tiny for smallest microcontrollers
        config = BitGenConfig(
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            memory_size=32,
            vocab_size=4096
        )
    elif size == 'tiny':
        # Default configuration
        config = BitGenConfig()
    elif size == 'small':
        # Larger model for more capable embedded systems
        config = BitGenConfig(
            embed_dim=256,
            num_layers=6,
            num_heads=8,
            memory_size=128,
            ffn_dim=512
        )
    else:
        raise ValueError(f"Unknown model size: {size}")

    return BitGenModel(config)
