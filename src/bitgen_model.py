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

    # Robot Selection with Explicit Types
    robot_types: List[str] = None  # Will be populated from dataset
    num_robots: int = 5  # Exactly 5 robots from multi_robot_selection_dataset.json
    top_k_robots: int = 3  # Top-3 robot selection for multi-robot deployment
    robot_embed_dim: int = 32

    def __post_init__(self):
        """Initialize robot types from multi_robot_selection_dataset.json"""
        if self.robot_types is None:
            # EXACT robot types from multi_robot_selection_dataset.json - DO NOT MODIFY
            # These must match the dataset exactly for confusion matrix alignment
            self.robot_types = [
                "Drone",
                "Underwater Robot",
                "Humanoid",
                "Robot with Wheels",
                "Robot with Legs"
            ]
        self.num_robots = len(self.robot_types)

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
        # FIXED: Better initialization for faster convergence
        self.weight = nn.Parameter(torch.randn(
            out_features, in_features) * math.sqrt(2.0 / in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Quantization parameters (used only during inference)
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('input_scale', torch.ones(1))

    def quantize_weights(self, weights):
        """Quantize weights to {-1, 0, +1} (1.58 bits) - INFERENCE ONLY"""
        # Calculate scale
        scale = weights.abs().mean().clamp(min=1e-5)

        # Quantize to ternary values
        quantized = torch.sign(weights) * (weights.abs() > 0.5 * scale).float()
        return quantized, scale

    def quantize_activations(self, x):
        """Quantize activations to 8-bit for embedded efficiency - INFERENCE ONLY"""
        scale = x.abs().max().clamp(min=1e-8) / 127.0
        quantized = torch.clamp(torch.round(x / scale), -128, 127)
        return quantized, scale

    def forward(self, x):
        # FIXED: Always use full precision during training for proper gradient flow
        if self.training:
            # Full precision during training - NO QUANTIZATION
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
        self.memory_keys = nn.Parameter(torch.randn(
            config.memory_size, config.memory_dim))
        self.memory_values = nn.Parameter(
            torch.randn(config.memory_size, config.embed_dim))

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
        similarities = torch.matmul(
            query_keys, self.memory_keys.T)  # [B, S, memory_size]
        attention_weights = F.softmax(
            similarities / math.sqrt(self.memory_dim), dim=-1)

        retrieved_memories = torch.matmul(
            attention_weights, self.memory_values)  # [B, S, embed_dim]

        # Combine input with retrieved memories
        combined = x + retrieved_memories
        output = self.output_proj(combined)

        # Memory update during training
        if self.training and update_memory:
            self.update_memories(query_keys, query_values, attention_weights)

        # Return additional info for analysis
        memory_info = {
            'memory_keys': self.memory_keys,
            'memory_values': self.memory_values,
            'memory_attention': attention_weights,
            'retrieved_memories': retrieved_memories,
            'memory_similarities': similarities
        }

        return output, memory_info

    def update_memories(self, keys, values, attention_weights):
        """Update episodic memory with new experiences"""
        # Compute memory update gates
        batch_size, seq_len, _ = keys.shape

        # Find most activated memory slots
        max_attention, max_indices = attention_weights.max(
            dim=1)  # [B, memory_size]

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


class VisionAttentionSink(nn.Module):
    """Attention layer specifically for vision processing with correct dimensions"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.vision_embed_dim // config.num_heads  # Use vision_embed_dim
        self.attention_sinks = config.attention_sinks
        self.window_size = config.window_size

        # Linear projections for vision embeddings
        self.q_proj = BitNetLinear(
            config.vision_embed_dim, config.num_heads * self.head_dim)
        self.k_proj = BitNetLinear(
            config.vision_embed_dim, config.num_heads * self.head_dim)
        self.v_proj = BitNetLinear(
            config.vision_embed_dim, config.num_heads * self.head_dim)
        self.out_proj = BitNetLinear(
            config.num_heads * self.head_dim, config.vision_embed_dim)

    def forward(self, x, cache=None):
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(out)

        return output, None  # No cache for vision processing


class AttentionSink(nn.Module):
    """Attention with sliding window and sink tokens for memory efficiency"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.attention_sinks = config.attention_sinks
        self.window_size = config.window_size

        # Linear projections
        self.q_proj = BitNetLinear(
            config.embed_dim, config.num_heads * config.head_dim)
        self.k_proj = BitNetLinear(
            config.embed_dim, config.num_heads * config.head_dim)
        self.v_proj = BitNetLinear(
            config.embed_dim, config.num_heads * config.head_dim)
        self.out_proj = BitNetLinear(
            config.num_heads * config.head_dim, config.embed_dim)

    def forward(self, x, cache=None):
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)

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
            mask = torch.triu(torch.ones(seq_len, k.size(2)),
                              diagonal=1).bool()
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
    """FIBER-inspired advanced cross-modal fusion with mandatory DINOv2 vision encoding"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.vision_embed_dim = config.vision_embed_dim
        self.num_fuse_layers = config.fusion_layers
        
        # FIBER-style queue for contrastive learning (momentum-based)
        self.queue_size = getattr(config, 'queue_size', 8192)  # Default 8192
        self.register_buffer('text_queue', torch.randn(config.embed_dim, self.queue_size))
        self.register_buffer('image_queue', torch.randn(config.embed_dim, self.queue_size))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize queues
        self.text_queue = F.normalize(self.text_queue, p=2, dim=0)
        self.image_queue = F.normalize(self.image_queue, p=2, dim=0)

        # MANDATORY: DINOv2 Vision Encoder - no fallback allowed
        try:
            from transformers import Dinov2Model, Dinov2Config
            print("🔄 Loading facebook/dinov2-base from HuggingFace...")

            # Load DINOv2-base model from HuggingFace
            self.dinov2_model = Dinov2Model.from_pretrained(
                'facebook/dinov2-base')
            # CRITICAL FIX: Keep in train mode to allow gradient flow!
            self.dinov2_model.train()

            # CRITICAL FIX: UNFREEZE DINOv2 parameters to allow learning!
            # The model MUST be trainable for contrastive learning to work
            for param in self.dinov2_model.parameters():
                param.requires_grad = True

            # DINOv2-base outputs 768 dimensions
            self.dinov2_dim = 768
            self.dinov2_to_vision = BitNetLinear(
                self.dinov2_dim, config.vision_embed_dim)

            print(
                "✅ facebook/dinov2-base loaded successfully - TRAINABLE (requires_grad=True)!")

        except ImportError as e:
            raise ImportError(
                f"❌ CRITICAL: transformers library required for DINOv2. "
                f"Install with: pip install transformers\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ CRITICAL: Failed to load facebook/dinov2-base from HuggingFace. "
                f"Check internet connection and HuggingFace access.\n"
                f"Original error: {e}"
            )

        # FIBER-style cross-modal transforms
        self.cross_modal_text_transform = BitNetLinear(
            config.embed_dim, config.embed_dim)
        self.cross_modal_image_transform = BitNetLinear(
            config.vision_embed_dim, config.embed_dim)

        # Contrastive learning transforms (ITC)
        self.cross_modal_text_transform_itc = BitNetLinear(
            config.embed_dim, config.embed_dim)
        self.cross_modal_image_transform_itc = BitNetLinear(
            config.vision_embed_dim, config.embed_dim)

        # FIBER-style poolers
        self.cross_modal_text_pooler = nn.Sequential(
            BitNetLinear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )
        self.cross_modal_image_pooler = nn.Sequential(
            BitNetLinear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )

        # ITC poolers for contrastive learning
        self.cross_modal_text_pooler_itc = nn.Sequential(
            BitNetLinear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )
        self.cross_modal_image_pooler_itc = nn.Sequential(
            BitNetLinear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )

        # Cross-modal attention layers (FIBER-style progressive fusion)
        self.cross_modal_att_layers = nn.ModuleList([
            BitNetLinear(config.embed_dim, config.embed_dim // 2)
            for _ in range(self.num_fuse_layers)
        ])

        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            BitNetLinear(config.embed_dim * 2, config.ffn_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            BitNetLinear(config.ffn_dim, config.embed_dim)
        )

        # Average pooling for image features
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, text_keys, image_keys):
        """Update queues with new features (momentum-based, no gradients)"""
        batch_size = text_keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest features in queue (circular buffer)
        if ptr + batch_size <= self.queue_size:
            self.text_queue[:, ptr:ptr + batch_size] = text_keys.T
            self.image_queue[:, ptr:ptr + batch_size] = image_keys.T
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.text_queue[:, ptr:] = text_keys[:remaining].T
            self.image_queue[:, ptr:] = image_keys[:remaining].T
            self.text_queue[:, :batch_size - remaining] = text_keys[remaining:].T
            self.image_queue[:, :batch_size - remaining] = image_keys[remaining:].T
            ptr = batch_size - remaining
        
        self.queue_ptr[0] = ptr

    def encode_vision_dinov2(self, images):
        """Mandatory DINOv2-base feature extraction from HuggingFace - no fallback"""
        batch_size, channels, height, width = images.shape

        # Preprocess images for DINOv2 (expects 224x224)
        if height != 224 or width != 224:
            images = F.interpolate(images, size=(
                224, 224), mode='bilinear', align_corners=False)

        # Normalize for DINOv2 (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(
            1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(
            1, 3, 1, 1).to(images.device)
        images_normalized = (images - mean) / std

        # CRITICAL FIX: Extract DINOv2 features WITH GRADIENTS for contrastive learning!
        # DINOv2 forward pass - TRAINABLE for end-to-end learning
        outputs = self.dinov2_model(pixel_values=images_normalized)

        # Get patch embeddings (exclude CLS token)
        # outputs.last_hidden_state shape: [batch_size, num_patches + 1, 768]
        # Remove CLS token
        patch_features = outputs.last_hidden_state[:, 1:, :]

        # Project DINOv2 features to our vision embedding dimension
        # Both DINOv2 AND projection are trainable for contrastive learning
        # [batch_size, num_patches, vision_embed_dim]
        vision_features = self.dinov2_to_vision(patch_features)

        return vision_features

    def encode_vision_fiber_style(self, images):
        """FIBER-style vision encoding using mandatory DINOv2-base"""
        return self.encode_vision_dinov2(images)

    def forward(self, text_embeddings, images=None, return_contrastive_features=False):
        """FIBER-style cross-modal fusion with progressive attention"""
        if images is None:
            return text_embeddings

        batch_size, seq_len, embed_dim = text_embeddings.shape

        # Encode vision with FIBER-style approach
        original_image_embeds = self.encode_vision_fiber_style(
            images)  # [B, num_patches, vision_embed_dim=128]

        # Transform to common embedding space
        image_embeds = self.cross_modal_image_transform(
            original_image_embeds)  # [B, num_patches, embed_dim=256]
        text_embeds = self.cross_modal_text_transform(
            text_embeddings)  # [B, seq_len, embed_dim=256]

        # FIBER-style progressive cross-modal attention fusion
        fused_text = text_embeds.clone()

        for i, cross_att_layer in enumerate(self.cross_modal_att_layers):
            # Apply cross-attention from text to image (similar to FIBER's encoder-decoder approach)
            # Simplified cross-attention mechanism

            # Query from text, Key-Value from image
            q = fused_text  # [B, seq_len, embed_dim]
            # [B, num_patches, embed_dim//2]
            k = v = cross_att_layer(image_embeds)

            # Expand k, v to match text embedding dimension
            k = torch.cat([k, k], dim=-1)  # [B, num_patches, embed_dim]
            v = torch.cat([v, v], dim=-1)  # [B, num_patches, embed_dim]

            # Simplified attention computation
            attn_scores = torch.matmul(
                q, k.transpose(-2, -1)) / math.sqrt(embed_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Apply attention to values
            cross_attended = torch.matmul(
                attn_weights, v)  # [B, seq_len, embed_dim]

            # Residual connection
            fused_text = fused_text + cross_attended * 0.1  # Small weight for stability

        # Final fusion
        # Create image summary for concatenation
        avg_image_features = self.avgpool(
            image_embeds.transpose(1, 2)).view(batch_size, 1, embed_dim)
        avg_image_features = avg_image_features.expand(-1, seq_len, -1)

        # Concatenate and fuse
        concatenated = torch.cat([fused_text, avg_image_features], dim=-1)
        fused_output = self.fusion_mlp(concatenated)

        # Generate contrastive features if requested (for ITC loss)
        if return_contrastive_features:
            # Text features for contrastive learning (FIBER: NO normalization!)
            text_itc = self.cross_modal_text_pooler_itc(
                self.cross_modal_text_transform_itc(text_embeddings)[:, 0:1]
            ).squeeze(1)  # [B, D]

            # Image features for contrastive learning (FIBER: NO normalization!)
            image_itc = self.cross_modal_image_transform_itc(original_image_embeds)
            image_avg = self.avgpool(image_itc.transpose(1, 2)).view(batch_size, 1, -1)
            image_cls = self.cross_modal_image_pooler_itc(image_avg).squeeze(1)  # [B, D]

            # Update queues (momentum-based, no gradients) - FIBER style
            if self.training:
                self._dequeue_and_enqueue(image_cls.clone().detach(), text_itc.clone().detach())

            return fused_output, {
                'text_features': text_itc,  # [B, D] - raw pooler output, NOT normalized
                'image_features': image_cls,  # [B, D] - raw pooler output, NOT normalized
                'text_queue': self.text_queue.clone(),  # [D, Q]
                'image_queue': self.image_queue.clone(),  # [D, Q]
                'temperature': self.temperature
            }

        return fused_output

    def compute_contrastive_loss(self, text_features, image_features, temperature):
        """FIBER-style contrastive loss computation"""
        batch_size = text_features.size(0)

        # Compute similarity matrix
        logits_per_text = torch.matmul(
            text_features, image_features.T) / temperature
        logits_per_image = logits_per_text.T

        # Labels for contrastive learning (diagonal)
        labels = torch.arange(batch_size, device=text_features.device)

        # Compute losses
        text_loss = F.cross_entropy(logits_per_text, labels)
        image_loss = F.cross_entropy(logits_per_image, labels)

        return (text_loss + image_loss) / 2

    def encode_vision(self, images):
        """Legacy method for backward compatibility"""
        return self.encode_vision_fiber_style(images)


class ReasoningModule(nn.Module):
    """Tiny-R1 inspired Chain-of-Thought reasoning module for robot selection"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.reasoning_dim = config.reasoning_dim
        self.max_steps = config.max_reasoning_steps

        # Chain-of-thought reasoning components
        self.reasoning_encoder = BitNetLinear(
            config.embed_dim, config.reasoning_dim)
        self.step_processor = nn.LSTM(
            config.reasoning_dim, config.reasoning_dim, batch_first=True)
        self.reasoning_decoder = BitNetLinear(
            config.reasoning_dim, config.embed_dim)

        # Step controller with confidence tracking
        self.step_gate = BitNetLinear(config.reasoning_dim, 1)
        self.step_confidence = BitNetLinear(config.reasoning_dim, 1)

    def forward(self, x, return_reasoning_trace=False):
        """Multi-step reasoning process with explicit chain-of-thought traces"""
        batch_size, seq_len, embed_dim = x.shape

        # Encode to reasoning space
        reasoning_input = self.reasoning_encoder(
            x.mean(dim=1))  # [B, reasoning_dim]

        # Store reasoning traces for chain-of-thought analysis (like tiny-r1)
        reasoning_states = []
        reasoning_confidences = []
        gate_scores = []
        hidden = None

        current_state = reasoning_input.unsqueeze(1)  # [B, 1, reasoning_dim]

        for step in range(self.max_steps):
            # Process reasoning step
            output, hidden = self.step_processor(current_state, hidden)
            reasoning_states.append(output.squeeze(1))  # Store for trace

            # Compute confidence for this reasoning step
            confidence = torch.sigmoid(self.step_confidence(output.squeeze(1)))
            reasoning_confidences.append(confidence)

            # Check if reasoning should continue
            gate_score = torch.sigmoid(self.step_gate(output.squeeze(1)))
            gate_scores.append(gate_score)

            if (gate_score < 0.5).all() and step > 2:  # Minimum 3 steps
                break

            current_state = output

        # Aggregate reasoning steps
        final_reasoning = torch.stack(reasoning_states, dim=0).mean(
            dim=0)  # [B, reasoning_dim]

        reasoning_output = self.reasoning_decoder(
            final_reasoning)  # [B, embed_dim]
        reasoning_output = reasoning_output.unsqueeze(
            1).expand(-1, seq_len, -1)  # [B, seq_len, embed_dim]

        # Return with optional reasoning trace for logging
        if return_reasoning_trace:
            reasoning_info = {
                # List of tensors [B, reasoning_dim]
                'reasoning_states': reasoning_states,
                'reasoning_confidences': reasoning_confidences,  # List of confidence scores
                'gate_scores': gate_scores,  # List of gate decisions
                # Number of reasoning steps taken
                'num_steps': len(reasoning_states)
            }
            return x + reasoning_output, reasoning_info

        return x + reasoning_output


class RobotSelector(nn.Module):
    """Top-K Multi-Label Robot Selection for multi-robot deployment scenarios"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.num_robots = config.num_robots
        self.robot_embed_dim = config.robot_embed_dim
        self.top_k = config.top_k_robots
        self.robot_types = config.robot_types

        # Learnable robot embeddings (each robot has semantic meaning)
        # STABILITY: Use smaller initialization to prevent extreme values
        self.robot_embeddings = nn.Parameter(
            torch.randn(config.num_robots, config.robot_embed_dim) * 0.02
        )

        # Selection network for multi-label classification (NOT softmax)
        self.task_encoder = BitNetLinear(
            config.embed_dim, config.robot_embed_dim)
        self.robot_scorer = BitNetLinear(
            config.robot_embed_dim * 2, 1)  # Binary score per robot

        # STABILITY: Initialize scorer with small weights
        if hasattr(self.robot_scorer, 'weight'):
            nn.init.normal_(self.robot_scorer.weight, mean=0.0, std=0.02)
            if hasattr(self.robot_scorer, 'bias') and self.robot_scorer.bias is not None:
                nn.init.zeros_(self.robot_scorer.bias)

    def forward(self, task_representation, return_top_k=True):
        """
        Select top-K robots for given task using multi-label classification

        Args:
            task_representation: [B, seq_len, embed_dim] - Task/scene representation
            return_top_k: Whether to return top-K results with robot names

        Returns:
            If return_top_k=True: Dict with all_probs, top_k_probs, top_k_indices, top_k_robots
            If return_top_k=False: [B, num_robots] probability tensor
        """
        batch_size = task_representation.size(0)

        # Encode task representation to robot embedding space
        task_repr_mean = task_representation.mean(dim=1)  # [B, embed_dim]

        # STABILITY: Check for NaN in input
        if torch.isnan(task_repr_mean).any() or torch.isinf(task_repr_mean).any():
            # Return safe default: uniform probabilities
            robot_probs = torch.ones(
                batch_size, self.num_robots, device=task_representation.device) * 0.5
            if return_top_k:
                return {
                    'all_probs': robot_probs,
                    'all_logits': torch.zeros_like(robot_probs),
                    'top_k_probs': robot_probs[:, :self.top_k],
                    'top_k_indices': torch.arange(self.top_k, device=task_representation.device).unsqueeze(0).expand(batch_size, -1),
                    'top_k_robots': [[self.robot_types[i] for i in range(self.top_k)] for _ in range(batch_size)]
                }
            return robot_probs

        task_encoded = self.task_encoder(
            task_repr_mean)  # [B, robot_embed_dim]

        # STABILITY: Normalize task encoding to prevent extreme values
        task_encoded = torch.nn.functional.normalize(
            task_encoded, p=2, dim=-1) * (self.robot_embed_dim ** 0.5)

        # Compute independent score for each robot (multi-label, not mutually exclusive)
        robot_scores = []
        for i, robot_emb in enumerate(self.robot_embeddings):
            # Normalize robot embedding too
            robot_emb_norm = torch.nn.functional.normalize(
                robot_emb, p=2, dim=-1) * (self.robot_embed_dim ** 0.5)

            # Concatenate task encoding with robot embedding
            combined = torch.cat([
                task_encoded,
                robot_emb_norm.unsqueeze(0).expand(batch_size, -1)
            ], dim=-1)
            # Score: how suitable is this robot for this task?
            score = self.robot_scorer(combined).squeeze(-1)  # [B]
            robot_scores.append(score)

        robot_scores = torch.stack(robot_scores, dim=1)  # [B, num_robots]

        # CRITICAL: Clamp logits to prevent numerical overflow in sigmoid
        # sigmoid(x) = 1/(1+exp(-x)) can overflow if x is too large/small
        robot_scores = torch.clamp(robot_scores, min=-50.0, max=50.0)

        # Use sigmoid for independent probabilities (multi-label)
        # Each robot has independent probability
        robot_probs = torch.sigmoid(robot_scores)

        if return_top_k:
            # Get top-K most suitable robots
            top_k_probs, top_k_indices = torch.topk(
                robot_probs, k=min(self.top_k, self.num_robots), dim=1)

            # SAFETY: Clamp indices to valid range to prevent index out of bounds
            max_valid_idx = len(self.robot_types) - 1
            top_k_indices = torch.clamp(top_k_indices, 0, max_valid_idx)

            # Convert indices to robot type names for interpretability
            top_k_robots = []
            for batch_indices in top_k_indices:
                batch_robots = [
                    self.robot_types[idx.item()]
                    for idx in batch_indices
                ]
                top_k_robots.append(batch_robots)

            return {
                # [B, num_robots] - All robot probabilities
                'all_probs': robot_probs,
                # [B, num_robots] - Raw logits (for loss calculation)
                'all_logits': robot_scores,
                'top_k_probs': top_k_probs,  # [B, top_k] - Top-K probabilities
                # [B, top_k] - Top-K robot indices
                'top_k_indices': top_k_indices,
                # List[List[str]] - Top-K robot names
                'top_k_robots': top_k_robots
            }

        return robot_probs


class BitNetTextDecoder(nn.Module):
    """BitNet-based text decoder for text generation"""

    def __init__(self, config: BitGenConfig, num_decoder_layers: int = 2):
        super().__init__()
        self.config = config
        # Use fewer layers for decoder to save memory (default: 2 instead of 6)
        self.num_layers = num_decoder_layers
        self.embed_dim = config.embed_dim

        # Decoder layers with causal attention
        self.decoder_layers = nn.ModuleList([
            self._build_decoder_layer(config) for _ in range(self.num_layers)
        ])

        # Layer norm
        self.layer_norm = nn.LayerNorm(
            config.embed_dim, eps=config.layer_norm_eps)

        # Output projection to vocabulary
        self.output_projection = BitNetLinear(
            config.embed_dim, config.vocab_size)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def _build_decoder_layer(self, config):
        """Build a single decoder layer"""
        return nn.ModuleDict({
            'self_attn': AttentionSink(config),
            # Cross-attention: combine decoder hidden state with encoder output
            'encoder_proj': BitNetLinear(config.embed_dim, config.embed_dim),
            'ffn': nn.Sequential(
                BitNetLinear(config.embed_dim, config.ffn_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                BitNetLinear(config.ffn_dim, config.embed_dim),
                nn.Dropout(config.dropout)
            ),
            'ln1': nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps),
            'ln2': nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps),
            'ln3': nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps),
        })

    def forward(self, x, encoder_output, attention_mask=None, causal_mask=None):
        """
        Forward pass through decoder
        Args:
            x: Decoder input embeddings [B, tgt_len, embed_dim]
            encoder_output: Encoder output [B, src_len, embed_dim]
            attention_mask: Optional attention mask
            causal_mask: Causal mask for autoregressive generation
        Returns:
            logits: [B, tgt_len, vocab_size]
        """
        batch_size, tgt_len, _ = x.shape

        # Create causal mask if not provided
        if causal_mask is None:
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=x.device,
                           dtype=torch.bool),
                diagonal=1
            )

        # Process through decoder layers
        for layer in self.decoder_layers:
            # Self-attention with causal mask (look only at previous tokens)
            residual = x
            x = layer['ln1'](x)

            # Apply self-attention (AttentionSink has built-in causal masking)
            attn_output, _ = layer['self_attn'](x, cache=None)
            x = residual + self.dropout(attn_output)

            # Add encoder context (simplified cross-attention)
            residual = x
            x = layer['ln2'](x)
            # Average pool encoder output to match sequence length
            encoder_context = encoder_output.mean(dim=1, keepdim=True).expand(-1, x.size(1), -1)
            encoder_proj = layer['encoder_proj'](encoder_context)
            x = residual + self.dropout(encoder_proj)

            # Feed-forward
            residual = x
            x = layer['ln3'](x)
            x = residual + layer['ffn'](x)

        # Final layer norm
        x = self.layer_norm(x)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits


class BitGenModel(nn.Module):
    """Complete BitGen model for embedded microcontrollers"""

    def __init__(self, config: BitGenConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # Core components
        self.episodic_memory = EpisodicMemory(config)
        self.attention_layers = nn.ModuleList([
            AttentionSink(config) for _ in range(config.num_layers)
        ])
        self.cross_modal_fusion = CrossModalFusion(config)
        self.reasoning_module = ReasoningModule(config)
        self.robot_selector = RobotSelector(config)

        # Text decoder for reconstruction loss (BitMar-style)
        self.text_decoder = BitNetTextDecoder(config)

        # Output layers (keep for backward compatibility and direct prediction)
        self.layer_norm = nn.LayerNorm(
            config.embed_dim, eps=config.layer_norm_eps)
        self.output_projection = BitNetLinear(
            config.embed_dim, config.vocab_size)

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

    def forward(self, input_ids, images=None, attention_mask=None, return_robot_selection=False, attention_cache=None, return_analysis_data=False, return_attention_weights=False, target_ids=None, use_decoder=False):
        """
        Forward pass through BitGen model with comprehensive token validation

        Args:
            input_ids: Input token IDs [B, seq_len]
            images: Optional image features [B, num_patches, image_dim]
            attention_mask: Optional attention mask
            return_robot_selection: Whether to return robot selection output
            attention_cache: Optional attention cache for inference
            return_analysis_data: Whether to return detailed analysis data
            return_attention_weights: Whether to return attention weights
            target_ids: Target token IDs for text reconstruction [B, tgt_len]
            use_decoder: Whether to use decoder for text generation (BitMar-style)
        """
        batch_size, seq_len = input_ids.shape

        # CRITICAL: Validate input tokens before any embedding operations
        if input_ids.max() >= self.config.vocab_size or input_ids.min() < 0:
            print(f"EMERGENCY: Token validation failed in model forward pass")
            print(
                f"Token range: [{input_ids.min().item()}, {input_ids.max().item()}], vocab_size: {self.config.vocab_size}")
            # Emergency clamp to prevent CUDA errors
            input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
            print(
                f"Applied emergency clamping: [{input_ids.min().item()}, {input_ids.max().item()}]")

        # Token and position embeddings with safe indexing
        try:
            token_emb = self.token_embedding(input_ids)
        except RuntimeError as e:
            if "index" in str(e).lower():
                print(f"Token embedding failed: {e}")
                print(
                    f"input_ids shape: {input_ids.shape}, max: {input_ids.max()}, vocab_size: {self.config.vocab_size}")
                raise RuntimeError(
                    f"Token embedding indexing error: max_token={input_ids.max()}, vocab_size={self.config.vocab_size}")
            raise

        # Safe position embedding
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        if seq_len > self.config.max_seq_len:
            print(
                f"WARNING: Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
            pos_ids = pos_ids[:, :self.config.max_seq_len]

        try:
            pos_emb = self.pos_embedding(pos_ids)
        except RuntimeError as e:
            if "index" in str(e).lower():
                print(f"Position embedding failed: {e}")
                print(
                    f"pos_ids shape: {pos_ids.shape}, max: {pos_ids.max()}, max_seq_len: {self.config.max_seq_len}")
                raise RuntimeError(
                    f"Position embedding indexing error: max_pos={pos_ids.max()}, max_seq_len={self.config.max_seq_len}")
            raise

        x = self.dropout(token_emb + pos_emb)

        # CORRECTED ARCHITECTURE: Cross-modal fusion BEFORE episodic memory
        # This ensures multimodal representations are stored in memory
        contrastive_features = None
        text_features = None
        image_features = None
        similarity_matrix = None

        if images is not None:
            # Get contrastive features and intermediate outputs for monitoring
            fusion_output = self.cross_modal_fusion(
                x, images, return_contrastive_features=True)
            if isinstance(fusion_output, tuple):
                x, contrastive_features = fusion_output
                # Extract text and image features for monitoring
                if hasattr(self.cross_modal_fusion, 'last_text_features'):
                    text_features = self.cross_modal_fusion.last_text_features
                if hasattr(self.cross_modal_fusion, 'last_image_features'):
                    image_features = self.cross_modal_fusion.last_image_features
                if hasattr(self.cross_modal_fusion, 'last_similarity_matrix'):
                    similarity_matrix = self.cross_modal_fusion.last_similarity_matrix
            else:
                x = fusion_output
        else:
            x = self.cross_modal_fusion(x, images)

        # CORRECTED: Episodic memory now processes MULTIMODAL representations
        # This stores the fused text+image features, not just text
        x, memory_info = self.episodic_memory(x)

        # Multi-layer attention with sinks - collect attention weights for analysis
        new_cache = []
        all_attention_weights = []
        cache_idx = 0

        for layer_idx, attention_layer in enumerate(self.attention_layers):
            layer_cache = attention_cache[cache_idx] if attention_cache else None

            # Modified to capture attention weights
            layer_output, cache, attention_weights = self._forward_attention_with_weights(
                attention_layer, x, layer_cache, attention_mask
            )

            x = layer_output
            new_cache.append(cache)
            if return_attention_weights or return_analysis_data:
                all_attention_weights.append(attention_weights)
            cache_idx += 1

        # Reasoning module with chain-of-thought traces (Tiny-R1 style)
        reasoning_input = x
        if return_robot_selection or return_analysis_data:
            # Get reasoning traces for robot selection and logging
            x, reasoning_info = self.reasoning_module(
                x, return_reasoning_trace=True)
        else:
            # Standard reasoning without traces
            x = self.reasoning_module(x, return_reasoning_trace=False)
            reasoning_info = {}

        # Layer normalization
        x = self.layer_norm(x)

        # Store encoder output for decoder
        encoder_output = x

        # Decoder for text reconstruction (BitMar-style)
        decoder_logits = None
        if use_decoder and target_ids is not None:
            # Prepare decoder input (shift targets right, prepend BOS)
            # For training, we use teacher forcing
            tgt_len = target_ids.shape[1]

            # Get target embeddings
            tgt_token_emb = self.token_embedding(target_ids)
            tgt_pos_ids = torch.arange(
                tgt_len, device=target_ids.device).unsqueeze(0)
            tgt_pos_emb = self.pos_embedding(tgt_pos_ids)
            decoder_input = self.dropout(tgt_token_emb + tgt_pos_emb)

            # Run through decoder
            decoder_logits = self.text_decoder(
                decoder_input, encoder_output, attention_mask=attention_mask)

        # Output projection (direct prediction, kept for backward compatibility)
        logits = self.output_projection(x)

        # Robot selection with top-K multi-label output
        robot_selection_output = None
        robot_info = {}
        if return_robot_selection:
            # Get top-K robots with confidence scores
            robot_selection_output = self.robot_selector(x, return_top_k=True)

            # Extract robot selection info for analysis
            robot_info = {
                'all_robot_probs': robot_selection_output['all_probs'],
                'top_k_robot_probs': robot_selection_output['top_k_probs'],
                'top_k_robot_indices': robot_selection_output['top_k_indices'],
                'top_k_robot_names': robot_selection_output['top_k_robots'],
                'robot_embeddings': self.robot_selector.robot_embeddings,
                'reasoning_info': reasoning_info  # Include reasoning traces
            }

        # Prepare output
        outputs = {
            'logits': logits,
            # Text reconstruction logits (BitMar-style)
            'decoder_logits': decoder_logits,
            'robot_selection': robot_selection_output,  # Full top-K robot selection output
            'attention_cache': new_cache
        }

        # Add contrastive features if available
        if contrastive_features is not None:
            outputs['contrastive_features'] = contrastive_features

        # Add monitoring data for wandb
        if return_attention_weights or return_analysis_data:
            outputs['attention_weights'] = all_attention_weights[0] if len(
                all_attention_weights) > 0 else None
            outputs['memory_bank'] = memory_info.get('memory_values')
            outputs['memory_usage'] = {
                'read_count': memory_info.get('read_count', 0),
                'write_count': memory_info.get('write_count', 0),
                'top_k_indices': memory_info.get('top_k_indices', [])
            }
            if text_features is not None:
                outputs['text_features'] = text_features
            if image_features is not None:
                outputs['image_features'] = image_features
            if similarity_matrix is not None:
                outputs['similarity_matrix'] = similarity_matrix

        # Add analysis data if requested
        if return_analysis_data:
            outputs.update({
                # Episodic memory analysis
                'memory_keys': memory_info['memory_keys'],
                'memory_values': memory_info['memory_values'],
                'memory_attention': memory_info['memory_attention'],
                'memory_similarities': memory_info['memory_similarities'],
                'retrieved_memories': memory_info['retrieved_memories'],

                # Attention analysis
                # List of attention weights per layer
                'all_attention_weights': all_attention_weights,
                'input_embeddings': token_emb + pos_emb,
                'final_embeddings': x,

                # Chain-of-Thought Reasoning analysis (Tiny-R1 style)
                'reasoning_states': reasoning_info.get('reasoning_states', []),
                'reasoning_steps_taken': reasoning_info.get('num_steps', 0),
                'reasoning_confidences': reasoning_info.get('reasoning_confidences', []),
                'reasoning_gate_scores': reasoning_info.get('gate_scores', []),

                # Robot selection analysis (Top-K multi-label)
                **robot_info
            })

        # Always include reasoning info if robot selection is enabled
        if return_robot_selection and reasoning_info:
            outputs['reasoning_info'] = reasoning_info

        return outputs

    def _forward_attention_with_weights(self, attention_layer, x, cache, attention_mask=None):
        """Forward attention layer and capture attention weights"""
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        q = attention_layer.q_proj(x).view(
            batch_size, seq_len, attention_layer.num_heads, attention_layer.head_dim).transpose(1, 2)
        k = attention_layer.k_proj(x).view(
            batch_size, seq_len, attention_layer.num_heads, attention_layer.head_dim).transpose(1, 2)
        v = attention_layer.v_proj(x).view(
            batch_size, seq_len, attention_layer.num_heads, attention_layer.head_dim).transpose(1, 2)

        # Handle cache
        if cache is not None and len(cache) > 0:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

            # Keep only sink tokens + recent window
            total_len = k.size(2)
            if total_len > attention_layer.attention_sinks + attention_layer.window_size:
                sink_k = k[:, :, :attention_layer.attention_sinks]
                sink_v = v[:, :, :attention_layer.attention_sinks]
                recent_k = k[:, :, -(attention_layer.window_size):]
                recent_v = v[:, :, -(attention_layer.window_size):]

                k = torch.cat([sink_k, recent_k], dim=2)
                v = torch.cat([sink_v, recent_v], dim=2)

        # Compute attention
        scale = 1.0 / math.sqrt(attention_layer.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, k.size(2)),
                              diagonal=1).bool()
            mask = mask.to(scores.device)
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions [batch, heads, seq_len, key_len]
            if attention_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # [batch, 1, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1)

            # Convert attention_mask (0s and 1s) to additive mask (0s and -inf)
            attention_mask = (1.0 - attention_mask) * -10000.0
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = attention_layer.out_proj(out)

        # Update cache
        new_cache = (k[:, :, -attention_layer.window_size:],
                     v[:, :, -attention_layer.window_size:])

        return output, new_cache, attn_weights

    def compute_loss(self, outputs, target_ids, text_features=None, image_features=None,
                     text_loss_weight=1.0, contrastive_loss_weight=0.1, memory_kl_weight=0.05):
        """
        Compute multi-component loss (BitMar-style)

        Args:
            outputs: Model outputs dictionary
            target_ids: Target token IDs for text reconstruction
            text_features: Text features for contrastive learning
            image_features: Image features for contrastive learning
            text_loss_weight: Weight for text reconstruction loss (default: 1.0)
            contrastive_loss_weight: Weight for contrastive loss (default: 0.1)
            memory_kl_weight: Weight for memory KL divergence (default: 0.05)

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. Text Reconstruction Loss (main learning signal)
        if outputs['decoder_logits'] is not None and target_ids is not None:
            decoder_logits = outputs['decoder_logits']
            # Shift targets for next-token prediction
            # decoder_logits: [B, tgt_len, vocab_size]
            # target_ids: [B, tgt_len]
            # We want to predict target_ids[1:] from decoder_logits[:-1]
            shift_logits = decoder_logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            text_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100  # Ignore padding
            )
            loss_dict['text_loss'] = text_loss.item()
            total_loss += text_loss_weight * text_loss

            # Compute perplexity
            perplexity = torch.exp(text_loss)
            loss_dict['perplexity'] = perplexity.item()

            # Compute token accuracy
            predictions = shift_logits.argmax(dim=-1)
            correct = (predictions == shift_labels).float()
            token_accuracy = correct.mean()
            loss_dict['token_accuracy'] = token_accuracy.item()

        # 2. Contrastive Loss (FIBER-style, for image-text alignment)
        if 'contrastive_features' in outputs and text_features is not None and image_features is not None:
            # Compute contrastive loss
            # Normalize features
            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)

            # Compute similarity matrix
            similarity = torch.matmul(
                text_features, image_features.t()) / 0.1  # temperature

            # Labels: diagonal elements are positive pairs
            batch_size = similarity.size(0)
            labels = torch.arange(batch_size, device=similarity.device)

            # Contrastive loss (symmetric)
            contrastive_loss = (
                F.cross_entropy(similarity, labels) +
                F.cross_entropy(similarity.t(), labels)
            ) / 2.0

            loss_dict['contrastive_loss'] = contrastive_loss.item()
            total_loss += contrastive_loss_weight * contrastive_loss

        # 3. Memory KL Divergence Loss (Larimar-style, for memory regularization)
        # This would be computed if episodic memory returns KL divergence
        # For now, placeholder - will be implemented when enhancing larima_memory.py
        if 'memory_kl' in outputs:
            memory_kl_loss = outputs['memory_kl']
            loss_dict['memory_kl_loss'] = memory_kl_loss.item()
            total_loss += memory_kl_weight * memory_kl_loss

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    # NOTE: _forward_reasoning_with_analysis() is now deprecated
    # Reasoning traces are now handled directly in ReasoningModule.forward()
    # with return_reasoning_trace=True parameter

    def export_for_embedded(self, output_path: str):
        """Export model for embedded deployment with quantization"""
        try:
            # Set model to evaluation mode for quantization
            self.eval()

            # Create embedded-optimized state dict
            embedded_state = {}

            with torch.no_grad():
                for name, param in self.named_parameters():
                    if 'BitNetLinear' in str(type(param)):
                        # Quantize BitNet layers
                        quantized_weight, scale = self._quantize_for_embedded(
                            param)
                        embedded_state[name] = quantized_weight
                        embedded_state[f"{name}_scale"] = scale
                    else:
                        # Keep other parameters in reduced precision
                        embedded_state[name] = param.half()

            # Add model configuration
            embedded_state['config'] = {
                'embed_dim': self.config.embed_dim,
                'num_layers': self.config.num_layers,
                'vocab_size': self.config.vocab_size,
                'max_seq_len': self.config.max_seq_len
            }

            # Save embedded model
            torch.save(embedded_state, output_path)
            print(f"✅ Exported embedded model to: {output_path}")

        except Exception as e:
            print(f"⚠️ Warning: Could not export embedded model: {e}")
            # Create a minimal fallback export
            torch.save({'config': self.config.__dict__}, output_path)

    def _quantize_for_embedded(self, weight: torch.Tensor):
        """Quantize weights to 1.58-bit format for embedded deployment"""
        scale = weight.abs().mean()
        quantized = torch.sign(weight) * (weight.abs() > 0.5 * scale).float()
        return quantized.half(), scale.half()

    def get_memory_usage(self):
        """Get model memory usage for embedded optimization"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        # Estimate memory in MB (assuming float32)
        memory_mb = total_params * 4 / (1024 * 1024)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_mb': memory_mb,
            'quantized_memory_mb': memory_mb * 0.2  # ~1.58 bits per parameter
        }

# Factory function for different model sizes with consistent vocab


def create_bitgen_model(size='tiny'):
    """Create BitGen model optimized for different embedded systems with consistent vocabulary"""

    if size == 'nano':
        # Ultra-tiny for smallest microcontrollers
        config = BitGenConfig(
            embed_dim=128,
            num_layers=4,
            num_heads=8,
            head_dim=16,
            ffn_dim=256,
            vocab_size=8192,  # Consistent with tiny tokenizer
            memory_size=32,
            max_seq_len=256
        )
    elif size == 'tiny':
        # Default configuration - FIXED vocabulary size
        config = BitGenConfig(
            embed_dim=256,
            num_layers=6,
            num_heads=8,
            head_dim=32,
            ffn_dim=512,
            vocab_size=16384,  # CRITICAL: Match this exactly with CLI config
            memory_size=64,
            max_seq_len=1024
        )
    elif size == 'small':
        # Larger model for more capable embedded systems
        config = BitGenConfig(
            embed_dim=512,
            num_layers=8,
            num_heads=16,
            head_dim=32,
            ffn_dim=1024,
            vocab_size=32768,  # CRITICAL: Match this exactly with CLI config
            memory_size=128,
            max_seq_len=2048
        )
    else:
        raise ValueError(f"Unknown model size: {size}")

    return BitGenModel(config)
