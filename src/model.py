"""
BitMar Model Architecture
BitNet-quantized Vision-Language Episodic Memory Transformer
Combines 1.58-bit quantization, DiNOv2 vision, and Larimar episodic memory
NOW WITH FIBER-STYLE BACKBONE FUSION for superior cross-modal understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer
import math
import logging

# Import FIBER fusion
from .fiber_fusion import create_fiber_fusion, FIBERIntegration
from .robot_reasoning import RobotReasoningIntegration, create_robot_reasoning_integration

logger = logging.getLogger(__name__)


class BitNetLinear(nn.Module):
    """1.58-bit Linear layer following BitNet b1.58 architecture"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (full precision for training)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Quantization scaling factors
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('input_scale', torch.ones(1))

    def quantize_weights(self):
        """Method for diagnostic detection"""
        return self.quantize_weights_1_58_bit(self.weight)

    def quantize_weights_1_58_bit(self, weight: torch.Tensor) -> torch.Tensor:
        """BitNet b1.58 weight quantization: {-1, 0, +1}"""
        # Compute scaling factor with numerical stability
        scale = weight.abs().mean()
        self.weight_scale.data = scale.clamp(min=1e-5, max=1e3)  # Prevent extreme scales

        # Normalize weights with gradient clipping
        weight_norm = torch.clamp(weight / self.weight_scale, min=-10.0, max=10.0)

        # 1.58-bit quantization with threshold
        threshold = 2.0 / 3.0  # Optimal threshold for ternary quantization

        # Create ternary weights
        quantized = torch.zeros_like(weight_norm)
        quantized[weight_norm > threshold] = 1.0
        quantized[weight_norm < -threshold] = -1.0
        # Values between -threshold and threshold remain 0

        return quantized

    def quantize_activations_8bit(self, x: torch.Tensor) -> torch.Tensor:
        """8-bit activation quantization with numerical stability"""
        # Clamp extreme values to prevent overflow
        x_clamped = torch.clamp(x, min=-1e6, max=1e6)

        # Compute quantization parameters
        x_min, x_max = x_clamped.min(), x_clamped.max()

        # Prevent division by zero
        range_val = x_max - x_min
        if range_val < 1e-8:
            return x_clamped

        scale = range_val / 255.0
        self.input_scale.data = scale.clamp(min=1e-8, max=1e3)

        # Quantize to 8-bit
        zero_point = (-x_min / scale).round().clamp(0, 255)
        quantized = ((x_clamped / scale) + zero_point).round().clamp(0, 255)

        # Dequantize
        dequantized = scale * (quantized - zero_point)
        return dequantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Full precision training with straight-through estimator
            # Forward pass with quantized weights but gradients flow through original weights
            weight_q = self.quantize_weights_1_58_bit(self.weight)
            weight_forward = weight_q * self.weight_scale

            # Use original weight for gradient computation
            weight_forward = weight_forward + \
                (self.weight - self.weight.detach())

            return F.linear(x, weight_forward, self.bias)
        else:
            # Inference with full quantization
            weight_q = self.quantize_weights_1_58_bit(
                self.weight) * self.weight_scale
            x_q = self.quantize_activations_8bit(x)
            return F.linear(x_q, weight_q, self.bias)


class BitNetMLP(nn.Module):
    """BitNet MLP block with 1.58-bit quantization"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = BitNetLinear(dim, hidden_dim)
        self.fc2 = BitNetLinear(hidden_dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class BitNetAttention(nn.Module):
    """Multi-head attention with BitNet quantization"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # BitNet quantized projections
        self.q_proj = BitNetLinear(dim, dim, bias=bias)
        self.k_proj = BitNetLinear(dim, dim, bias=bias)
        self.v_proj = BitNetLinear(dim, dim, bias=bias)
        self.out_proj = BitNetLinear(dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.shape[:2]

        # Validate input dimensions
        if query.size(-1) != self.dim:
            raise ValueError(f"Query dimension {query.size(-1)} doesn't match expected {self.dim}")
        if key.size(-1) != self.dim:
            raise ValueError(f"Key dimension {key.size(-1)} doesn't match expected {self.dim}")
        if value.size(-1) != self.dim:
            raise ValueError(f"Value dimension {value.size(-1)} doesn't match expected {self.dim}")

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Get key/value sequence length (handle different shapes)
        key_seq_len = key.size(1)
        
        # Reshape for multi-head attention with proper dimension checking
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Debug: BitNet attention - input shapes: query={query.shape}, key={key.shape}, value={value.shape}
        # Debug: BitNet attention - attention_scores shape: {attention_scores.shape}
        # Debug: BitNet attention - mask shape: {mask.shape if mask is not None else None}

        # Apply attention mask properly
        if mask is not None:
            # Debug: Applying attention mask
            # Ensure mask is on same device as attention_scores
            mask = mask.to(attention_scores.device)
            
            # Handle different mask shapes
            if mask.dim() == 3:  # [batch_size, 1, seq_len] or [batch_size, seq_len, seq_len]
                if mask.shape[1] == 1:
                    # Broadcast [batch_size, 1, seq_len] to [batch_size, 1, 1, seq_len]
                    mask = mask.unsqueeze(1)
                elif mask.shape[1] == mask.shape[2]:
                    # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                    mask = mask.unsqueeze(1)
            elif mask.dim() == 4:  # Already [batch_size, 1, seq_len, seq_len] or similar
                pass
            else:
                # Debug: Unexpected mask dimensions: {mask.shape}
                pass
            
            # Apply mask by setting masked positions to large negative value
            mask = mask.to(attention_scores.dtype)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            # Debug: Mask applied successfully

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # Debug: BitNet attention - attention_weights shape: {attention_weights.shape}

        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        # Debug: BitNet attention - attended shape: {attended.shape}

        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        output = self.out_proj(attended)
        
        # Debug: BitNet attention - final output shape: {output.shape}
        return output, attention_weights.mean(dim=1)  # Average across heads


class BitNetTransformerBlock(nn.Module):
    """BitNet Transformer block with quantized components"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = BitNetAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = BitNetMLP(dim, int(dim * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Debug: BitNet transformer block - input shape: {x.shape}
        # Debug: BitNet transformer block - mask shape: {mask.shape if mask is not None else None}
        
        # Self-attention with residual connection
        normed_x = self.norm1(x)
        # Debug: BitNet transformer block - normed_x shape: {normed_x.shape}
        
        try:
            attn_out, attn_weights = self.attn(normed_x, normed_x, normed_x, mask)
            # Debug: BitNet transformer block - attention output shape: {attn_out.shape}
        except Exception as e:
            # Debug: BitNet transformer block attention failed: {e}
            # Input shapes: query={normed_x.shape}, key={normed_x.shape}, value={normed_x.shape}
            # Mask shape: {mask.shape if mask is not None else None}
            raise e
            
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        # Debug: BitNet transformer block - final output shape: {x.shape}

        return x, attn_weights


class BitNetTextEncoder(nn.Module):
    """BitNet-based text encoder"""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embeddings (kept full precision)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # BitNet transformer layers
        self.layers = nn.ModuleList([
            BitNetTransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        # Debug: Text encoder - batch_size: {batch_size}, seq_len: {seq_len}

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + \
            self.position_embedding(positions)
        x = self.dropout(x)
        # Debug: Text encoder - embeddings shape: {x.shape}

        # Transform through BitNet layers
        attention_patterns = []
        for i, layer in enumerate(self.layers):
            # Debug: Text encoder layer {i} - input shape: {x.shape}
            
            # Convert attention mask to the right format for the layer
            layer_mask = None
            if attention_mask is not None:
                # Debug: Text encoder layer {i} - attention_mask shape: {attention_mask.shape}
                # Create a mask where 1 means attend, 0 means don't attend
                layer_mask = attention_mask.unsqueeze(
                    1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
                # Debug removed

            try:
                x, attn_weights = layer(x, layer_mask)
                # Debug removed
                attention_patterns.append(attn_weights)
            except Exception as e:
                # Debug removed
                print(f"   Input shape: {x.shape}")
                print(f"   Mask shape: {layer_mask.shape if layer_mask is not None else None}")
                raise e

        x = self.norm(x)
        # Debug removed
        return x, attention_patterns


class BitNetTextDecoder(nn.Module):
    """BitNet-based text decoder with causal masking"""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # BitNet transformer layers
        self.layers = nn.ModuleList([
            BitNetTransformerBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Output projection to vocabulary
        self.lm_head = BitNetLinear(dim, vocab_size, bias=False)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        # Register causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)
                       ).unsqueeze(0).unsqueeze(0)
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(
                seq_len, device=input_ids.device).unsqueeze(0)
            x = self.token_embedding(input_ids) + \
                self.position_embedding(positions)
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            positions = torch.arange(
                seq_len, device=inputs_embeds.device).unsqueeze(0)
            x = inputs_embeds + self.position_embedding(positions)
        else:
            raise ValueError(
                "Either input_ids or inputs_embeds must be provided")

        x = self.dropout(x)

        # Create causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        if attention_mask is not None:
            # Ensure attention_mask is properly shaped [batch_size, seq_len]
            if attention_mask.dim() != 2:
                raise ValueError(f"attention_mask should be 2D, got {attention_mask.dim()}D: {attention_mask.shape}")
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(f"attention_mask shape {attention_mask.shape} doesn't match input shape ({batch_size}, {seq_len})")
                
            # Expand attention mask to 4D: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            expanded_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Combine with causal mask: both should be [batch_size, 1, seq_len, seq_len]
            # Expand attention mask to match causal mask shape
            expanded_attention_mask = expanded_attention_mask.expand(batch_size, 1, seq_len, seq_len)
            
            # Element-wise multiplication
            mask = expanded_attention_mask * causal_mask
        else:
            mask = causal_mask

        # Transform through BitNet layers
        attention_patterns = []
        hidden_states = []
        
        for layer in self.layers:
            if output_hidden_states:
                hidden_states.append(x)
            x, attn_weights = layer(x, mask)
            if output_attentions:
                attention_patterns.append(attn_weights)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        result = {
            'logits': logits,
            'loss': loss,
        }
        
        if output_attentions:
            result['attentions'] = attention_patterns
            result['attention_patterns'] = attention_patterns  # Keep backward compatibility
            
        if output_hidden_states:
            hidden_states.append(x)  # Add final hidden state
            result['hidden_states'] = hidden_states

        return result


class EpisodicMemory(nn.Module):
    """Episodic Memory mechanism inspired by Larimar with performance optimizations and external storage support"""

    def __init__(
        self,
        memory_size: int,
        episode_dim: int,
        alpha: float = 0.1,
        direct_writing: bool = True,
        observation_noise_std: float = 1e-6,
        external_storage: bool = False,
        memory_storage_path: str = None,
        compression_enabled: bool = True,
        lazy_loading: bool = False
    ):
        super().__init__()
        self.memory_size = memory_size
        self.episode_dim = episode_dim
        self.alpha = alpha
        self.direct_writing = direct_writing
        self.observation_noise_std = observation_noise_std

        # External storage configuration
        self.external_storage = external_storage
        self.memory_storage_path = memory_storage_path
        self.compression_enabled = compression_enabled
        self.lazy_loading = lazy_loading
        self._memory_loaded = False
        self._memory_version = 1

        # Memory storage with improved initialization
        if external_storage and lazy_loading:
            # For lazy loading, we'll initialize empty and load when needed
            self._memory_data = None
            self._metadata = None
        else:
            # Standard initialization for compatibility
            self.register_buffer('memory', torch.randn(memory_size, episode_dim) * 0.02)
            self.register_buffer('memory_age', torch.zeros(memory_size))
            self.register_buffer('memory_usage', torch.zeros(memory_size))

        # Always initialize these for proper functioning
        self.register_buffer('memory_quality', torch.zeros(memory_size))
        self.register_buffer('memory_importance', torch.ones(memory_size))
        self.register_buffer('memory_mean', torch.zeros(episode_dim))
        self.register_buffer('memory_std', torch.ones(episode_dim))
        self.register_buffer('update_count', torch.tensor(0))

        # Enhanced memory access networks with residual connections
        self.query_net = nn.Sequential(
            BitNetLinear(episode_dim, episode_dim),
            nn.LayerNorm(episode_dim),
            nn.GELU(),
            BitNetLinear(episode_dim, episode_dim)
        )
        self.key_net = nn.Sequential(
            BitNetLinear(episode_dim, episode_dim),
            nn.LayerNorm(episode_dim),
            nn.GELU(),
            BitNetLinear(episode_dim, episode_dim)
        )
        self.value_net = nn.Sequential(
            BitNetLinear(episode_dim, episode_dim),
            nn.LayerNorm(episode_dim),
            nn.GELU(),
            BitNetLinear(episode_dim, episode_dim)
        )

        # Add temperature parameter for attention sharpening
        self.register_parameter('attention_temperature', nn.Parameter(torch.tensor(1.0)))

        # Memory consolidation network for better episode encoding
        self.consolidation_net = nn.Sequential(
            BitNetLinear(episode_dim, episode_dim * 2),
            nn.LayerNorm(episode_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            BitNetLinear(episode_dim * 2, episode_dim),
            nn.LayerNorm(episode_dim)
        )

    def _ensure_memory_loaded(self):
        """Ensure memory is loaded into device memory"""
        if self.external_storage and self.lazy_loading and not self._memory_loaded:
            self.load_external_memory()
        elif not hasattr(self, 'memory'):
            # Initialize if not present (compatibility mode)
            self.register_buffer('memory', torch.randn(self.memory_size, self.episode_dim) * 0.02)
            self.register_buffer('memory_age', torch.zeros(self.memory_size))
            self.register_buffer('memory_usage', torch.zeros(self.memory_size))

    def save_external_memory(self, path: str = None, compress: bool = None) -> str:
        """Save episodic memory to external storage"""
        import os
        import json
        from pathlib import Path

        # Use provided path or default
        save_path = path or self.memory_storage_path or "episodic_memory.pt"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Use provided compression setting or default
        use_compression = compress if compress is not None else self.compression_enabled

        # Prepare memory data
        memory_data = {
            'memory': self.memory.cpu() if hasattr(self, 'memory') else torch.randn(self.memory_size, self.episode_dim) * 0.02,
            'memory_age': self.memory_age.cpu() if hasattr(self, 'memory_age') else torch.zeros(self.memory_size),
            'memory_usage': self.memory_usage.cpu() if hasattr(self, 'memory_usage') else torch.zeros(self.memory_size),
            'memory_quality': self.memory_quality.cpu(),
            'memory_importance': self.memory_importance.cpu(),
            'memory_mean': self.memory_mean.cpu(),
            'memory_std': self.memory_std.cpu(),
            'update_count': self.update_count.cpu(),
            'version': self._memory_version,
            'metadata': {
                'memory_size': self.memory_size,
                'episode_dim': self.episode_dim,
                'alpha': self.alpha,
                'creation_timestamp': torch.tensor(time.time()),
                'compression_enabled': use_compression
            }
        }

        # Apply compression if enabled
        if use_compression:
            # Quantize memory to reduce storage size
            memory_data['memory'] = self._compress_memory_tensor(memory_data['memory'])
            memory_data['compressed'] = True
        else:
            memory_data['compressed'] = False

        # Save to file
        torch.save(memory_data, save_path)

        # Also save metadata separately for quick access
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'memory_size': self.memory_size,
                'episode_dim': self.episode_dim,
                'version': self._memory_version,
                'compressed': use_compression,
                'file_size_mb': save_path.stat().st_size / (1024 * 1024),
                'creation_timestamp': time.time()
            }, f, indent=2)

        logger.info(f"💾 Episodic memory saved to: {save_path}")
        logger.info(f"📊 Memory size: {save_path.stat().st_size / 1024:.1f} KB")

        return str(save_path)

    def load_external_memory(self, path: str = None, device: str = None) -> bool:
        """Load episodic memory from external storage"""
        import json
        from pathlib import Path

        # Use provided path or default
        load_path = path or self.memory_storage_path or "episodic_memory.pt"
        load_path = Path(load_path)

        if not load_path.exists():
            logger.warning(f"⚠️ External memory file not found: {load_path}")
            return False

        try:
            # Load memory data
            memory_data = torch.load(load_path, map_location='cpu')

            # Validate compatibility
            if memory_data['metadata']['memory_size'] != self.memory_size:
                logger.error(f"❌ Memory size mismatch: expected {self.memory_size}, got {memory_data['metadata']['memory_size']}")
                return False

            if memory_data['metadata']['episode_dim'] != self.episode_dim:
                logger.error(f"❌ Episode dimension mismatch: expected {self.episode_dim}, got {memory_data['metadata']['episode_dim']}")
                return False

            # Set device
            device = device or next(self.parameters()).device

            # Decompress if needed
            if memory_data.get('compressed', False):
                memory_tensor = self._decompress_memory_tensor(memory_data['memory'])
            else:
                memory_tensor = memory_data['memory']

            # Load memory tensors
            if hasattr(self, 'memory'):
                self.memory.copy_(memory_tensor.to(device))
                self.memory_age.copy_(memory_data['memory_age'].to(device))
                self.memory_usage.copy_(memory_data['memory_usage'].to(device))
            else:
                # Register buffers if not present (lazy loading case)
                self.register_buffer('memory', memory_tensor.to(device))
                self.register_buffer('memory_age', memory_data['memory_age'].to(device))
                self.register_buffer('memory_usage', memory_data['memory_usage'].to(device))

            self.memory_quality.copy_(memory_data['memory_quality'].to(device))
            self.memory_importance.copy_(memory_data['memory_importance'].to(device))
            self.memory_mean.copy_(memory_data['memory_mean'].to(device))
            self.memory_std.copy_(memory_data['memory_std'].to(device))
            self.update_count.copy_(memory_data['update_count'].to(device))

            self._memory_version = memory_data.get('version', 1)
            self._memory_loaded = True

            logger.info(f"✅ Episodic memory loaded from: {load_path}")
            logger.info(f"📊 Memory version: {self._memory_version}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load external memory: {e}")
            return False

    def _compress_memory_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress memory tensor for storage"""
        # Quantize to int8 to reduce storage size
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        # Avoid division by zero
        tensor_range = tensor_max - tensor_min
        if tensor_range < 1e-8:
            return tensor

        # Quantize to int8 range
        quantized = ((tensor - tensor_min) / tensor_range * 255).round().clamp(0, 255).to(torch.uint8)

        # Store quantization parameters
        return {
            'data': quantized,
            'min': tensor_min,
            'max': tensor_max,
            'original_shape': tensor.shape
        }

    def _decompress_memory_tensor(self, compressed_data) -> torch.Tensor:
        """Decompress memory tensor"""
        if isinstance(compressed_data, dict):
            quantized = compressed_data['data'].float()
            tensor_min = compressed_data['min']
            tensor_max = compressed_data['max']

            # Dequantize
            tensor_range = tensor_max - tensor_min
            dequantized = (quantized / 255.0) * tensor_range + tensor_min

            return dequantized.view(compressed_data['original_shape'])
        else:
            # Not compressed, return as-is
            return compressed_data

    def _update_memory_statistics(self, episodes: torch.Tensor):
        """Update running statistics for memory normalization"""
        with torch.no_grad():
            batch_mean = episodes.mean(dim=0)
            batch_var = episodes.var(dim=0, unbiased=False)

            # Exponential moving average
            momentum = 0.1
            self.memory_mean = (1 - momentum) * self.memory_mean + momentum * batch_mean
            self.memory_std = torch.sqrt((1 - momentum) * self.memory_std**2 + momentum * batch_var)
            self.update_count += 1

    def _normalize_episodes(self, episodes: torch.Tensor) -> torch.Tensor:
        """Normalize episodes using running statistics"""
        if self.update_count > 10:  # Only normalize after some updates
            return (episodes - self.memory_mean) / (self.memory_std + 1e-8)
        return episodes

    def _compute_episode_quality(self, episode: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        """Compute quality score for memory episodes"""
        # Quality based on diversity and relevance
        # episode: [batch_size, episode_dim], memory: [memory_size, episode_dim]
        # We want to compute similarity between each episode and all memory slots
        
        batch_size = episode.size(0)
        
        # Expand dimensions for broadcasting: [batch_size, 1, episode_dim] vs [1, memory_size, episode_dim]
        episode_expanded = episode.unsqueeze(1)  # [batch_size, 1, episode_dim]
        memory_expanded = self.memory.unsqueeze(0)  # [1, memory_size, episode_dim]
        
        # Compute cosine similarity across all combinations
        similarity_to_memory = torch.cosine_similarity(
            episode_expanded, memory_expanded, dim=-1  # [batch_size, memory_size]
        )
        
        # Get maximum similarity for each episode (most similar memory slot)
        max_similarity = similarity_to_memory.max(dim=-1)[0]  # [batch_size]

        # Encourage diversity - lower similarity = higher quality
        diversity_score = 1.0 - max_similarity

        # Relevance score based on retrieval quality
        retrieval_quality = torch.cosine_similarity(episode, retrieved, dim=-1)

        # Combined quality score
        return 0.7 * diversity_score + 0.3 * retrieval_quality

    def write_memory(self, episode: torch.Tensor) -> torch.Tensor:
        """Optimized memory writing with intelligent slot selection"""
        batch_size = episode.size(0)

        # Apply consolidation to improve episode representation
        consolidated_episode = self.consolidation_net(episode) + episode  # Residual connection

        # Update statistics
        self._update_memory_statistics(consolidated_episode)

        # Normalize episodes
        normalized_episode = self._normalize_episodes(consolidated_episode)

        if self.direct_writing:
            # Enhanced slot selection combining age, usage, and quality
            if batch_size <= self.memory_size:
                # Compute composite scores for slot selection
                age_scores = -self.memory_age  # Prefer older slots
                usage_scores = -self.memory_usage  # Prefer less used slots
                quality_scores = -self.memory_quality  # Prefer lower quality slots
                importance_scores = -self.memory_importance  # Prefer less important slots

                # Weighted combination
                composite_scores = (
                    0.4 * age_scores +
                    0.3 * usage_scores +
                    0.2 * quality_scores +
                    0.1 * importance_scores
                )

                _, best_indices = composite_scores.topk(batch_size, largest=True)

                # Update memory slots with momentum-based updates
                momentum = self.alpha
                self.memory[best_indices] = (
                    (1 - momentum) * self.memory[best_indices] +
                    momentum * normalized_episode.detach()
                )

                # Update metadata
                self.memory_age[best_indices] = self.memory_age.max() + 1
                self.memory_usage[best_indices] += 1

                # Update quality scores (will be computed during read)
                with torch.no_grad():
                    # Temporary quality estimation based on internal consistency
                    temp_quality = torch.norm(normalized_episode, dim=-1)
                    self.memory_quality[best_indices] = temp_quality.detach()

            else:
                # Handle large batches efficiently
                for i in range(0, batch_size, self.memory_size):
                    end_idx = min(i + self.memory_size, batch_size)
                    chunk_size = end_idx - i

                    # Apply same logic for chunks
                    age_scores = -self.memory_age
                    usage_scores = -self.memory_usage
                    quality_scores = -self.memory_quality
                    importance_scores = -self.memory_importance

                    composite_scores = (
                        0.4 * age_scores +
                        0.3 * usage_scores +
                        0.2 * quality_scores +
                        0.1 * importance_scores
                    )

                    _, chunk_indices = composite_scores.topk(chunk_size, largest=True)

                    momentum = self.alpha
                    self.memory[chunk_indices] = (
                        (1 - momentum) * self.memory[chunk_indices] +
                        momentum * normalized_episode[i:end_idx].detach()
                    )

                    self.memory_age[chunk_indices] = self.memory_age.max() + 1 + i
                    self.memory_usage[chunk_indices] += 1

                    temp_quality = torch.norm(normalized_episode[i:end_idx], dim=-1)
                    self.memory_quality[chunk_indices] = temp_quality.detach()

        return consolidated_episode

    def read_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized memory reading with enhanced attention"""
        batch_size = query.size(0)

        # Validate query dimensions
        if query.size(-1) != self.episode_dim:
            raise ValueError(f"Query dimension {query.size(-1)} doesn't match memory episode_dim {self.episode_dim}")

        # Normalize query
        normalized_query = self._normalize_episodes(query)

        # Enhanced query, key, value computation with residual connections
        q = self.query_net(normalized_query) + normalized_query  # Residual
        k = self.key_net(self.memory) + self.memory  # Residual
        v = self.value_net(self.memory) + self.memory  # Residual

        # Scaled dot-product attention with learnable temperature
        attention_scores = torch.matmul(q, k.transpose(0, 1)) / (
            math.sqrt(self.episode_dim) * self.attention_temperature.clamp(min=0.1, max=10.0)
        )

        # Add importance weighting to attention scores
        importance_weights = self.memory_importance.unsqueeze(0).expand(batch_size, -1)
        attention_scores = attention_scores + torch.log(importance_weights + 1e-8)

        # Apply attention with improved stability
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Add attention dropout for regularization during training
        if self.training:
            attention_weights = F.dropout(attention_weights, p=0.1)

        # Weighted memory retrieval
        retrieved = torch.matmul(attention_weights, v)

        # Update memory access statistics and importance
        with torch.no_grad():
            access_counts = attention_weights.sum(0)
            self.memory_usage += access_counts

            # Update importance based on usage frequency
            self.memory_importance = 0.9 * self.memory_importance + 0.1 * (access_counts + 1e-8)

            # Update quality scores based on retrieval effectiveness
            if hasattr(self, '_last_query_quality'):
                quality_update = self._compute_episode_quality(query, retrieved)
                # Update quality for attended slots
                attended_indices = attention_weights.max(0)[1]  # Most attended slots
                self.memory_quality[attended_indices] = (
                    0.8 * self.memory_quality[attended_indices] +
                    0.2 * quality_update.mean()
                )

        return retrieved, attention_weights

    def forward(self, episode: torch.Tensor, mode: str = "read_write") -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward pass with memory consolidation"""
        if mode == "write":
            return self.write_memory(episode), None
        elif mode == "read":
            return self.read_memory(episode)
        else:  # read_write
            # Write episode to memory with consolidation
            consolidated_episode = self.write_memory(episode)

            # Read from memory using consolidated episode as query
            retrieved, attention_weights = self.read_memory(consolidated_episode)

            # Memory-augmented output combining input and retrieved memory
            output = 0.7 * consolidated_episode + 0.3 * retrieved

            return output, attention_weights

    def get_memory_statistics(self) -> Dict[str, torch.Tensor]:
        """Get comprehensive memory statistics for monitoring"""
        return {
            'memory_usage_distribution': self.memory_usage,
            'memory_age_distribution': self.memory_age,
            'memory_quality_scores': self.memory_quality,
            'memory_importance': self.memory_importance,
            'attention_temperature': self.attention_temperature,
            'memory_utilization': (self.memory_usage > 0).float().mean(),
            'memory_diversity': torch.std(self.memory, dim=0).mean(),
            'update_count': self.update_count
        }

    def consolidate_memory(self):
        """Explicit memory consolidation for improved organization"""
        with torch.no_grad():
            # Sort memory by importance and quality
            importance_quality_score = 0.6 * self.memory_importance + 0.4 * self.memory_quality
            sorted_indices = torch.argsort(importance_quality_score, descending=True)

            # Reorganize memory to group similar episodes
            sorted_memory = self.memory[sorted_indices]
            self.memory.copy_(sorted_memory)

            # Update corresponding metadata
            self.memory_age[:] = self.memory_age[sorted_indices]
            self.memory_usage[:] = self.memory_usage[sorted_indices]
            self.memory_quality[:] = self.memory_quality[sorted_indices]
            self.memory_importance[:] = self.memory_importance[sorted_indices]

    def get_memory_info(self) -> Dict:
        """Get comprehensive memory information"""
        info = {
            'memory_size': self.memory_size,
            'episode_dim': self.episode_dim,
            'external_storage': self.external_storage,
            'compression_enabled': self.compression_enabled,
            'lazy_loading': self.lazy_loading,
            'memory_loaded': self._memory_loaded if self.external_storage else True,
            'version': self._memory_version,
            'storage_path': self.memory_storage_path
        }

        if hasattr(self, 'memory'):
            info.update({
                'memory_utilization': (self.memory_usage > 0).float().mean().item(),
                'memory_diversity': torch.std(self.memory, dim=0).mean().item(),
                'update_count': self.update_count.item(),
                'memory_device': str(self.memory.device)
            })

        return info

    def create_memory_snapshot(self, snapshot_name: str = None) -> str:
        """Create a named snapshot of the current memory state"""
        import time
        from pathlib import Path

        timestamp = int(time.time())
        snapshot_name = snapshot_name or f"memory_snapshot_{timestamp}"

        # Create snapshots directory
        snapshots_dir = Path("memory_snapshots")
        snapshots_dir.mkdir(exist_ok=True)

        snapshot_path = snapshots_dir / f"{snapshot_name}.pt"

        # Save current memory state
        saved_path = self.save_external_memory(str(snapshot_path), compress=True)

        logger.info(f"📸 Memory snapshot created: {saved_path}")
        return saved_path

    def load_memory_snapshot(self, snapshot_name: str) -> bool:
        """Load a named memory snapshot"""
        from pathlib import Path

        snapshots_dir = Path("memory_snapshots")
        snapshot_path = snapshots_dir / f"{snapshot_name}.pt"

        if not snapshot_path.exists():
            logger.warning(f"⚠️ Snapshot not found: {snapshot_path}")
            return False

        success = self.load_external_memory(str(snapshot_path))
        if success:
            logger.info(f"📸 Memory snapshot loaded: {snapshot_name}")

        return success

    def enable_external_storage(self, storage_path: str = None, compress: bool = True, lazy: bool = False):
        """Enable external storage mode for edge deployment"""
        self.external_storage = True
        self.memory_storage_path = storage_path or "episodic_memory.pt"
        self.compression_enabled = compress
        self.lazy_loading = lazy

        logger.info(f"🔄 External storage enabled: {self.memory_storage_path}")
        logger.info(f"   Compression: {compress}, Lazy loading: {lazy}")

    def disable_external_storage(self):
        """Disable external storage and return to integrated mode"""
        # Ensure memory is loaded before disabling external storage
        self._ensure_memory_loaded()

        self.external_storage = False
        self.lazy_loading = False
        self._memory_loaded = True

        logger.info("🔄 External storage disabled, using integrated mode")

    # ...existing code for other methods...
class CrossModalFusion(nn.Module):
    """Cross-modal fusion module for text and vision features"""

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim

        # Projection layers
        self.text_proj = BitNetLinear(text_dim, hidden_dim)
        self.vision_proj = BitNetLinear(vision_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            BitNetAttention(
                dim=hidden_dim,
                num_heads=num_heads
            ) for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = BitNetLinear(hidden_dim, hidden_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            text_features: [batch_size, seq_len, text_dim]
            vision_features: [batch_size, vision_dim]

        Returns:
            fused_features: [batch_size, seq_len, hidden_dim]
            attention_weights: Dict of attention patterns
        """
        batch_size, seq_len = text_features.shape[:2]

        # Validate input dimensions
        if text_features.size(-1) != self.text_dim:
            raise ValueError(f"Text features dimension {text_features.size(-1)} doesn't match expected {self.text_dim}")
        if vision_features.size(-1) != self.vision_dim:
            raise ValueError(f"Vision features dimension {vision_features.size(-1)} doesn't match expected {self.vision_dim}")

        # Project to common dimension
        # [batch_size, seq_len, hidden_dim]
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Cross-attention fusion
        fused = text_proj
        attention_weights = {}

        for i, (attn_layer, norm_layer) in enumerate(zip(self.cross_attention_layers, self.layer_norms)):
            # Text-to-vision cross-attention
            attn_output, attn_weights = attn_layer(
                query=fused,
                key=vision_proj,
                value=vision_proj
            )

            # Residual connection and normalization
            fused = norm_layer(fused + attn_output)
            attention_weights[f'layer_{i}'] = attn_weights

        # Output projection
        output = self.output_proj(fused)

        return output, attention_weights


class VisionEncoder(nn.Module):
    """Quantized Vision Encoder for DiNOv2 features"""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2
    ):
        super().__init__()

        # Quantized layers
        self.layers = nn.ModuleList([
            BitNetLinear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Output projection
        self.output_proj = BitNetLinear(hidden_dim, output_dim)

        # Activation and normalization
        self.activation = nn.GELU()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, input_dim] - DiNOv2 features

        Returns:
            encoded_features: [batch_size, output_dim]
        """
        # Handle potential extra dimensions
        if vision_features.dim() > 2:
            # Flatten any extra dimensions except batch
            original_shape = vision_features.shape
            vision_features = vision_features.view(original_shape[0], -1)

            # Ensure we have the expected input dimension
            if vision_features.size(-1) != self.layers[0].in_features:
                # Take only the first input_dim features if we have more
                if vision_features.size(-1) > self.layers[0].in_features:
                    vision_features = vision_features[:, :self.layers[0].in_features]
                else:
                    raise ValueError(f"Vision features dimension {vision_features.size(-1)} is smaller than expected {self.layers[0].in_features}")

        x = vision_features

        for layer, norm in zip(self.layers, self.layer_norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        # Output projection
        output = self.output_proj(x)

        return output


class BitMarModel(nn.Module):
    """
    BitMar: BitNet-quantized Vision-Language Episodic Memory Transformer
    Combines 1.58-bit quantization, DiNOv2 vision features, and Larimar episodic memory
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # NEW: Episodic memory toggle for ablation studies
        self.use_episodic_memory = config.get('use_episodic_memory', True)
        logger.info(f"🧠 Episodic Memory: {'Enabled' if self.use_episodic_memory else 'Disabled (Ablation Study)'}")

        # Loss balancing parameters - use training config values
        training_config = config.get('training', {})
        self.cross_modal_loss_weight = training_config.get('cross_modal_loss_weight', 
                                                          config.get('cross_modal_loss_weight', 0.1))
        self.text_loss_weight = training_config.get('text_generation_loss_weight', 
                                                   config.get('text_loss_weight', 1.0))
        self.vision_loss_weight = config.get('vision_loss_weight', 0.1)
        self.memory_loss_weight = training_config.get('memory_regularization_weight', 
                                                    config.get('memory_loss_weight', 0.05)) if self.use_episodic_memory else 0.0

        # Dynamic loss scaling
        self.adaptive_loss_scaling = config.get('adaptive_loss_scaling', True)
        self.loss_scale_temperature = config.get('loss_scale_temperature', 0.07)

        # Encoder freezing parameters
        self.freeze_text_encoder_steps = config.get('freeze_text_encoder_steps', 0)
        self.freeze_vision_encoder_steps = config.get('freeze_vision_encoder_steps', 0)
        self.current_step = 0

        # BitNet text encoder/decoder
        self.text_encoder = BitNetTextEncoder(
            vocab_size=config['vocab_size'],
            dim=config['text_encoder_dim'],
            num_layers=config['text_encoder_layers'],
            num_heads=config['text_encoder_heads'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        )

        self.text_decoder = BitNetTextDecoder(
            vocab_size=config['vocab_size'],
            dim=config['text_decoder_dim'],
            num_layers=config['text_decoder_layers'],
            num_heads=config['text_decoder_heads'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        )

        # Vision processing with BitNet quantization
        self.vision_encoder = VisionEncoder(
            input_dim=config['vision_encoder_dim'],
            hidden_dim=config['vision_hidden_size'],
            output_dim=config['vision_latent_size']
        )

        # Cross-modal fusion with BitNet
        #self.fusion = CrossModalFusion(
        #    text_dim=config['text_encoder_dim'],
        #    vision_dim=config['vision_latent_size'],
        #    hidden_dim=config['fusion_hidden_size'],
        #    num_heads=config['fusion_num_heads'],
        #    num_layers=config['fusion_num_layers']
        #)
        logger.info("� Implementing AUTHENTIC FIBER backbone fusion for superior cross-modal understanding")
        logger.info("   • ITC (Image-Text Contrastive) loss with momentum queue")
        logger.info("   • ITM (Image-Text Matching) loss with hard negatives") 
        logger.info("   • MLM (Masked Language Modeling) integration")
        logger.info("   • Temperature-scaled contrastive learning")
        
        # Use authentic FIBER implementation that includes all original components
        self.fusion = create_fiber_fusion(
            text_encoder_dim=config['text_encoder_dim'],
            vision_encoder_dim=config['vision_latent_size'],
            fusion_hidden_size=config['fusion_hidden_size'],
            vocab_size=config.get('vocab_size', 50257),  # GPT-2 vocab size
            num_heads=config.get('fusion_num_heads', 8),
            num_layers=config.get('fusion_num_layers', 2),
            num_fusion_layers=config.get('num_fiber_fusion_layers', 6),
            dropout=config.get('dropout', 0.1),
            config=config  # Pass full config for FIBER-specific parameters
        )

        # Log FIBER configuration details
        logger.info("✅ FIBER Configuration:")
        logger.info(f"  • Fusion layers: {config.get('num_fiber_fusion_layers', 6)}")
        logger.info(f"  • Text fusion layers: {config.get('fiber_text_fusion_layers', 4)}")
        logger.info(f"  • Vision fusion layers: {config.get('fiber_vision_fusion_layers', 2)}")
        logger.info(f"  • Fusion strategy: {config.get('fiber_fusion_strategy', 'deep_backbone')}")
        logger.info(f"  • Learnable alpha: {config.get('fiber_learnable_alpha', True)}")
        logger.info(f"  • Bidirectional fusion: {config.get('fiber_bidirectional_fusion', True)}")
        logger.info(f"  • Backbone integration: {config.get('fiber_backbone_integration', True)}")
        logger.info(f"  • Attention temperature: {config.get('fiber_attention_temperature', 1.0)}")
        logger.info(f"  • Cross-attention dropout: {config.get('fiber_cross_attention_dropout', 0.1)}")
        logger.info(f"  • Gradient checkpointing: {config.get('fiber_enable_gradient_checkpointing', True)}")

        # Store FIBER config for downstream use
        self.fiber_config = {
            'num_fusion_layers': config.get('num_fiber_fusion_layers', 6),
            'text_fusion_layers': config.get('fiber_text_fusion_layers', 4),
            'vision_fusion_layers': config.get('fiber_vision_fusion_layers', 2),
            'fusion_strategy': config.get('fiber_fusion_strategy', 'deep_backbone'),
            'learnable_alpha': config.get('fiber_learnable_alpha', True),
            'bidirectional_fusion': config.get('fiber_bidirectional_fusion', True),
            'backbone_integration': config.get('fiber_backbone_integration', True),
            'attention_temperature': config.get('fiber_attention_temperature', 1.0),
            'cross_attention_dropout': config.get('fiber_cross_attention_dropout', 0.1),
            'enable_gradient_checkpointing': config.get('fiber_enable_gradient_checkpointing', True),
            'residual_connection': config.get('fiber_residual_connection', True),
            'layer_norm_eps': config.get('fiber_layer_norm_eps', 1e-6),
            'mlp_ratio': config.get('fiber_mlp_ratio', 4.0)
        }

        # Enhanced Episodic memory with BitNet quantization (OPTIONAL for ablation study)
        if self.use_episodic_memory:
            from .enhanced_episodic_memory import create_enhanced_episodic_memory, EpisodicMemoryConfig
            
            # Create enhanced memory configuration
            memory_config = EpisodicMemoryConfig(
                memory_size=config['memory_size'],
                episode_dim=config['episode_dim'],
                alpha=config['memory_alpha'],
                direct_writing=config['direct_writing'],
                external_storage=config.get('external_memory_storage', True),
                memory_storage_path=config.get('memory_storage_path', './episodic_memory'),
                compression_enabled=config.get('memory_compression', True),
                lazy_loading=config.get('memory_lazy_loading', True),
                cross_modal_fusion=config.get('memory_cross_modal_fusion', True),
                vision_memory_weight=config.get('vision_memory_weight', 0.3),
                text_memory_weight=config.get('text_memory_weight', 0.7),
                max_memory_age=config.get('max_memory_age', 10000),
                memory_consolidation_threshold=config.get('memory_consolidation_threshold', 0.8),
                async_save=config.get('memory_async_save', True)
            )
            
            self.memory = create_enhanced_episodic_memory(memory_config.__dict__)
            logger.info(f"🧠 Enhanced Episodic Memory initialized (Larimar-inspired):")
            logger.info(f"  • Memory size: {config['memory_size']}")
            logger.info(f"  • Episode dimension: {config['episode_dim']}")
            logger.info(f"  • Alpha: {config['memory_alpha']}")
            logger.info(f"  • External storage: {config.get('external_memory_storage', True)}")
            logger.info(f"  • Cross-modal fusion: {config.get('memory_cross_modal_fusion', True)}")
            logger.info(f"  • Lazy loading: {config.get('memory_lazy_loading', True)}")

            # Memory projection layers (only if memory is enabled)
            self.text_to_episode = BitNetLinear(
                config['text_encoder_dim'],
                config['episode_dim']
            )

            self.vision_to_episode = BitNetLinear(
                config['vision_latent_size'],
                config['episode_dim']
            )

            self.memory_to_decoder = BitNetLinear(
                config['episode_dim'],
                config['text_decoder_dim']
            )

            # Decoder input projection (with memory)
            self.decoder_input_proj = BitNetLinear(
                config['fusion_hidden_size'],
                config['text_decoder_dim']
            )

            logger.info(f"✅ Episodic Memory initialized: {config['memory_size']} slots, {config['episode_dim']} dimensions")
        else:
            self.memory = None
            logger.info("🧠 Enhanced Episodic Memory disabled for ablation study")
            self.text_to_episode = None
            self.vision_to_episode = None
            self.memory_to_decoder = None
            self.decoder_input_proj = None

            # Alternative fusion for non-memory model (direct path to decoder)
            self.direct_fusion_proj = BitNetLinear(
                config['fusion_hidden_size'],
                config['text_decoder_dim']
            )

            logger.info("🚫 Episodic Memory disabled for ablation study")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # NEW: Robot Reasoning Integration (similar to deepseek-r1's structured reasoning)
        self.enable_robot_reasoning = config.get('enable_robot_reasoning', False)
        if self.enable_robot_reasoning:
            # Initialize robot reasoning capabilities
            self.robot_reasoning_integration = None  # Will be set up during model creation
            logger.info("🤖 Robot reasoning capabilities will be integrated")
        else:
            self.robot_reasoning_integration = None
            logger.info("🚫 Robot reasoning disabled in config")

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode text using BitNet encoder"""
        text_features, attention_patterns = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask)
        return text_features, attention_patterns

    def encode_vision(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Encode vision features using quantized vision encoder"""
        vision_latent = self.vision_encoder(
            vision_features)  # [batch_size, vision_latent_size]
        return vision_latent

    def create_episode(
        self,
        text_features: torch.Tensor,
        vision_latent: torch.Tensor,
        attention_weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Create multimodal episode for memory storage"""
        if not self.use_episodic_memory:
            return None

        # Pool text features (mean pooling)
        # [batch_size, text_encoder_dim]
        text_pooled = text_features.mean(dim=1)

        # Project both text and vision to episode dimension
        text_projected = self.text_to_episode(text_pooled)
        vision_projected = self.vision_to_episode(vision_latent)

        # Combine text and vision features (both now have episode_dim)
        episode = text_projected + vision_projected

        return episode

    def compute_cross_modal_contrastive_loss(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute cross-modal contrastive loss similar to CLIP
        """
        batch_size = text_features.shape[0]

        # Handle dimension mismatch between text and vision features
        text_dim = text_features.shape[-1]
        vision_dim = vision_features.shape[-1]

        if text_dim != vision_dim:
            # Project to smaller dimension to maintain compatibility
            target_dim = min(text_dim, vision_dim)

            if text_dim > vision_dim:
                # Project text features to vision dimension
                text_features = text_features[:, :target_dim]
            else:
                # Project vision features to text dimension
                vision_features = vision_features[:, :target_dim]

        # Normalize features
        text_features = F.normalize(text_features, dim=-1)
        vision_features = F.normalize(vision_features, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(text_features, vision_features.T) / temperature

        # Create labels (diagonal should be positive pairs)
        labels = torch.arange(batch_size, device=logits.device)

        # Compute cross-entropy loss for both directions
        text_to_vision_loss = F.cross_entropy(logits, labels)
        vision_to_text_loss = F.cross_entropy(logits.T, labels)

        return (text_to_vision_loss + vision_to_text_loss) / 2

    def compute_vision_reconstruction_loss(
        self,
        original_vision: torch.Tensor,
        reconstructed_vision: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute vision reconstruction loss to prevent vision encoder collapse
        """
        return F.mse_loss(reconstructed_vision, original_vision)

    def compute_memory_consistency_loss(
        self,
        episode: torch.Tensor,
        retrieved_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute memory consistency loss to encourage meaningful memory usage
        """
        if not self.use_episodic_memory or episode is None or retrieved_memory is None:
            return torch.tensor(0.0, device=episode.device if episode is not None else next(self.parameters()).device)

        # L2 regularization on memory difference
        memory_diff = episode - retrieved_memory
        return torch.mean(torch.norm(memory_diff, dim=-1))

    def compute_balanced_loss(
        self,
        decoder_loss: torch.Tensor,
        cross_modal_loss: torch.Tensor,
        vision_loss: Optional[torch.Tensor] = None,
        memory_loss: Optional[torch.Tensor] = None,
        robot_reasoning_loss: Optional[torch.Tensor] = None,
        fiber_losses: Optional[Dict[str, torch.Tensor]] = None,  # NEW: FIBER losses
        step: int = 0,
        adaptive_controller=None  # NEW: Adaptive training controller
    ) -> Dict[str, torch.Tensor]:
        """
        Compute balanced multi-objective loss with adaptive scaling
        Now includes FIBER losses (ITC + ITM + MLM)
        """
        # ENHANCED LEARNING VERIFICATION - Only in debug mode or for critical monitoring
        if not hasattr(self, '_loss_debug_count'):
            self._loss_debug_count = 0
            self._loss_history = []
        
        self._loss_debug_count += 1
        # Only show debug output for first 5 steps or every 200 steps (reduced frequency)
        should_debug = False  # Disable all debug prints to clean up output
        
        if should_debug:
            # Debug removed
            print(f"   Decoder loss: {decoder_loss.item():.6f}" if decoder_loss is not None else "   Decoder loss: None")
            print(f"   Cross-modal loss: {cross_modal_loss.item():.6f}" if cross_modal_loss is not None else "   Cross-modal loss: None")
            
            # Only show additional losses if they're significant
            if vision_loss is not None and vision_loss.item() > 0.001:
                print(f"   Vision loss: {vision_loss.item():.6f}")
            if memory_loss is not None and memory_loss.item() > 0.001:
                print(f"   Memory loss: {memory_loss.item():.6f}")
            if robot_reasoning_loss is not None and robot_reasoning_loss.item() > 0.001:
                print(f"   Robot reasoning loss: {robot_reasoning_loss.item():.6f}")
            
            # NEW: Log FIBER losses only if significant
            if fiber_losses is not None:
                significant_fiber_losses = {k: v for k, v in fiber_losses.items() 
                                          if v is not None and 'loss' in k and v.item() > 0.001}
                if significant_fiber_losses:
                    print(f"🔥 FIBER LOSSES:")
                    for key, loss in significant_fiber_losses.items():
                        print(f"   {key}: {loss.item():.6f}")
            
            # Track loss progression (include FIBER total loss)
            current_total = decoder_loss.item() if decoder_loss is not None else 0.0
            current_total += cross_modal_loss.item() if cross_modal_loss is not None else 0.0
            if fiber_losses and fiber_losses.get('fiber_total_loss') is not None:
                current_total += fiber_losses['fiber_total_loss'].item()
            self._loss_history.append(current_total)
            
            # Check if losses are decreasing (learning indicator)
            if len(self._loss_history) > 5:
                recent_avg = sum(self._loss_history[-3:]) / 3
                older_avg = sum(self._loss_history[-6:-3]) / 3
                if recent_avg < older_avg:
                    print(f"   ✅ LEARNING DETECTED: Loss decreasing (recent: {recent_avg:.6f} < older: {older_avg:.6f})")
                else:
                    print(f"   ⚠️  POTENTIAL ISSUE: Loss not decreasing (recent: {recent_avg:.6f} >= older: {older_avg:.6f})")
            
            # Check loss magnitudes
            if decoder_loss is not None and decoder_loss.item() > 10.0:
                print(f"   ⚠️  WARNING: Very high decoder loss - potential training instability")
            if cross_modal_loss is not None and cross_modal_loss.item() < 0.001:
                print(f"   ⚠️  WARNING: Very low cross-modal loss - model may not be learning alignment")
        
        losses = {'decoder_loss': decoder_loss, 'cross_modal_loss': cross_modal_loss}

        if vision_loss is not None:
            losses['vision_loss'] = vision_loss
        if memory_loss is not None and self.use_episodic_memory:
            losses['memory_loss'] = memory_loss
        if robot_reasoning_loss is not None:
            losses['robot_reasoning_loss'] = robot_reasoning_loss

        if self.adaptive_loss_scaling:
            # Adaptive scaling based on loss magnitudes
            with torch.no_grad():
                decoder_scale = 1.0
                cross_modal_scale = decoder_loss.item() / (cross_modal_loss.item() + 1e-8)
                vision_scale = decoder_loss.item() / (vision_loss.item() + 1e-8) if vision_loss is not None else 1.0
                memory_scale = decoder_loss.item() / (memory_loss.item() + 1e-8) if memory_loss is not None and self.use_episodic_memory else 1.0
                robot_reasoning_scale = decoder_loss.item() / (robot_reasoning_loss.item() + 1e-8) if robot_reasoning_loss is not None else 1.0

                # Clamp scales to reasonable range
                cross_modal_scale = torch.clamp(torch.tensor(cross_modal_scale), 0.1, 10.0).item()
                vision_scale = torch.clamp(torch.tensor(vision_scale), 0.1, 10.0).item() if vision_loss is not None else 1.0
                memory_scale = torch.clamp(torch.tensor(memory_scale), 0.1, 10.0).item() if memory_loss is not None and self.use_episodic_memory else 1.0
                robot_reasoning_scale = torch.clamp(torch.tensor(robot_reasoning_scale), 0.1, 10.0).item() if robot_reasoning_loss is not None else 1.0

        else:
            # Fixed scaling
            decoder_scale = self.text_loss_weight
            cross_modal_scale = self.cross_modal_loss_weight
            vision_scale = self.vision_loss_weight
            memory_scale = self.memory_loss_weight
            robot_reasoning_scale = 1.0
            
            # Debug loss weights every few steps
            if should_debug:
                print(f"   🔧 LOSS WEIGHTS USED:")
                print(f"      Text (decoder) weight: {decoder_scale}")
                print(f"      Cross-modal weight: {cross_modal_scale}")
                print(f"      Vision weight: {vision_scale}")
                print(f"      Memory weight: {memory_scale}")
                print(f"      Robot reasoning weight: {robot_reasoning_scale}")

        # Apply adaptive controller modifications if available
        if adaptive_controller is not None:
            try:
                # Get current similarity for adaptive training
                current_similarity = adaptive_controller.compute_cross_modal_similarity(
                    cross_modal_loss.detach().cpu().numpy()
                )

                # Check if intervention is needed
                intervention_applied = adaptive_controller.check_and_apply_intervention(
                    similarity_score=current_similarity,
                    step=step
                )

                if intervention_applied:
                    # Modify loss weights during intervention
                    cross_modal_scale *= adaptive_controller.loss_rebalance_factor
                    logger.info(f"🤖 Adaptive intervention applied at step {step} - boosting cross-modal loss by {adaptive_controller.loss_rebalance_factor}x")

            except Exception as e:
                logger.warning(f"⚠️ Adaptive controller error: {e}")

        # Compute weighted total loss (including FIBER losses)
        total_loss = (
            decoder_scale * decoder_loss +
            cross_modal_scale * cross_modal_loss
        )

        if vision_loss is not None:
            total_loss += vision_scale * vision_loss

        if memory_loss is not None and self.use_episodic_memory:
            total_loss += memory_scale * memory_loss

        if robot_reasoning_loss is not None:
            total_loss += robot_reasoning_scale * robot_reasoning_loss

        # NEW: Add FIBER losses to total loss with reduced scale for stability
        fiber_scale = self.config.get('fiber_loss_scale', 0.1)  # Fixed: use self.config instead of config
        if fiber_losses is not None and fiber_losses.get('fiber_total_loss') is not None:
            total_loss += fiber_scale * fiber_losses['fiber_total_loss']

        return {
            'total_loss': total_loss,
            'decoder_loss': decoder_loss,
            'cross_modal_loss': cross_modal_loss,
            'vision_loss': vision_loss,
            'memory_loss': memory_loss if self.use_episodic_memory else torch.tensor(0.0),
            'robot_reasoning_loss': robot_reasoning_loss if robot_reasoning_loss is not None else torch.tensor(0.0),
            'fiber_losses': fiber_losses,  # NEW: Include FIBER losses in output
            'decoder_scale': decoder_scale,
            'cross_modal_scale': cross_modal_scale,
            'vision_scale': vision_scale,
            'memory_scale': memory_scale if self.use_episodic_memory else 0.0,
            'robot_reasoning_scale': robot_reasoning_scale if robot_reasoning_loss is not None else 0.0
        }

    def freeze_encoders_conditionally(self, step: int):
        """Conditionally freeze encoders based on training step"""
        self.current_step = step

        # Freeze text encoder if specified
        if self.freeze_text_encoder_steps > 0 and step < self.freeze_text_encoder_steps:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.text_encoder.parameters():
                param.requires_grad = True

        # Freeze vision encoder if specified
        if self.freeze_vision_encoder_steps > 0 and step < self.freeze_vision_encoder_steps:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        step: int = 0,
        has_vision: Optional[torch.Tensor] = None,
        adaptive_controller=None,  # Adaptive training controller
        output_attentions: bool = False,  # NEW: For attention visualization
        output_hidden_states: bool = False,  # NEW: For hidden state analysis
        return_dict: bool = False  # NEW: For structured output
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass with FIBER-enhanced cross-modal fusion
        """
        batch_size = input_ids.shape[0]

        # Conditional encoder freezing
        self.freeze_encoders_conditionally(step)

        # 1. Encode text using BitNet encoder
        # Debug removed
        # Debug removed
        
        # 🔍 DATA DIAGNOSTIC: Check if we're seeing different data (reduced logging)
        if hasattr(self, '_last_input_hash'):
            current_hash = hash(input_ids.sum().item())
            if current_hash == self._last_input_hash:
                logger.warning(f"⚠️ Same input data as previous batch - training may be stuck!")
            # Remove noisy "Different input data" messages
            self._last_input_hash = current_hash
        else:
            self._last_input_hash = hash(input_ids.sum().item())
            logger.debug(f"✅ First batch processed")
        
        # Check input data properties
        unique_tokens = torch.unique(input_ids).numel()
        # Debug removed
        logger.debug(f"🔍 Debug: Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
        
        try:
            text_features, text_attention_patterns = self.encode_text(input_ids, attention_mask)
            # Debug removed
            logger.debug(f"🔍 Debug: Text features shape: {text_features.shape}")
        except Exception as e:
            # Debug removed
            raise e

        # 2. Encode vision using quantized vision encoder
        # Debug removed
        logger.debug(f"🔍 Debug: Vision input shape: {vision_features.shape}")
        
        try:
            vision_latent = self.encode_vision(vision_features)
            # Debug removed
            logger.debug(f"🔍 Debug: Vision latent shape: {vision_latent.shape}")
        except Exception as e:
            # Debug removed
            raise e

        # 3. AUTHENTIC FIBER-enhanced cross-modal fusion with proper loss objectives
        # Debug removed
        logger.debug(f"🔄 Applying AUTHENTIC FIBER fusion - Text: {text_features.shape}, Vision: {vision_latent.shape}")

        try:
            # Use authentic FIBER with all loss components
            logger.info(f"🔥 Calling FIBER with compute_losses=True, input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")
            fused_features, fiber_outputs = self.fusion(
                text_features=text_features,
                vision_features=vision_latent,
                input_ids=input_ids,
                text_attention_mask=attention_mask,
                labels=labels,
                compute_losses=True  # Enable FIBER loss computation
            )
            logger.info(f"✅ FIBER fusion completed - Output: {fused_features.shape}")
            logger.info(f"   FIBER outputs keys: {list(fiber_outputs.keys()) if fiber_outputs else 'None'}")
        except Exception as e:
            # Debug removed
            print(f"   Text features: {text_features.shape}")
            print(f"   Vision latent: {vision_latent.shape}")
            print(f"   Attention mask: {attention_mask.shape}")
            logger.error(f"❌ FIBER fusion failed: {e}")
            logger.error(f"   Text features: {text_features.shape}")
            logger.error(f"   Vision latent: {vision_latent.shape}")
            logger.error(f"   Attention mask: {attention_mask.shape}")
            raise e

        # Log FIBER fusion results
        logger.debug(f"✅ AUTHENTIC FIBER fusion completed - Output: {fused_features.shape}")
        logger.debug(f"   Fusion type: {fiber_outputs.get('fusion_type', 'unknown')}")

        # Extract enhanced vision features from FIBER output
        enhanced_vision = fiber_outputs.get('enhanced_vision')
        if enhanced_vision is not None:
            # Use enhanced vision features for downstream tasks
            if enhanced_vision.dim() == 3 and enhanced_vision.shape[1] == 1:
                enhanced_vision = enhanced_vision.squeeze(1)  # Remove sequence dimension
            logger.debug(f"🔥 Enhanced vision features: {enhanced_vision.shape}")

        # Extract FIBER losses for integration with main training loop
        fiber_itc_loss = fiber_outputs.get('itc_loss')
        fiber_itm_loss = fiber_outputs.get('itm_loss') 
        fiber_mlm_loss = fiber_outputs.get('mlm_loss')
        fiber_total_loss = fiber_outputs.get('fiber_total_loss')
        
        # Log FIBER loss components
        if fiber_total_loss is not None:
            logger.debug(f"🔥 FIBER Losses - Total: {fiber_total_loss.item():.4f}")
            logger.debug(f"   • ITC: {fiber_itc_loss.item():.4f}")
            logger.debug(f"   • ITM: {fiber_itm_loss.item():.4f}")
            logger.debug(f"   • MLM: {fiber_mlm_loss.item():.4f}")

        # 4. Enhanced Episodic memory processing (OPTIONAL for ablation study)
        memory_output = None
        memory_attention_weights = None
        memory_metadata = {}
        episode = None

        logger.info(f"🧠 Enhanced Memory enabled: {self.use_episodic_memory}")
        if self.use_episodic_memory:
            logger.info(f"🧠 Starting enhanced episodic memory processing")
            try:
                # Create multimodal episode using FIBER outputs
                episode = self.create_episode(
                    text_features, vision_latent, fiber_outputs)
                logger.info(f"🧠 Episode created: {episode.shape}")
                
                # Enhanced memory processing with cross-modal features
                text_episode = self.text_to_episode(text_features.mean(dim=1))  # Pool text features
                vision_episode = self.vision_to_episode(vision_latent)
                
                # Use enhanced memory with multi-modal support
                if hasattr(self.memory, 'write_to_memory'):
                    # Write to enhanced memory with modality-specific features
                    memory_state, kl_div = self.memory.write_to_memory(
                        episodes=episode.unsqueeze(0),  # Add episode dimension
                        text_features=text_episode.unsqueeze(0),
                        vision_features=vision_episode.unsqueeze(0)
                    )
                    
                    # Read from enhanced memory
                    memory_output, memory_attention_weights, memory_metadata = self.memory.read_from_memory(
                        query=episode.unsqueeze(0),
                        text_query=text_episode.unsqueeze(0),
                        vision_query=vision_episode.unsqueeze(0),
                        return_attention=output_attentions
                    )
                    
                    # Remove episode dimension for downstream processing
                    if memory_output is not None:
                        memory_output = memory_output.squeeze(0)
                    if memory_attention_weights is not None:
                        memory_attention_weights = memory_attention_weights.squeeze(0)
                        
                    logger.info(f"🧠 Enhanced memory processing completed: {memory_output.shape if memory_output is not None else 'None'}")
                else:
                    # Fallback to original memory
                    memory_output, memory_attention_weights = self.memory(episode)
                    
            except Exception as e:
                logger.error(f"🧠 Enhanced memory processing failed: {e}")
                print(f"   Episode shape: {episode.shape}")
                if hasattr(self.memory, 'memory_size'):
                    print(f"   Memory size: {self.memory.memory_size}")
                    print(f"   Episode dim: {self.memory.episode_dim}")
                raise e

            # Debug removed
            try:
                # Project memory output for decoder - Larimar-style integration
                # Debug removed
                
                # Instead of expanding to sequence length, integrate memory at the latent level
                # Following Larimar's approach: memory enhances the latent representation
                
                # Option 1: Add memory as a residual connection to the fused features
                # by projecting memory to sequence-level representation
                memory_enhanced_features = fused_features.clone()  # Start with original features
                
                # Pool fused features to get a single representation per batch item
                fused_pooled = torch.mean(fused_features, dim=1)  # [batch, hidden_dim]
                # Debug removed
                
                # Combine pooled features with memory output
                if memory_output.shape == fused_pooled.shape:
                    memory_combined = memory_output + fused_pooled
                    # Debug removed
                else:
                    # Project to matching dimensions
                    memory_proj = self.memory_to_decoder(memory_output)  
                    # Debug removed
                    memory_combined = memory_proj + fused_pooled
                    # Debug removed
                
                # Broadcast the combined representation to all sequence positions
                memory_broadcast = memory_combined.unsqueeze(1).expand(-1, fused_features.size(1), -1)
                # Debug removed
                
                # Create decoder input by adding memory to fused features
                decoder_input = self.decoder_input_proj(fused_features) + memory_broadcast
                # Debug removed
                
            except Exception as e:
                # Debug removed
                print(f"   Memory output shape: {memory_output.shape}")
                print(f"   Fused features shape: {fused_features.shape}")
                import traceback
                traceback.print_exc()
                raise e
        else:
            # Debug removed
            # Direct fusion to decoder (no memory)
            decoder_input = self.direct_fusion_proj(fused_features)
            # Debug removed

        # 5. Generate text using BitNet decoder
        # Debug removed
        # Debug removed
        # Debug removed
        # Debug removed
        
        try:
            decoder_outputs = self.text_decoder(
                inputs_embeds=decoder_input,
                attention_mask=attention_mask,
                labels=labels
            )
            # Debug removed
        except Exception as e:
            # Debug removed
            print(f"   Decoder input shape: {decoder_input.shape}")
            print(f"   Attention mask shape: {attention_mask.shape}")
            print(f"   Labels shape: {labels.shape if labels is not None else None}")
            raise e

        # 6. Compute losses
        decoder_loss = decoder_outputs['loss'] if decoder_outputs['loss'] is not None else torch.tensor(0.0)

        # Cross-modal contrastive loss using FIBER-enhanced features
        text_pooled = fused_features.mean(dim=1)  # Use FIBER-enhanced text
        vision_for_contrastive = enhanced_vision if enhanced_vision is not None else vision_latent

        cross_modal_loss = self.compute_cross_modal_contrastive_loss(
            text_pooled, vision_for_contrastive, temperature=self.loss_scale_temperature
        )

        # Vision reconstruction loss (optional)
        vision_loss = None
        if enhanced_vision is not None:
            vision_loss = self.compute_vision_reconstruction_loss(
                vision_latent, enhanced_vision
            )

        # Memory consistency loss (only if memory is enabled)
        memory_loss = None
        if self.use_episodic_memory and episode is not None and memory_output is not None:
            memory_loss = self.compute_memory_consistency_loss(episode, memory_output)

        # NEW: Robot reasoning loss (similar to deepseek-r1's multi-objective training)
        robot_reasoning_loss = None
        robot_reasoning_outputs = None
        # Temporarily disable robot reasoning to resolve device issues
        # if self.robot_reasoning_integration is not None:
        if False:  # Temporarily disabled
            try:
                # Ensure fused_features is on the same device as the model
                device = next(self.parameters()).device
                fused_features = fused_features.to(device)
                
                # Ensure robot reasoning head is on the correct device
                if hasattr(self, 'robot_reasoning') and self.robot_reasoning is not None:
                    self.robot_reasoning = self.robot_reasoning.to(device)
                
                # Apply robot reasoning to fused features
                robot_reasoning_outputs = self.robot_reasoning_integration.robot_selection_head(fused_features)
                
                # Ensure robot reasoning outputs are on correct device
                if isinstance(robot_reasoning_outputs, torch.Tensor):
                    robot_reasoning_outputs = robot_reasoning_outputs.to(device)
                elif isinstance(robot_reasoning_outputs, dict):
                    robot_reasoning_outputs = {k: v.to(device) if torch.is_tensor(v) else v 
                                             for k, v in robot_reasoning_outputs.items()}

                # Compute robot selection loss if labels are provided and this is a reasoning batch
                if hasattr(self, '_current_robot_labels') and self._current_robot_labels:
                    # Robot labels are strings, so we can pass them directly
                    robot_reasoning_loss = self.robot_reasoning_integration.compute_reasoning_loss(
                        outputs={'fused_features': fused_features.to(device)},  # Ensure device consistency
                        robot_labels=self._current_robot_labels  # Keep as strings
                    )

            except Exception as e:
                logger.warning(f"Robot reasoning forward pass failed: {e}")
                # Continue training without robot reasoning
                robot_reasoning_loss = None

        # Balanced loss computation (enhanced for FIBER and robot reasoning)
        fiber_losses_dict = {
            'itc_loss': fiber_itc_loss,
            'itm_loss': fiber_itm_loss,
            'mlm_loss': fiber_mlm_loss,
            'fiber_total_loss': fiber_total_loss
        }  # Always include dict, even with None values
        
        loss_dict = self.compute_balanced_loss(
            decoder_loss=decoder_loss,
            cross_modal_loss=cross_modal_loss,
            vision_loss=vision_loss,
            memory_loss=memory_loss,
            robot_reasoning_loss=robot_reasoning_loss,  # Include robot reasoning loss
            fiber_losses=fiber_losses_dict,  # NEW: Include FIBER losses
            step=step,
            adaptive_controller=adaptive_controller
        )

        # Collect attention patterns for visualization if requested
        attention_outputs = {}
        if output_attentions:
            # Text encoder attention patterns
            attention_outputs['text_attentions'] = text_attention_patterns
            
            # Text decoder attention patterns
            attention_outputs['decoder_attentions'] = decoder_outputs.get('attention_patterns', [])
            
            # FIBER fusion attention patterns
            if 'attention_patterns' in fiber_outputs:
                attention_outputs['fiber_attentions'] = fiber_outputs['attention_patterns']
            
            # Cross-modal attention patterns
            if 'cross_attention_weights' in fiber_outputs:
                attention_outputs['cross_attentions'] = fiber_outputs['cross_attention_weights']
            
            # Episodic memory attention patterns
            if memory_attention_weights is not None:
                attention_outputs['memory_attention'] = memory_attention_weights
                
            # Memory usage patterns
            if memory_metadata:
                attention_outputs['memory_usage'] = memory_metadata.get('memory_usage')
                attention_outputs['memory_metadata'] = memory_metadata

        # Collect hidden states if requested
        hidden_states_outputs = {}
        if output_hidden_states:
            hidden_states_outputs['text_hidden_states'] = text_features
            hidden_states_outputs['vision_hidden_states'] = vision_latent
            hidden_states_outputs['fused_hidden_states'] = fused_features
            if enhanced_vision is not None:
                hidden_states_outputs['enhanced_vision_states'] = enhanced_vision
            if memory_output is not None:
                hidden_states_outputs['memory_states'] = memory_output

        # Return comprehensive outputs
        outputs = {
            'loss': loss_dict['total_loss'],
            'logits': decoder_outputs['logits'],
            'text_features': text_features,
            'vision_latent': vision_latent,
            'enhanced_vision': enhanced_vision,  # FIBER-enhanced vision
            'fused_features': fused_features,   # FIBER-fused features
            'fiber_outputs': fiber_outputs,  # AUTHENTIC FIBER outputs with attention patterns and losses
            'fiber_losses': fiber_losses_dict,  # FIBER losses for easy access
            'text_attention_patterns': text_attention_patterns,
            'decoder_attention_patterns': decoder_outputs['attention_patterns'],
            'episode': episode,
            'memory_output': memory_output,
            'memory_state': memory_output,  # Add memory state for diagnostics
            'memory_attention_weights': memory_attention_weights,
            'memory_metadata': memory_metadata,  # Enhanced memory metadata
            'loss_components': loss_dict,
            'fiber_config': self.fiber_config,
            # NEW: Robot reasoning outputs (similar to deepseek-r1's structured outputs)
            'robot_reasoning_outputs': robot_reasoning_outputs,
            'robot_reasoning_loss': robot_reasoning_loss,
            # NEW: Attention and hidden states for visualization
            **attention_outputs,
            **hidden_states_outputs
        }
        
        # Enhanced cross-modal similarity for visualization
        if output_attentions and text_features is not None and vision_latent is not None:
            text_pooled = text_features.mean(dim=1)
            vision_pooled = vision_latent
            cross_modal_similarity = torch.cosine_similarity(text_pooled, vision_pooled, dim=-1).mean()
            outputs['cross_modal_similarity'] = cross_modal_similarity
        
        # Return structured output if requested
        if return_dict:
            return outputs
        else:
            return outputs['loss']  # Backward compatibility
        
        # Log metrics to wandb if available (instead of printing)
        if hasattr(self, 'use_wandb') and self.use_wandb and hasattr(self, 'wandb_logger') and self.wandb_logger:
            try:
                log_dict = {
                    'loss/total': loss_dict['total_loss'].item(),
                    'loss/decoder': loss_dict.get('decoder_loss', torch.tensor(0.0)).item() if loss_dict.get('decoder_loss') is not None else 0.0,
                    'loss/cross_modal': loss_dict.get('cross_modal_loss', torch.tensor(0.0)).item() if loss_dict.get('cross_modal_loss') is not None else 0.0,
                    'loss/vision': loss_dict.get('vision_loss', torch.tensor(0.0)).item() if loss_dict.get('vision_loss') is not None else 0.0,
                    'loss/memory': loss_dict.get('memory_loss', torch.tensor(0.0)).item() if loss_dict.get('memory_loss') is not None else 0.0,
                }
                
                # Add FIBER losses if available
                if hasattr(self, 'fiber_losses') and self.fiber_losses:
                    for key, value in self.fiber_losses.items():
                        if value is not None and 'loss' in key:
                            log_dict[f'fiber/{key}'] = value.item()
                
                self.wandb_logger.log(log_dict)
            except Exception as e:
                pass  # Silent fail for wandb logging
        
        # Enhanced gradient flow diagnostics with comprehensive monitoring
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 1
            self._param_norms_history = []
            self._grad_norms_history = []
            
        should_debug = False  # Disable all debug prints
        
        if should_debug:  # Show first 10 steps and every 50 steps for monitoring
            param_count = 0
            grad_count = 0
            total_grad_norm = 0.0
            total_param_norm = 0.0
            component_grads = {}  # Track gradients by component
            
            for name, param in self.named_parameters():
                param_count += 1
                param_norm = param.norm().item()
                total_param_norm += param_norm ** 2
                
                if param.grad is not None:
                    grad_count += 1
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    
                    # Track component gradients
                    if 'text_encoder' in name:
                        component_grads.setdefault('text_encoder', []).append(grad_norm)
                    elif 'vision' in name:
                        component_grads.setdefault('vision', []).append(grad_norm)
                    elif 'fiber' in name:
                        component_grads.setdefault('fiber', []).append(grad_norm)
                    elif 'memory' in name:
                        component_grads.setdefault('memory', []).append(grad_norm)
                    elif 'decoder' in name:
                        component_grads.setdefault('decoder', []).append(grad_norm)
                    
                # Show first 3 params for debugging
                if param_count <= 3:
                    grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                    print(f"   Param '{name[:50]}': shape={param.shape}, grad_norm={grad_norm:.6f}, param_norm={param_norm:.6f}")
            
            total_grad_norm = (total_grad_norm ** 0.5)
            total_param_norm = (total_param_norm ** 0.5)
            
            # Debug removed
            print(f"   Parameters: {param_count} total, {grad_count} with gradients ({grad_count/param_count*100:.1f}%)")
            print(f"   Total gradient norm: {total_grad_norm:.6f}")
            print(f"   Total parameter norm: {total_param_norm:.6f}")
            
            # Component-wise gradient analysis
            for component, grads in component_grads.items():
                avg_grad = sum(grads) / len(grads)
                print(f"   {component} avg gradient norm: {avg_grad:.6f} ({len(grads)} params)")
            
            # Track gradient history and check for learning
            self._grad_norms_history.append(total_grad_norm)
            self._param_norms_history.append(total_param_norm)
            
            if len(self._grad_norms_history) > 5:
                param_change = abs(self._param_norms_history[-1] - self._param_norms_history[-6])
                if param_change > 1e-6:
                    print(f"   ✅ PARAMETERS UPDATING: Change of {param_change:.6f} over 5 steps")
                else:
                    print(f"   ⚠️  WARNING: Parameters barely changing ({param_change:.8f}) - potential learning issue")
            
            # Gradient magnitude warnings
            if total_grad_norm < 1e-6:
                print(f"   ⚠️  CRITICAL: Gradient norm extremely small ({total_grad_norm:.8f}) - model likely NOT learning!")
            elif total_grad_norm > 100.0:
                print(f"   ⚠️  WARNING: Gradient norm very large ({total_grad_norm:.2f}) - potential exploding gradients")
            elif 1e-4 < total_grad_norm < 1e-1:
                print(f"   ✅ HEALTHY: Gradient norm in good range for learning ({total_grad_norm:.6f})")

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate text conditioned on vision using FIBER-enhanced fusion
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generated_ids = input_ids.clone()

        with torch.no_grad():
            for step in range(max_length):
                # Get current sequence length
                current_length = generated_ids.shape[1]

                # Update attention mask for current sequence
                current_attention_mask = torch.ones(
                    batch_size, current_length, device=device
                )

                # Forward pass with FIBER fusion
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=current_attention_mask,
                    vision_features=vision_features,
                    step=step
                )

                # Get logits for next token
                next_token_logits = outputs['logits'][:, -1, :] / temperature

                # Apply top-p filtering
                if do_sample and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove, float('-inf'))

                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Check for EOS token
                if torch.all(next_token == self.tokenizer.eos_token_id):
                    break

        self.train()
        return generated_ids

    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        info = {
            'model_type': 'BitMar',
            'text_encoder_dim': self.config['text_encoder_dim'],
            'text_decoder_dim': self.config['text_decoder_dim'],
            'vision_latent_size': self.config['vision_latent_size'],
            'fusion_hidden_size': self.config['fusion_hidden_size'],
            'use_episodic_memory': self.use_episodic_memory,
            'fusion_type': 'authentic_fiber_backbone_fusion',
            'fiber_config': self.fiber_config
        }

        if self.use_episodic_memory:
            info.update({
                'memory_size': self.config['memory_size'],
                'episode_dim': self.config['episode_dim'],
                'memory_alpha': self.config['memory_alpha']
            })

            if hasattr(self, 'memory'):
                info['memory_info'] = self.memory.get_memory_info()

        return info


def create_bitmar_model(config: Dict) -> BitMarModel:
    """Create BitMar model with FIBER fusion and robot reasoning"""
    logger.info("🚀 Creating BitMar model with authentic FIBER fusion and robot reasoning...")

    # Log FIBER integration status
    logger.info("🔥 FIBER Integration Status:")
    logger.info(f"  • Authentic FIBER fusion: ✅ Enabled")
    logger.info(f"  • Fusion strategy: {config.get('fiber_fusion_strategy', 'deep_backbone')}")
    logger.info(f"  • Backbone integration: {config.get('fiber_backbone_integration', True)}")
    logger.info(f"  • Bidirectional fusion: {config.get('fiber_bidirectional_fusion', True)}")

    model = BitMarModel(config)

    # NEW: Integrate robot reasoning capabilities (similar to deepseek-r1's post-model setup)
    if model.enable_robot_reasoning:
        try:
            robot_data_dir = config.get('robot_data_dir', "D:/BabyLM/robot_selection_data/data")
            model.robot_reasoning_integration = create_robot_reasoning_integration(model, robot_data_dir)
            logger.info("🤖 Robot reasoning integration completed successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to integrate robot reasoning: {e}")
            model.robot_reasoning_integration = None
            model.enable_robot_reasoning = False

    logger.info("✅ BitMar model created with FIBER-enhanced cross-modal fusion")
    logger.info(f"   Episodic Memory: {'Enabled' if model.use_episodic_memory else 'Disabled (Ablation Study)'}")
    logger.info(f"   Robot Reasoning: {'Enabled' if model.robot_reasoning_integration is not None else 'Disabled'}")

    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }
