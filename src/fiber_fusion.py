"""
FIBER Integration for BitGen
Based on the FIBER implementation from Microsoft Research
Implements coarse-to-fine vision-language pre-training with fusion in the backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FIBERCrossModalLayer(nn.Module):
    """
    FIBER cross-modal layer that enables bidirectional attention
    between vision and text modalities within transformer blocks
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        is_decoder: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Cross-modal attention layers (vision attending to text)
        self.vision_to_text_query = nn.Linear(hidden_size, self.all_head_size)
        self.vision_to_text_key = nn.Linear(hidden_size, self.all_head_size)
        self.vision_to_text_value = nn.Linear(hidden_size, self.all_head_size)

        # Cross-modal attention layers (text attending to vision)
        self.text_to_vision_query = nn.Linear(hidden_size, self.all_head_size)
        self.text_to_vision_key = nn.Linear(hidden_size, self.all_head_size)
        self.text_to_vision_value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        # Output projections
        self.vision_output = nn.Linear(hidden_size, hidden_size)
        self.text_output = nn.Linear(hidden_size, hidden_size)

        # Layer normalization
        self.vision_LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.text_LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Dropout
        self.vision_dropout = nn.Dropout(hidden_dropout_prob)
        self.text_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        vision_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        vision_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Bidirectional cross-modal attention as in FIBER

        Args:
            vision_hidden_states: [batch_size, vision_seq_len, hidden_size]
            text_hidden_states: [batch_size, text_seq_len, hidden_size]
            vision_attention_mask: Optional vision attention mask
            text_attention_mask: Optional text attention mask

        Returns:
            Enhanced vision features, enhanced text features, and optionally attention weights
        """
        batch_size = vision_hidden_states.shape[0]

        # Vision attending to text
        vision_query_layer = self.transpose_for_scores(self.vision_to_text_query(vision_hidden_states))
        text_key_layer = self.transpose_for_scores(self.text_to_vision_key(text_hidden_states))
        text_value_layer = self.transpose_for_scores(self.text_to_vision_value(text_hidden_states))

        # Compute vision-to-text attention scores
        v2t_attention_scores = torch.matmul(vision_query_layer, text_key_layer.transpose(-1, -2))
        v2t_attention_scores = v2t_attention_scores / math.sqrt(self.attention_head_size)

        if text_attention_mask is not None:
            # Apply text attention mask to vision-to-text attention
            v2t_attention_scores = v2t_attention_scores + text_attention_mask.unsqueeze(1).unsqueeze(1)

        v2t_attention_probs = F.softmax(v2t_attention_scores, dim=-1)
        v2t_attention_probs = self.dropout(v2t_attention_probs)

        # Vision enhanced by text
        v2t_context_layer = torch.matmul(v2t_attention_probs, text_value_layer)
        v2t_context_layer = v2t_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = v2t_context_layer.size()[:-2] + (self.all_head_size,)
        v2t_context_layer = v2t_context_layer.view(*new_context_layer_shape)

        # Text attending to vision
        text_query_layer = self.transpose_for_scores(self.text_to_vision_query(text_hidden_states))
        vision_key_layer = self.transpose_for_scores(self.vision_to_text_key(vision_hidden_states))
        vision_value_layer = self.transpose_for_scores(self.vision_to_text_value(vision_hidden_states))

        # Compute text-to-vision attention scores
        t2v_attention_scores = torch.matmul(text_query_layer, vision_key_layer.transpose(-1, -2))
        t2v_attention_scores = t2v_attention_scores / math.sqrt(self.attention_head_size)

        if vision_attention_mask is not None:
            # Apply vision attention mask to text-to-vision attention
            t2v_attention_scores = t2v_attention_scores + vision_attention_mask.unsqueeze(1).unsqueeze(1)

        t2v_attention_probs = F.softmax(t2v_attention_scores, dim=-1)
        t2v_attention_probs = self.dropout(t2v_attention_probs)

        # Text enhanced by vision
        t2v_context_layer = torch.matmul(t2v_attention_probs, vision_value_layer)
        t2v_context_layer = t2v_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = t2v_context_layer.size()[:-2] + (self.all_head_size,)
        t2v_context_layer = t2v_context_layer.view(*new_context_layer_shape)

        # Apply output transformations and residual connections
        enhanced_vision = self.vision_output(v2t_context_layer)
        enhanced_vision = self.vision_dropout(enhanced_vision)
        enhanced_vision = self.vision_LayerNorm(enhanced_vision + vision_hidden_states)

        enhanced_text = self.text_output(t2v_context_layer)
        enhanced_text = self.text_dropout(enhanced_text)
        enhanced_text = self.text_LayerNorm(enhanced_text + text_hidden_states)

        outputs = (enhanced_vision, enhanced_text)
        if output_attentions:
            outputs += (v2t_attention_probs, t2v_attention_probs)

        return outputs


class FIBERTransformerBlock(nn.Module):
    """
    FIBER transformer block with integrated cross-modal attention
    Based on the FIBER implementation
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        enable_cross_modal: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_cross_modal = enable_cross_modal

        # Self-attention for both modalities
        self.vision_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            batch_first=True
        )

        self.text_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            batch_first=True
        )

        # Cross-modal attention (FIBER's key innovation)
        if enable_cross_modal:
            self.cross_modal_layer = FIBERCrossModalLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                layer_norm_eps=layer_norm_eps
            )

        # Feed-forward networks
        self.vision_intermediate = nn.Linear(hidden_size, intermediate_size)
        self.vision_output = nn.Linear(intermediate_size, hidden_size)
        self.vision_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.vision_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.text_intermediate = nn.Linear(hidden_size, intermediate_size)
        self.text_output = nn.Linear(intermediate_size, hidden_size)
        self.text_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.text_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.activation = nn.GELU()

    def forward(
        self,
        vision_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        vision_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with FIBER fusion

        Args:
            vision_hidden_states: [batch_size, vision_seq_len, hidden_size]
            text_hidden_states: [batch_size, text_seq_len, hidden_size]

        Returns:
            Enhanced vision and text hidden states, attention patterns
        """
        attention_outputs = {}

        # Self-attention for vision
        vision_normed = self.vision_norm1(vision_hidden_states)
        
        # Handle vision attention mask - ensure it matches vision sequence length
        vision_key_padding_mask = None
        if vision_attention_mask is not None:
            # Ensure vision_attention_mask has the right shape for vision sequence
            vision_seq_len = vision_hidden_states.shape[1]
            if vision_attention_mask.shape[1] != vision_seq_len:
                # Create appropriate mask for vision sequence length
                vision_key_padding_mask = torch.zeros(
                    vision_hidden_states.shape[0], vision_seq_len,
                    dtype=torch.bool, device=vision_hidden_states.device
                )
            else:
                vision_key_padding_mask = ~vision_attention_mask
        
        vision_self_output, vision_self_weights = self.vision_attention(
            vision_normed, vision_normed, vision_normed,
            key_padding_mask=vision_key_padding_mask,
            need_weights=output_attentions
        )
        vision_hidden_states = vision_hidden_states + vision_self_output
        if output_attentions:
            attention_outputs['vision_self_attention'] = vision_self_weights

        # Self-attention for text
        text_normed = self.text_norm1(text_hidden_states)
        
        # Handle text attention mask - ensure dimensions are correct
        text_key_padding_mask = None
        if text_attention_mask is not None:
            # text_attention_mask should be [batch_size, seq_len]
            if text_attention_mask.shape != text_hidden_states.shape[:2]:
                raise ValueError(f"Text attention mask shape {text_attention_mask.shape} doesn't match text features {text_hidden_states.shape[:2]}")
            text_key_padding_mask = ~text_attention_mask
            
        text_self_output, text_self_weights = self.text_attention(
            text_normed, text_normed, text_normed,
            key_padding_mask=text_key_padding_mask,
            need_weights=output_attentions
        )
        text_hidden_states = text_hidden_states + text_self_output
        if output_attentions:
            attention_outputs['text_self_attention'] = text_self_weights

        # Cross-modal attention (FIBER's core innovation)
        if self.enable_cross_modal:
            cross_modal_outputs = self.cross_modal_layer(
                vision_hidden_states=vision_hidden_states,
                text_hidden_states=text_hidden_states,
                vision_attention_mask=vision_attention_mask,
                text_attention_mask=text_attention_mask,
                output_attentions=output_attentions
            )

            vision_hidden_states = cross_modal_outputs[0]
            text_hidden_states = cross_modal_outputs[1]

            if output_attentions:
                attention_outputs['vision_to_text_attention'] = cross_modal_outputs[2]
                attention_outputs['text_to_vision_attention'] = cross_modal_outputs[3]

        # Feed-forward for vision
        vision_intermediate_output = self.vision_intermediate(vision_hidden_states)
        vision_intermediate_output = self.activation(vision_intermediate_output)
        vision_layer_output = self.vision_output(vision_intermediate_output)
        vision_layer_output = self.dropout(vision_layer_output)
        vision_hidden_states = self.vision_norm2(vision_layer_output + vision_hidden_states)

        # Feed-forward for text
        text_intermediate_output = self.text_intermediate(text_hidden_states)
        text_intermediate_output = self.activation(text_intermediate_output)
        text_layer_output = self.text_output(text_intermediate_output)
        text_layer_output = self.dropout(text_layer_output)
        text_hidden_states = self.text_norm2(text_layer_output + text_hidden_states)

        return vision_hidden_states, text_hidden_states, attention_outputs


class FIBERFusion(nn.Module):
    """
    FIBER fusion module based on the Microsoft Research implementation
    Implements coarse-to-fine vision-language pre-training with fusion in the backbone
    """

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_attention_heads: int = 8,
        num_layers: int = 6,
        num_fuse_layers: int = 6,
        intermediate_size: int = None,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_fuse_layers = num_fuse_layers

        if intermediate_size is None:
            intermediate_size = hidden_dim * 4

        logger.info(f"🔥 Initializing FIBER Fusion:")
        logger.info(f"  • Text dim: {text_dim}")
        logger.info(f"  • Vision dim: {vision_dim}")
        logger.info(f"  • Hidden dim: {hidden_dim}")
        logger.info(f"  • Total layers: {num_layers}")
        logger.info(f"  • Fusion layers: {num_fuse_layers}")

        # Input projections to common hidden dimension
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.vision_projection = nn.Linear(vision_dim, hidden_dim)

        # FIBER transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Enable cross-modal fusion in the last num_fuse_layers
            enable_cross_modal = i >= (num_layers - num_fuse_layers)

            layer = FIBERTransformerBlock(
                hidden_size=hidden_dim,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                layer_norm_eps=layer_norm_eps,
                enable_cross_modal=enable_cross_modal
            )
            self.layers.append(layer)

        # Output normalization
        self.vision_final_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.text_final_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Output projections
        self.vision_output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.text_output_projection = nn.Linear(hidden_dim, hidden_dim)

        logger.info(f"✅ FIBER Fusion initialized with {num_fuse_layers}/{num_layers} fusion layers")

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        FIBER forward pass with coarse-to-fine fusion

        Args:
            text_features: [batch_size, text_seq_len, text_dim]
            vision_features: [batch_size, vision_seq_len, vision_dim] or [batch_size, vision_dim]
            text_attention_mask: [batch_size, text_seq_len]
            vision_attention_mask: [batch_size, vision_seq_len] (optional)

        Returns:
            Enhanced text features, enhanced vision features, attention patterns
        """
        batch_size = text_features.shape[0]
        text_seq_len = text_features.shape[1]

        # Handle vision features - convert to sequence format if needed
        if vision_features.dim() == 2:  # [batch_size, vision_dim]
            vision_features = vision_features.unsqueeze(1)  # [batch_size, 1, vision_dim]
            vision_seq_len = 1
        else:  # [batch_size, vision_seq_len, vision_dim]
            vision_seq_len = vision_features.shape[1]

        # Project to common hidden dimension
        text_hidden = self.text_projection(text_features)
        vision_hidden = self.vision_projection(vision_features)

        # Store attention patterns
        all_attention_patterns = {
            'vision_self_attention': [],
            'text_self_attention': [],
            'vision_to_text_attention': [],
            'text_to_vision_attention': []
        }

        # Pass through FIBER transformer layers
        for layer_idx, layer in enumerate(self.layers):
            vision_hidden, text_hidden, attention_patterns = layer(
                vision_hidden_states=vision_hidden,
                text_hidden_states=text_hidden,
                vision_attention_mask=vision_attention_mask,
                text_attention_mask=text_attention_mask,
                output_attentions=output_attentions
            )

            if output_attentions:
                for key, value in attention_patterns.items():
                    if key in all_attention_patterns:
                        all_attention_patterns[key].append(value)

        # Final normalization
        vision_hidden = self.vision_final_norm(vision_hidden)
        text_hidden = self.text_final_norm(text_hidden)

        # Output projections
        enhanced_vision = self.vision_output_projection(vision_hidden)
        enhanced_text = self.text_output_projection(text_hidden)

        return enhanced_text, enhanced_vision, all_attention_patterns


class FIBERIntegration(nn.Module):
    """
    Integration layer to use FIBER fusion with BitGen's architecture
    This replaces the simplified cross-modal fusion with FIBER
    """

    def __init__(
        self,
        text_encoder_dim: int,
        vision_encoder_dim: int,
        fusion_hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 6,
        num_fuse_layers: int = 6,
        dropout: float = 0.1,
        fiber_config: Dict = None
    ):
        super().__init__()

        # Store configuration
        self.text_encoder_dim = text_encoder_dim
        self.vision_encoder_dim = vision_encoder_dim
        self.fusion_hidden_size = fusion_hidden_size
        self.hidden_dim = fusion_hidden_size  # Alias for compatibility

        # Apply FIBER-specific configuration if provided
        if fiber_config:
            num_fuse_layers = fiber_config.get('num_fiber_fusion_layers', num_fuse_layers)
            attention_temperature = fiber_config.get('fiber_attention_temperature', 1.0)
            cross_attention_dropout = fiber_config.get('fiber_cross_attention_dropout', dropout)

            logger.info("🔥 Applying FIBER configuration:")
            logger.info(f"  • Fusion layers: {num_fuse_layers}/{num_layers}")
            logger.info(f"  • Attention temperature: {attention_temperature}")
            logger.info(f"  • Cross-attention dropout: {cross_attention_dropout}")

        # Create FIBER fusion
        self.fiber_fusion = FIBERFusion(
            text_dim=text_encoder_dim,
            vision_dim=vision_encoder_dim,
            hidden_dim=fusion_hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            num_fuse_layers=num_fuse_layers,
            attention_probs_dropout_prob=cross_attention_dropout if fiber_config else dropout,
            hidden_dropout_prob=dropout
        )

        logger.info("🔥 FIBER integration initialized")

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass using FIBER fusion

        Args:
            text_features: [batch_size, seq_len, text_dim]
            vision_features: [batch_size, vision_dim]
            text_input_ids: Not used in FIBER (operates on features)
            text_attention_mask: [batch_size, seq_len]

        Returns:
            fused_features: [batch_size, seq_len, hidden_dim]
            attention_weights: Dict of attention patterns
        """
        # Use FIBER fusion
        enhanced_text, enhanced_vision, attention_patterns = self.fiber_fusion(
            text_features=text_features,
            vision_features=vision_features,
            text_attention_mask=text_attention_mask,
            output_attentions=True
        )

        # Return enhanced text features as the main output
        # (BitGen expects [batch_size, seq_len, hidden_dim])
        fused_features = enhanced_text

        # Compile attention patterns for compatibility with BitGen
        attention_weights = {
            'fiber_attention_patterns': attention_patterns,
            'fusion_type': 'fiber_backbone_fusion',
            'enhanced_vision': enhanced_vision,  # Also return enhanced vision
        }

        return fused_features, attention_weights


def create_fiber_fusion(
    text_encoder_dim: int,
    vision_encoder_dim: int,
    fusion_hidden_size: int,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
    num_fusion_layers: int = 6,
    config: Dict = None
) -> FIBERIntegration:
    """
    Create FIBER fusion module based on Microsoft Research's implementation
    """
    logger.info("🔥 Creating FIBER fusion to replace basic cross-attention")
    logger.info("   Based on Microsoft Research's FIBER implementation")
    logger.info(f"  • Text encoder dim: {text_encoder_dim}")
    logger.info(f"  • Vision encoder dim: {vision_encoder_dim}")
    logger.info(f"  • Fusion hidden size: {fusion_hidden_size}")
    logger.info(f"  • Fusion layers: {num_fusion_layers}")

    return FIBERIntegration(
        text_encoder_dim=text_encoder_dim,
        vision_encoder_dim=vision_encoder_dim,
        fusion_hidden_size=fusion_hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        num_fuse_layers=num_fusion_layers,
        dropout=dropout,
        fiber_config=config
    )

# Keep compatibility with old naming for now
create_authentic_fiber_fusion = create_fiber_fusion
AuthenticFIBERIntegration = FIBERIntegration
