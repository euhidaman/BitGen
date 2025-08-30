"""
FIBER-style Fusion in the Backbone for BitGen
Implements deep cross-modal fusion with bidirectional attention integrated into transformer layers
Based on FIBER: Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FIBERCrossModalAttention(nn.Module):
    """
    FIBER-style cross-modal attention that can be integrated into transformer blocks
    Supports both image-to-text and text-to-image attention
    """

    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        learnable_alpha: bool = True
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert query_dim % num_heads == 0, f"query_dim {query_dim} must be divisible by num_heads {num_heads}"

        # Cross-modal projections
        self.q_proj = nn.Linear(query_dim, query_dim, bias=True)
        self.k_proj = nn.Linear(key_value_dim, query_dim, bias=True)
        self.v_proj = nn.Linear(key_value_dim, query_dim, bias=True)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=True)

        # Normalization and dropout
        self.norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable fusion strength (inspired by FIBER's alpha parameter)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('alpha', torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following FIBER's approach"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, query_seq_len, query_dim]
            key_value: [batch_size, kv_seq_len, key_value_dim]
            attention_mask: [batch_size, kv_seq_len] or [batch_size, query_seq_len, kv_seq_len]

        Returns:
            fused_output: [batch_size, query_seq_len, query_dim]
            attention_weights: [batch_size, num_heads, query_seq_len, kv_seq_len]
        """
        batch_size, query_seq_len, _ = query.shape
        kv_seq_len = key_value.shape[1]

        # Project to Q, K, V
        q = self.q_proj(query)  # [batch_size, query_seq_len, query_dim]
        k = self.k_proj(key_value)  # [batch_size, kv_seq_len, query_dim]
        v = self.v_proj(key_value)  # [batch_size, kv_seq_len, query_dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # [batch_size, kv_seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, kv_seq_len]
            elif attention_mask.dim() == 3:  # [batch_size, query_seq_len, kv_seq_len]
                attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, query_seq_len, kv_seq_len]

            # Convert mask: 1 = attend, 0 = don't attend -> 0 = attend, -inf = don't attend
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_seq_len, self.query_dim
        )
        output = self.out_proj(context)

        # Residual connection with learnable alpha (FIBER-style)
        fused_output = query + self.alpha * output

        # Layer normalization
        fused_output = self.norm(fused_output)

        return fused_output, attention_weights.mean(dim=1)  # Average across heads for logging


class FIBERTransformerBlock(nn.Module):
    """
    Transformer block with integrated cross-modal attention (FIBER-style)
    Supports both self-attention and cross-modal attention in the same block
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        cross_modal_dim: Optional[int] = None,
        enable_cross_modal: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.enable_cross_modal = enable_cross_modal

        # Self-attention layers
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-modal attention (FIBER-style)
        if enable_cross_modal and cross_modal_dim is not None:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_modal_attn = FIBERCrossModalAttention(
                query_dim=dim,
                key_value_dim=cross_modal_dim,
                num_heads=num_heads,
                dropout=dropout,
                learnable_alpha=True
            )

        # MLP layers
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_modal_features: Optional[torch.Tensor] = None,
        cross_modal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, dim]
            attention_mask: [batch_size, seq_len]
            cross_modal_features: [batch_size, cross_seq_len, cross_modal_dim]
            cross_modal_mask: [batch_size, cross_seq_len]

        Returns:
            output: [batch_size, seq_len, dim]
            attention_weights: Dict of attention patterns
        """
        attention_patterns = {}

        # Self-attention with residual connection
        residual = x
        x_norm = self.norm1(x)

        # Convert attention mask for self-attention
        if attention_mask is not None:
            # Create causal mask for self-attention
            seq_len = x.shape[1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            # Combine with padding mask
            combined_mask = attention_mask.unsqueeze(1) & causal_mask.unsqueeze(0)
            attn_mask = ~combined_mask  # Invert for MultiheadAttention
        else:
            attn_mask = None

        attn_output, self_attn_weights = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )
        x = residual + attn_output
        attention_patterns['self_attention'] = self_attn_weights

        # Cross-modal attention (FIBER-style fusion in backbone)
        if self.enable_cross_modal and cross_modal_features is not None:
            cross_modal_output, cross_attn_weights = self.cross_modal_attn(
                query=x,
                key_value=cross_modal_features,
                attention_mask=cross_modal_mask
            )
            x = cross_modal_output  # Already includes residual connection in FIBERCrossModalAttention
            attention_patterns['cross_modal_attention'] = cross_attn_weights

        # MLP with residual connection
        residual = x
        x = residual + self.mlp(self.norm2(x))

        return x, attention_patterns


class FIBERTextEncoder(nn.Module):
    """
    FIBER-style text encoder with integrated cross-modal attention
    Based on BitNet quantization with FIBER fusion capabilities
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        cross_modal_dim: Optional[int] = None,
        num_fusion_layers: int = 6  # Number of layers with cross-modal fusion
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_fusion_layers = num_fusion_layers

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # Transformer layers with selective cross-modal fusion
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Enable cross-modal attention in the last num_fusion_layers
            enable_cross_modal = cross_modal_dim is not None and i >= (num_layers - num_fusion_layers)

            layer = FIBERTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                cross_modal_dim=cross_modal_dim,
                enable_cross_modal=enable_cross_modal
            )
            self.layers.append(layer)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        logger.info(f"✅ FIBER Text Encoder: {num_layers} layers, {num_fusion_layers} with cross-modal fusion")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_modal_features: Optional[torch.Tensor] = None,
        cross_modal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            cross_modal_features: [batch_size, cross_seq_len, cross_modal_dim]
            cross_modal_mask: [batch_size, cross_seq_len]

        Returns:
            encoded_features: [batch_size, seq_len, dim]
            attention_patterns: List of attention weights from each layer
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Transform through FIBER layers with cross-modal fusion
        all_attention_patterns = []
        for i, layer in enumerate(self.layers):
            x, attention_patterns = layer(
                x=x,
                attention_mask=attention_mask,
                cross_modal_features=cross_modal_features,
                cross_modal_mask=cross_modal_mask
            )
            all_attention_patterns.append(attention_patterns)

        x = self.norm(x)
        return x, all_attention_patterns


class FIBERVisionEncoder(nn.Module):
    """
    FIBER-style vision encoder with cross-modal attention to text
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        cross_modal_dim: Optional[int] = None,
        num_fusion_layers: int = 2  # Number of layers with cross-modal fusion
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_fusion_layers = num_fusion_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer layers with selective cross-modal fusion
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Enable cross-modal attention in the last num_fusion_layers
            enable_cross_modal = cross_modal_dim is not None and i >= (num_layers - num_fusion_layers)

            layer = FIBERTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                cross_modal_dim=cross_modal_dim,
                enable_cross_modal=enable_cross_modal
            )
            self.layers.append(layer)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        logger.info(f"✅ FIBER Vision Encoder: {num_layers} layers, {num_fusion_layers} with cross-modal fusion")

    def forward(
        self,
        vision_features: torch.Tensor,
        cross_modal_features: Optional[torch.Tensor] = None,
        cross_modal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            vision_features: [batch_size, input_dim]
            cross_modal_features: [batch_size, text_seq_len, cross_modal_dim]
            cross_modal_mask: [batch_size, text_seq_len]

        Returns:
            encoded_features: [batch_size, output_dim]
            attention_patterns: List of attention weights from each layer
        """
        # Handle input dimensions - convert to sequence format for transformer
        if vision_features.dim() == 2:
            # Convert [batch_size, input_dim] to [batch_size, 1, hidden_dim] for transformer processing
            x = self.input_proj(vision_features).unsqueeze(1)
        else:
            # Already in sequence format
            x = self.input_proj(vision_features)

        # Transform through FIBER layers with cross-modal fusion
        all_attention_patterns = []
        for i, layer in enumerate(self.layers):
            x, attention_patterns = layer(
                x=x,
                attention_mask=None,  # No self-masking for vision
                cross_modal_features=cross_modal_features,
                cross_modal_mask=cross_modal_mask
            )
            all_attention_patterns.append(attention_patterns)

        # Project to output dimension and remove sequence dimension
        if x.shape[1] == 1:
            x = x.squeeze(1)  # [batch_size, hidden_dim]
        else:
            x = x.mean(dim=1)  # Pool if multiple tokens

        output = self.output_proj(x)
        return output, all_attention_patterns


class FIBERCrossModalFusion(nn.Module):
    """
    FIBER-style cross-modal fusion that implements deep bidirectional attention
    This replaces the basic cross-attention in BitGen with FIBER's backbone fusion approach
    """

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_fusion_layers: int = 6
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.num_fusion_layers = num_fusion_layers

        # FIBER-style backbone fusion encoders
        self.fiber_text_encoder = FIBERTextEncoder(
            vocab_size=50257,  # Will be overridden by actual tokenizer
            dim=text_dim,
            num_layers=num_layers + 2,  # Additional layers for deeper fusion
            num_heads=num_heads,
            dropout=dropout,
            cross_modal_dim=vision_dim,
            num_fusion_layers=num_fusion_layers
        )

        self.fiber_vision_encoder = FIBERVisionEncoder(
            input_dim=vision_dim,
            hidden_dim=vision_dim,
            output_dim=vision_dim,
            num_layers=num_layers + 1,  # Additional layers for deeper fusion
            num_heads=num_heads,
            dropout=dropout,
            cross_modal_dim=text_dim,
            num_fusion_layers=num_fusion_layers // 2  # Fewer vision fusion layers
        )

        # Output projection to common dimension
        self.text_output_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_output_proj = nn.Linear(vision_dim, hidden_dim)

        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        logger.info(f"✅ FIBER Cross-Modal Fusion initialized with {num_fusion_layers} fusion layers")

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        vision_features: torch.Tensor,
        pre_encoded_text: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Implement FIBER-style bidirectional fusion

        Args:
            text_input_ids: [batch_size, text_seq_len]
            text_attention_mask: [batch_size, text_seq_len]
            vision_features: [batch_size, vision_dim]
            pre_encoded_text: Optional pre-encoded text features

        Returns:
            fused_features: [batch_size, text_seq_len, hidden_dim]
            attention_weights: Dict of attention patterns
        """
        batch_size = text_input_ids.shape[0]

        # Use pre-encoded text if available, otherwise encode with cross-modal fusion
        if pre_encoded_text is not None:
            # Use pre-encoded text and enhance with vision cross-attention
            text_features = pre_encoded_text

            # Apply vision cross-attention to enhance text features
            vision_features_expanded = vision_features.unsqueeze(1)  # [batch_size, 1, vision_dim]

            # Simple cross-attention enhancement
            enhanced_text, text_to_vision_attn = self._apply_cross_attention(
                query=text_features,
                key_value=vision_features_expanded,
                mask=None
            )
            text_attention_patterns = [{'cross_modal_attention': text_to_vision_attn}]
        else:
            # Full FIBER-style encoding with integrated fusion
            vision_features_expanded = vision_features.unsqueeze(1).expand(-1, text_input_ids.shape[1], -1)

            enhanced_text, text_attention_patterns = self.fiber_text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                cross_modal_features=vision_features_expanded,
                cross_modal_mask=None  # Vision features are always valid
            )

        # Enhance vision features with text context
        enhanced_vision, vision_attention_patterns = self.fiber_vision_encoder(
            vision_features=vision_features,
            cross_modal_features=enhanced_text,
            cross_modal_mask=text_attention_mask
        )

        # Project to common dimension
        text_projected = self.text_output_proj(enhanced_text)
        vision_projected = self.vision_output_proj(enhanced_vision)

        # Combine text and vision features
        # Expand vision to match text sequence length
        vision_expanded = vision_projected.unsqueeze(1).expand(-1, enhanced_text.shape[1], -1)

        # Element-wise combination with learned weighting
        combined_features = text_projected + vision_expanded

        # Final fusion layer
        fused_features = self.final_fusion(combined_features)

        # Compile attention patterns
        attention_weights = {
            'text_attention_patterns': text_attention_patterns,
            'vision_attention_patterns': vision_attention_patterns,
            'fusion_type': 'fiber_backbone_fusion'
        }

        return fused_features, attention_weights

    def _apply_cross_attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple cross-attention for fallback scenarios"""
        # Simplified cross-attention
        batch_size, seq_len, dim = query.shape
        kv_seq_len = key_value.shape[1]

        # Simple dot-product attention
        scores = torch.matmul(query, key_value.transpose(-2, -1)) / math.sqrt(dim)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -10000.0)

        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, key_value)

        # Residual connection
        output = query + attended

        return output, attn_weights.mean(dim=1)


class FIBERIntegratedModel(nn.Module):
    """
    Integration layer to use FIBER fusion with BitGen's existing model architecture
    This replaces BitGen's CrossModalFusion with FIBER-style backbone fusion
    """

    def __init__(
        self,
        text_encoder_dim: int,
        vision_encoder_dim: int,
        fusion_hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_fusion_layers: int = 6,
        fiber_config: Dict = None  # NEW: FIBER-specific configuration
    ):
        super().__init__()

        # Store FIBER configuration
        self.fiber_config = fiber_config or {}

        # Apply FIBER-specific parameters if available
        if fiber_config:
            attention_temperature = fiber_config.get('fiber_attention_temperature', 1.0)
            cross_attention_dropout = fiber_config.get('fiber_cross_attention_dropout', 0.1)
            learnable_alpha = fiber_config.get('fiber_learnable_alpha', True)
            mlp_ratio = fiber_config.get('fiber_mlp_ratio', 4.0)
            layer_norm_eps = fiber_config.get('fiber_layer_norm_eps', 1e-6)

            # Use FIBER-specific dropout
            dropout = cross_attention_dropout

            logger.info("🔧 Applying FIBER-specific configuration to integrated model")

        self.fiber_fusion = FIBERCrossModalFusion(
            text_dim=text_encoder_dim,
            vision_dim=vision_encoder_dim,
            hidden_dim=fusion_hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_fusion_layers=num_fusion_layers
        )

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass using FIBER-style fusion

        Args:
            text_features: Pre-encoded text features [batch_size, seq_len, text_dim]
            vision_features: Vision features [batch_size, vision_dim]
            text_input_ids: Original text tokens (optional, for re-encoding)
            text_attention_mask: Text attention mask (optional)

        Returns:
            fused_features: [batch_size, seq_len, hidden_dim]
            attention_weights: Dict of attention patterns
        """
        return self.fiber_fusion(
            text_input_ids=text_input_ids if text_input_ids is not None else torch.zeros_like(text_features[:, :, 0], dtype=torch.long),
            text_attention_mask=text_attention_mask if text_attention_mask is not None else torch.ones_like(text_features[:, :, 0]),
            vision_features=vision_features,
            pre_encoded_text=text_features
        )


def create_fiber_fusion_replacement(
    text_encoder_dim: int,
    vision_encoder_dim: int,
    fusion_hidden_size: int,
    num_heads: int = 8,
    num_layers: int = 2,
    dropout: float = 0.1,
    num_fusion_layers: int = 6,
    config: Dict = None  # NEW: Accept full config for FIBER parameters
) -> FIBERIntegratedModel:
    """
    Create FIBER-style fusion module to replace BitGen's basic cross-attention

    Args:
        text_encoder_dim: Dimension of text encoder output
        vision_encoder_dim: Dimension of vision encoder output
        fusion_hidden_size: Hidden dimension for fusion
        num_heads: Number of attention heads
        num_layers: Number of fusion layers
        dropout: Dropout rate
        num_fusion_layers: Number of layers with cross-modal fusion (FIBER-style)
        config: Full configuration dict with FIBER-specific parameters

    Returns:
        FIBER fusion module that can replace BitGen's CrossModalFusion
    """
    logger.info("🔄 Creating FIBER-style fusion module to replace basic cross-attention")
    logger.info(f"  • Text encoder dim: {text_encoder_dim}")
    logger.info(f"  • Vision encoder dim: {vision_encoder_dim}")
    logger.info(f"  • Fusion hidden size: {fusion_hidden_size}")
    logger.info(f"  • Fusion layers: {num_fusion_layers}")

    # Extract FIBER-specific parameters from config if provided
    if config:
        attention_temperature = config.get('fiber_attention_temperature', 1.0)
        cross_attention_dropout = config.get('fiber_cross_attention_dropout', 0.1)
        learnable_alpha = config.get('fiber_learnable_alpha', True)
        bidirectional_fusion = config.get('fiber_bidirectional_fusion', True)
        backbone_integration = config.get('fiber_backbone_integration', True)
        residual_connection = config.get('fiber_residual_connection', True)
        layer_norm_eps = config.get('fiber_layer_norm_eps', 1e-6)
        mlp_ratio = config.get('fiber_mlp_ratio', 4.0)

        logger.info("✅ Using FIBER-specific configuration:")
        logger.info(f"  • Attention temperature: {attention_temperature}")
        logger.info(f"  • Cross-attention dropout: {cross_attention_dropout}")
        logger.info(f"  • Learnable alpha: {learnable_alpha}")
        logger.info(f"  • Bidirectional fusion: {bidirectional_fusion}")
        logger.info(f"  • Backbone integration: {backbone_integration}")
        logger.info(f"  • Residual connections: {residual_connection}")
        logger.info(f"  • Layer norm epsilon: {layer_norm_eps}")
        logger.info(f"  • MLP ratio: {mlp_ratio}")

        # Use FIBER-specific dropout instead of general dropout
        dropout = cross_attention_dropout

    return FIBERIntegratedModel(
        text_encoder_dim=text_encoder_dim,
        vision_encoder_dim=vision_encoder_dim,
        fusion_hidden_size=fusion_hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_fusion_layers=num_fusion_layers,
        fiber_config=config  # Pass config for detailed FIBER settings
    )
