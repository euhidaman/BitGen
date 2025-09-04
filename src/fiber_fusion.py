"""
FIBER Integration for BitGen
Based on the FIBER implementation from Microsoft Research
Implements coarse-to-fine vision-language pre-training with fusion in the backbone

ENHANCED with AUTHENTIC FIBER loss objectives:
- ITC (Image-Text Contrastive) with momentum queue
- ITM (Image-Text Matching) with hard negatives  
- MLM (Masked Language Modeling) integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AuthenticFIBERLoss(nn.Module):
    """
    Authentic FIBER loss computation following the original Microsoft Research paper
    Implements ITC + ITM + MLM objectives with proper negative mining
    """
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        queue_size: int = 4096,
        momentum: float = 0.995,
        temperature: float = 0.07,
        alpha_itc: float = 1.0,
        alpha_itm: float = 1.0,
        alpha_mlm: float = 1.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.queue_size = queue_size
        self.momentum = momentum
        self.alpha_itc = alpha_itc
        self.alpha_itm = alpha_itm
        self.alpha_mlm = alpha_mlm
        
        # Learnable temperature for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
        # Momentum queue for negative mining (like ALBEF)
        self.register_buffer("image_queue", torch.randn(hidden_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(hidden_dim, queue_size)) 
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Normalize queues
        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)
        
        # Projection heads for contrastive learning
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.vision_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ITM head for image-text matching
        self.itm_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # match vs no-match
        )
        
        # MLM head for masked language modeling
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat: torch.Tensor, text_feat: torch.Tensor):
        """Update momentum queue with current batch features"""
        batch_size = image_feat.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the features at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue_size:
            self.image_queue[:, ptr:ptr + batch_size] = image_feat.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feat.T
        else:
            # Wrap around if queue is full
            remaining = self.queue_size - ptr
            self.image_queue[:, ptr:] = image_feat[:remaining].T
            self.text_queue[:, ptr:] = text_feat[:remaining].T
            
            overflow = batch_size - remaining
            self.image_queue[:, :overflow] = image_feat[remaining:].T
            self.text_queue[:, :overflow] = text_feat[remaining:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def compute_itc_loss(
        self, 
        text_features: torch.Tensor, 
        vision_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Image-Text Contrastive loss with momentum queue"""
        # Project features for contrastive learning
        text_proj_feat = F.normalize(self.text_proj(text_features), dim=-1)
        vision_proj_feat = F.normalize(self.vision_proj(vision_features), dim=-1)
        
        # Concatenate with momentum queue
        text_feat_all = torch.cat([text_proj_feat.t(), self.text_queue.detach()], dim=1)
        vision_feat_all = torch.cat([vision_proj_feat.t(), self.image_queue.detach()], dim=1)
        
        # Compute similarity matrices
        sim_i2t = vision_proj_feat @ text_feat_all / self.temperature
        sim_t2i = text_proj_feat @ vision_feat_all / self.temperature
        
        # Create targets (positive pairs are on the diagonal)
        batch_size = text_proj_feat.shape[0]
        targets = torch.arange(batch_size, device=text_proj_feat.device)
        
        # Compute contrastive losses
        loss_i2t = F.cross_entropy(sim_i2t, targets)
        loss_t2i = F.cross_entropy(sim_t2i, targets)
        
        itc_loss = (loss_i2t + loss_t2i) / 2
        
        # Update momentum queue
        self._dequeue_and_enqueue(vision_proj_feat.detach(), text_proj_feat.detach())
        
        return itc_loss, text_proj_feat, vision_proj_feat

    def compute_itm_loss(
        self,
        fused_features: torch.Tensor,
        text_proj_feat: torch.Tensor,
        vision_proj_feat: torch.Tensor
    ) -> torch.Tensor:
        """Compute Image-Text Matching loss with hard negatives"""
        batch_size = fused_features.shape[0]
        
        # Create hard negatives using similarity scores
        with torch.no_grad():
            sim_matrix = text_proj_feat @ vision_proj_feat.t()
            weights_t2i = F.softmax(sim_matrix, dim=1)
            weights_i2t = F.softmax(sim_matrix.t(), dim=1)
            weights_t2i.fill_diagonal_(0)
            weights_i2t.fill_diagonal_(0)
            
        # Sample negative pairs
        neg_idx_t2i = torch.multinomial(weights_t2i, 1).squeeze()
        neg_idx_i2t = torch.multinomial(weights_i2t, 1).squeeze()
        
        # Create positive and negative pairs
        pos_features = torch.cat([fused_features.mean(dim=1), fused_features.mean(dim=1)], dim=-1)
        pos_labels = torch.ones(batch_size, device=fused_features.device, dtype=torch.long)
        
        # Negative pairs
        neg_features = torch.cat([
            fused_features.mean(dim=1), 
            fused_features[neg_idx_t2i % batch_size].mean(dim=1)
        ], dim=-1)
        neg_labels = torch.zeros(batch_size, device=fused_features.device, dtype=torch.long)
        
        # Combine positive and negative pairs
        all_features = torch.cat([pos_features, neg_features], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # Compute ITM loss
        itm_logits = self.itm_head(all_features)
        itm_loss = F.cross_entropy(itm_logits, all_labels)
        
        return itm_loss

    def compute_mlm_loss(
        self,
        fused_features: torch.Tensor,
        input_ids: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Masked Language Modeling loss"""
        if masked_positions is None:
            # Create random mask (15% of tokens)
            mask_prob = 0.15
            masked_positions = torch.rand(input_ids.shape, device=input_ids.device) < mask_prob
            # Don't mask special tokens
            masked_positions &= (input_ids != 0)  # Not padding
            masked_positions &= (input_ids != 1)  # Not start token
            masked_positions &= (input_ids != 2)  # Not end token
        
        # Get MLM predictions
        mlm_logits = self.mlm_head(fused_features)
        
        # Create labels (-100 for non-masked positions)
        mlm_labels = input_ids.clone()
        mlm_labels[~masked_positions.bool()] = -100
        
        mlm_loss = F.cross_entropy(mlm_logits.view(-1, self.vocab_size), mlm_labels.view(-1), ignore_index=-100)
        
        return mlm_loss

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor, 
        fused_features: torch.Tensor,
        input_ids: torch.Tensor,
        masked_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute all FIBER losses: ITC + ITM + MLM"""
        losses = {}
        
        # 1. Image-Text Contrastive Loss
        itc_loss, text_proj_feat, vision_proj_feat = self.compute_itc_loss(
            text_features.mean(dim=1),  # Pool text features
            vision_features
        )
        losses['itc_loss'] = self.alpha_itc * itc_loss
        
        # 2. Image-Text Matching Loss  
        itm_loss = self.compute_itm_loss(fused_features, text_proj_feat, vision_proj_feat)
        losses['itm_loss'] = self.alpha_itm * itm_loss
        
        # 3. Masked Language Modeling Loss
        mlm_loss = self.compute_mlm_loss(fused_features, input_ids, masked_positions)
        losses['mlm_loss'] = self.alpha_mlm * mlm_loss
        
        # Total FIBER loss
        total_loss = losses['itc_loss'] + losses['itm_loss'] + losses['mlm_loss']
        losses['fiber_total_loss'] = total_loss
        
        # Add individual components for logging
        losses['itc_loss_raw'] = itc_loss
        losses['itm_loss_raw'] = itm_loss
        losses['mlm_loss_raw'] = mlm_loss
        losses['temperature'] = self.temperature
        
        return losses


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
        # Debug: transpose_for_scores input: {x.shape}
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # Debug: new_x_shape: {new_x_shape}
        x = x.view(*new_x_shape)
        # Debug: after view: {x.shape}
        result = x.permute(0, 2, 1, 3)
        # Debug: transpose_for_scores output: {result.shape}
        return result

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
        # Debug: FIBERCrossModalLayer - Starting forward
        # Debug: Input shapes: vision={vision_hidden_states.shape}, text={text_hidden_states.shape}
        
        batch_size = vision_hidden_states.shape[0]

        # Vision attending to text
        # Debug: Vision-to-text attention - Starting
        
        try:
            vision_query_layer = self.transpose_for_scores(self.vision_to_text_query(vision_hidden_states))
            text_key_layer = self.transpose_for_scores(self.text_to_vision_key(text_hidden_states))
            text_value_layer = self.transpose_for_scores(self.text_to_vision_value(text_hidden_states))
            # Debug: V2T projections successful: query={vision_query_layer.shape}, key={text_key_layer.shape}, value={text_value_layer.shape}
        except Exception as e:
            print(f"❌ V2T projections failed: {e}")
            raise e

        # Compute vision-to-text attention scores
        try:
            # Debug: V2T attention matrix multiplication:
            # Debug: Vision query: {vision_query_layer.shape}
            # Debug: Text key transposed: {text_key_layer.transpose(-1, -2).shape}
            
            # The issue: vision_query has different seq_len than text_key
            # vision_query: [batch, heads, vision_seq_len, head_dim]
            # text_key: [batch, heads, text_seq_len, head_dim]
            # When we do matmul(query, key.T) we get [batch, heads, vision_seq_len, text_seq_len]
            # This should be valid - the error must be elsewhere
            
            v2t_attention_scores = torch.matmul(vision_query_layer, text_key_layer.transpose(-1, -2))
            v2t_attention_scores = v2t_attention_scores / math.sqrt(self.attention_head_size)
            # Debug: V2T attention scores computed successfully: {v2t_attention_scores.shape}
        except Exception as e:
            print(f"❌ V2T attention scores failed: {e}")
            print(f"   Vision query: {vision_query_layer.shape}")
            print(f"   Text key: {text_key_layer.shape}")
            print(f"   Vision query device: {vision_query_layer.device}")
            print(f"   Text key device: {text_key_layer.device}")
            print(f"   Vision query dtype: {vision_query_layer.dtype}")
            print(f"   Text key dtype: {text_key_layer.dtype}")
            raise e

        if text_attention_mask is not None:
            # Debug: Applying text attention mask: {text_attention_mask.shape}
            # Apply text attention mask to vision-to-text attention
            v2t_attention_scores = v2t_attention_scores + text_attention_mask.unsqueeze(1).unsqueeze(1)

        v2t_attention_probs = F.softmax(v2t_attention_scores, dim=-1)
        v2t_attention_probs = self.dropout(v2t_attention_probs)

        # Vision enhanced by text
        try:
            v2t_context_layer = torch.matmul(v2t_attention_probs, text_value_layer)
            v2t_context_layer = v2t_context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = v2t_context_layer.size()[:-2] + (self.all_head_size,)
            v2t_context_layer = v2t_context_layer.view(*new_context_layer_shape)
            # Debug: V2T context layer computed: {v2t_context_layer.shape}
        except Exception as e:
            print(f"❌ V2T context layer failed: {e}")
            raise e

        # Text attending to vision
        # Debug: Text-to-vision attention - Starting
        
        try:
            text_query_layer = self.transpose_for_scores(self.text_to_vision_query(text_hidden_states))
            vision_key_layer = self.transpose_for_scores(self.vision_to_text_key(vision_hidden_states))
            vision_value_layer = self.transpose_for_scores(self.vision_to_text_value(vision_hidden_states))
            # Debug: T2V projections successful: query={text_query_layer.shape}, key={vision_key_layer.shape}, value={vision_value_layer.shape}
        except Exception as e:
            print(f"❌ T2V projections failed: {e}")
            raise e

        # Compute text-to-vision attention scores
        try:
            # Debug: T2V attention matrix multiplication:
            # Debug: Text query: {text_query_layer.shape}  
            # Debug: Vision key transposed: {vision_key_layer.transpose(-1, -2).shape}
            
            # Same logic: text_query and vision_key can have different seq_lens
            # text_query: [batch, heads, text_seq_len, head_dim] 
            # vision_key: [batch, heads, vision_seq_len, head_dim]
            # Result: [batch, heads, text_seq_len, vision_seq_len]
            
            t2v_attention_scores = torch.matmul(text_query_layer, vision_key_layer.transpose(-1, -2))
            t2v_attention_scores = t2v_attention_scores / math.sqrt(self.attention_head_size)
            # Debug: T2V attention scores computed successfully: {t2v_attention_scores.shape}
        except Exception as e:
            print(f"❌ T2V attention scores failed: {e}")
            print(f"   Text query: {text_query_layer.shape}")
            print(f"   Vision key: {vision_key_layer.shape}")
            print(f"   Text query device: {text_query_layer.device}")
            print(f"   Vision key device: {vision_key_layer.device}")
            print(f"   Text query dtype: {text_query_layer.dtype}")
            print(f"   Vision key dtype: {vision_key_layer.dtype}")
            raise e

        if vision_attention_mask is not None:
            # Debug: Applying vision attention mask: {vision_attention_mask.shape}
            # Apply vision attention mask to text-to-vision attention
            t2v_attention_scores = t2v_attention_scores + vision_attention_mask.unsqueeze(1).unsqueeze(1)

        t2v_attention_probs = F.softmax(t2v_attention_scores, dim=-1)
        t2v_attention_probs = self.dropout(t2v_attention_probs)

        # Text enhanced by vision
        try:
            t2v_context_layer = torch.matmul(t2v_attention_probs, vision_value_layer)
            t2v_context_layer = t2v_context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = t2v_context_layer.size()[:-2] + (self.all_head_size,)
            t2v_context_layer = t2v_context_layer.view(*new_context_layer_shape)
            # Debug: T2V context layer computed: {t2v_context_layer.shape}
        except Exception as e:
            print(f"❌ T2V context layer failed: {e}")
            raise e

        # Apply output transformations and residual connections
        # Debug: Applying output transformations
        
        try:
            enhanced_vision = self.vision_output(v2t_context_layer)
            enhanced_vision = self.vision_dropout(enhanced_vision)
            enhanced_vision = self.vision_LayerNorm(enhanced_vision + vision_hidden_states)
            # Debug: Enhanced vision computed: {enhanced_vision.shape}
        except Exception as e:
            print(f"❌ Enhanced vision computation failed: {e}")
            print(f"   V2T context: {v2t_context_layer.shape}")
            print(f"   Original vision: {vision_hidden_states.shape}")
            raise e

        try:
            enhanced_text = self.text_output(t2v_context_layer)
            enhanced_text = self.text_dropout(enhanced_text)
            enhanced_text = self.text_LayerNorm(enhanced_text + text_hidden_states)
            # Debug: Enhanced text computed: {enhanced_text.shape}
        except Exception as e:
            print(f"❌ Enhanced text computation failed: {e}")
            print(f"   T2V context: {t2v_context_layer.shape}")
            print(f"   Original text: {text_hidden_states.shape}")
            raise e

        outputs = (enhanced_vision, enhanced_text)
        if output_attentions:
            outputs += (v2t_attention_probs, t2v_attention_probs)

        # Debug: FIBERCrossModalLayer - Forward completed successfully
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
        # Debug: FIBERTransformerBlock - Starting forward
        # Debug: Input shapes: vision={vision_hidden_states.shape}, text={text_hidden_states.shape}
        
        attention_outputs = {}

        # Self-attention for vision
        # Debug: Vision self-attention - Starting
        vision_normed = self.vision_norm1(vision_hidden_states)
        # Debug: Vision normed shape: {vision_normed.shape}
        
        # Handle vision attention mask - ensure it matches vision sequence length
        vision_key_padding_mask = None
        if vision_attention_mask is not None:
            # Debug removed
            # Ensure vision_attention_mask has the right shape for vision sequence
            vision_seq_len = vision_hidden_states.shape[1]
            if vision_attention_mask.shape[1] != vision_seq_len:
                # Debug removed
                # Create appropriate mask for vision sequence length
                vision_key_padding_mask = torch.zeros(
                    vision_hidden_states.shape[0], vision_seq_len,
                    dtype=torch.bool, device=vision_hidden_states.device
                )
                # Debug removed
            else:
                # Convert to boolean before using ~ operator
                vision_key_padding_mask = ~vision_attention_mask.bool()
                # Debug removed
        
        try:
            vision_self_output, vision_self_weights = self.vision_attention(
                vision_normed, vision_normed, vision_normed,
                key_padding_mask=vision_key_padding_mask,
                need_weights=output_attentions
            )
            # Debug removed
        except Exception as e:
            print(f"❌ Vision self-attention failed: {e}")
            print(f"   Vision normed shape: {vision_normed.shape}")
            print(f"   Vision key padding mask: {vision_key_padding_mask.shape if vision_key_padding_mask is not None else None}")
            raise e
        
        vision_hidden_states = vision_hidden_states + vision_self_output
        if output_attentions:
            attention_outputs['vision_self_attention'] = vision_self_weights

        # Self-attention for text
        # Debug removed
        text_normed = self.text_norm1(text_hidden_states)
        # Debug removed
        
        # Handle text attention mask - ensure dimensions are correct
        text_key_padding_mask = None
        if text_attention_mask is not None:
            # Debug removed
            # Debug removed
            # text_attention_mask should be [batch_size, seq_len]
            if text_attention_mask.shape != text_hidden_states.shape[:2]:
                print(f"❌ Text attention mask shape mismatch!")
                raise ValueError(f"Text attention mask shape {text_attention_mask.shape} doesn't match text features {text_hidden_states.shape[:2]}")
            # Convert to boolean before using ~ operator
            text_key_padding_mask = ~text_attention_mask.bool()
            # Debug removed
            
        try:
            text_self_output, text_self_weights = self.text_attention(
                text_normed, text_normed, text_normed,
                key_padding_mask=text_key_padding_mask,
                need_weights=output_attentions
            )
            # Debug removed
        except Exception as e:
            print(f"❌ Text self-attention failed: {e}")
            print(f"   Text normed shape: {text_normed.shape}")
            print(f"   Text key padding mask: {text_key_padding_mask.shape if text_key_padding_mask is not None else None}")
            raise e
            
        text_hidden_states = text_hidden_states + text_self_output
        if output_attentions:
            attention_outputs['text_self_attention'] = text_self_weights

        # Cross-modal attention (FIBER's core innovation)
        # Debug removed
        if self.enable_cross_modal:
            # Debug removed
            # Debug removed
            
            try:
                cross_modal_outputs = self.cross_modal_layer(
                    vision_hidden_states=vision_hidden_states,
                    text_hidden_states=text_hidden_states,
                    vision_attention_mask=vision_attention_mask,
                    text_attention_mask=text_attention_mask,
                    output_attentions=output_attentions
                )
                # Debug removed
            except Exception as e:
                print(f"❌ Cross-modal layer failed: {e}")
                print(f"   Vision input shape: {vision_hidden_states.shape}")
                print(f"   Text input shape: {text_hidden_states.shape}")
                raise e

            vision_hidden_states = cross_modal_outputs[0]
            text_hidden_states = cross_modal_outputs[1]
            # Debug removed

            if output_attentions:
                attention_outputs['vision_to_text_attention'] = cross_modal_outputs[2]
                attention_outputs['text_to_vision_attention'] = cross_modal_outputs[3]

        # Feed-forward for vision
        # Debug removed
        try:
            vision_intermediate_output = self.vision_intermediate(vision_hidden_states)
            vision_intermediate_output = self.activation(vision_intermediate_output)
            vision_layer_output = self.vision_output(vision_intermediate_output)
            vision_layer_output = self.dropout(vision_layer_output)
            vision_hidden_states = self.vision_norm2(vision_layer_output + vision_hidden_states)
            # Debug removed
        except Exception as e:
            print(f"❌ Vision feed-forward failed: {e}")
            raise e

        # Feed-forward for text
        # Debug removed
        try:
            text_intermediate_output = self.text_intermediate(text_hidden_states)
            text_intermediate_output = self.activation(text_intermediate_output)
            text_layer_output = self.text_output(text_intermediate_output)
            text_layer_output = self.dropout(text_layer_output)
            text_hidden_states = self.text_norm2(text_layer_output + text_hidden_states)
            # Debug removed
        except Exception as e:
            print(f"❌ Text feed-forward failed: {e}")
            raise e

        # Debug removed
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

        # Only log essential FIBER initialization info
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"🔥 FIBER Fusion: {text_dim}→{hidden_dim}, {num_fuse_layers}/{num_layers} fusion layers")

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

        if logger.isEnabledFor(logging.INFO):
            logger.info(f"✅ FIBER Fusion ready: {num_fuse_layers}/{num_layers} layers")

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
        import logging
        logger = logging.getLogger(__name__)
        
        # Debug logging removed for cleaner output
        # Input shapes: text_features: {text_features.shape}, vision_features: {vision_features.shape}
        
        batch_size = text_features.shape[0]
        text_seq_len = text_features.shape[1]

        # Handle vision features - convert to sequence format if needed
        if vision_features.dim() == 2:  # [batch_size, vision_dim]
            vision_features = vision_features.unsqueeze(1)  # [batch_size, 1, vision_dim]
            vision_seq_len = 1
            # Vision converted to sequence format: {vision_features.shape}
        else:  # [batch_size, vision_seq_len, vision_dim]
            vision_seq_len = vision_features.shape[1]

        # Project to common hidden dimension
        try:
            text_hidden = self.text_projection(text_features)
            # Text projection: {text_hidden.shape}
        except Exception as e:
            logger.error(f"❌ FIBER text projection failed: {e}")
            raise e
            
        try:
            vision_hidden = self.vision_projection(vision_features)
            # Vision projection: {vision_hidden.shape}
        except Exception as e:
            logger.error(f"❌ FIBER vision projection failed: {e}")
            raise e

        # Store attention patterns
        all_attention_patterns = {
            'vision_self_attention': [],
            'text_self_attention': [],
            'vision_to_text_attention': [],
            'text_to_vision_attention': []
        }

        # Pass through FIBER transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Debug removed
            # Debug removed
            # FIBER Layer {layer_idx} processing...
            try:
                vision_hidden, text_hidden, attention_patterns = layer(
                    vision_hidden_states=vision_hidden,
                    text_hidden_states=text_hidden,
                    vision_attention_mask=vision_attention_mask,
                    text_attention_mask=text_attention_mask,
                    output_attentions=output_attentions
                )
                # Debug removed
                # FIBER Layer {layer_idx} completed
            except Exception as e:
                print(f"❌ FIBER Layer {layer_idx} failed: {e}")
                print(f"   Vision input shape: {vision_hidden.shape}")
                print(f"   Text input shape: {text_hidden.shape}")
                print(f"   Vision mask: {vision_attention_mask.shape if vision_attention_mask is not None else None}")
                print(f"   Text mask: {text_attention_mask.shape if text_attention_mask is not None else None}")
                logger.error(f"❌ FIBER Layer {layer_idx} failed: {e}")
                logger.error(f"   Vision input shape: {vision_hidden.shape}")
                logger.error(f"   Text input shape: {text_hidden.shape}")
                logger.error(f"   Vision mask: {vision_attention_mask.shape if vision_attention_mask is not None else None}")
                logger.error(f"   Text mask: {text_attention_mask.shape if text_attention_mask is not None else None}")
                raise e

            if output_attentions:
                for key, value in attention_patterns.items():
                    if key in all_attention_patterns:
                        all_attention_patterns[key].append(value)

        # Debug removed

        # Final normalization
        # Debug removed
        try:
            vision_hidden = self.vision_final_norm(vision_hidden)
            text_hidden = self.text_final_norm(text_hidden)
            # Debug removed
        except Exception as e:
            print(f"❌ FIBER - Final normalization failed: {e}")
            raise e

        # Output projections
        # Debug removed
        try:
            enhanced_vision = self.vision_output_projection(vision_hidden)
            enhanced_text = self.text_output_projection(text_hidden)
            # Debug removed
        except Exception as e:
            print(f"❌ FIBER - Output projection failed: {e}")
            raise e

        # Debug removed
        return enhanced_text, enhanced_vision, all_attention_patterns


class FIBERIntegration(nn.Module):
    """
    AUTHENTIC FIBER Integration module based on Microsoft Research's implementation
    Provides authentic coarse-to-fine vision-language pre-training with fusion in the backbone
    
    Includes all three FIBER loss objectives:
    1. ITC (Image-Text Contrastive) with momentum queue
    2. ITM (Image-Text Matching) with hard negatives  
    3. MLM (Masked Language Modeling) integration
    """

    def __init__(
        self,
        text_encoder_dim: int,
        vision_encoder_dim: int,
        fusion_hidden_size: int,
        vocab_size: int = 50257,  # GPT-2 vocab size
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
        self.vocab_size = vocab_size

        # Apply FIBER-specific configuration if provided
        if fiber_config:
            num_fuse_layers = fiber_config.get('num_fiber_fusion_layers', num_fuse_layers)
            attention_temperature = fiber_config.get('fiber_attention_temperature', 1.0)
            cross_attention_dropout = fiber_config.get('fiber_cross_attention_dropout', dropout)

            # AUTHENTIC FIBER configuration applied
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"🔥 AUTHENTIC FIBER: {num_fuse_layers}/{num_layers} layers, temp={attention_temperature}")

        # Create FIBER fusion backbone
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

        # AUTHENTIC FIBER Loss computation
        self.fiber_loss = AuthenticFIBERLoss(
            hidden_dim=fusion_hidden_size,
            vocab_size=vocab_size,
            queue_size=fiber_config.get('fiber_queue_size', 4096) if fiber_config else 4096,
            momentum=fiber_config.get('fiber_momentum', 0.995) if fiber_config else 0.995,
            temperature=fiber_config.get('fiber_temperature', 0.07) if fiber_config else 0.07,
            alpha_itc=fiber_config.get('alpha_itc', 1.0) if fiber_config else 1.0,
            alpha_itm=fiber_config.get('alpha_itm', 1.0) if fiber_config else 1.0,
            alpha_mlm=fiber_config.get('alpha_mlm', 1.0) if fiber_config else 1.0
        )

        logger.info("🔥 AUTHENTIC FIBER integration initialized with ITC+ITM+MLM losses")

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_losses: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with authentic FIBER cross-modal learning

        Args:
            text_features: [batch_size, seq_len, text_dim]
            vision_features: [batch_size, vision_dim]
            input_ids: [batch_size, seq_len] - needed for MLM
            text_attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - for MLM loss
            compute_losses: Whether to compute FIBER losses

        Returns:
            fused_features: [batch_size, seq_len, hidden_dim]
            outputs: Dict with attention patterns and losses
        """
        # 1. Deep backbone fusion
        enhanced_text, enhanced_vision, attention_patterns = self.fiber_fusion(
            text_features=text_features,
            vision_features=vision_features,
            text_attention_mask=text_attention_mask,
            output_attentions=True
        )

        # 2. Return enhanced text features as the main output
        fused_features = enhanced_text

        # 3. Compile outputs
        outputs = {
            'fiber_attention_patterns': attention_patterns,
            'fusion_type': 'authentic_fiber_backbone_fusion',
            'enhanced_vision': enhanced_vision,
            'enhanced_text': enhanced_text
        }

        # 4. Compute FIBER losses if requested
        if compute_losses and input_ids is not None:
            fiber_losses = self.fiber_loss(
                text_features=text_features,
                vision_features=enhanced_vision if enhanced_vision.dim() == 2 else enhanced_vision.mean(dim=1),
                fused_features=fused_features,
                input_ids=input_ids
            )
            outputs.update(fiber_losses)
            
            # Log loss components for monitoring
            logger.debug(f"FIBER Losses - ITC: {fiber_losses['itc_loss_raw']:.4f}, "
                        f"ITM: {fiber_losses['itm_loss_raw']:.4f}, "
                        f"MLM: {fiber_losses['mlm_loss_raw']:.4f}")

        return fused_features, outputs


def create_fiber_fusion(
    text_encoder_dim: int,
    vision_encoder_dim: int,
    fusion_hidden_size: int,
    vocab_size: int = 50257,  # GPT-2 vocab size
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
    num_fusion_layers: int = 6,
    config: Dict = None
) -> FIBERIntegration:
    """
    Create AUTHENTIC FIBER fusion module based on Microsoft Research's implementation
    Now includes all three FIBER loss objectives: ITC + ITM + MLM
    """
    logger.info("🔥 Creating AUTHENTIC FIBER fusion with complete loss objectives")
    logger.info("   Based on Microsoft Research's FIBER implementation")
    logger.info(f"  • Text encoder dim: {text_encoder_dim}")
    logger.info(f"  • Vision encoder dim: {vision_encoder_dim}")
    logger.info(f"  • Fusion hidden size: {fusion_hidden_size}")
    logger.info(f"  • Vocab size: {vocab_size}")
    logger.info(f"  • Fusion layers: {num_fusion_layers}")

    return FIBERIntegration(
        text_encoder_dim=text_encoder_dim,
        vision_encoder_dim=vision_encoder_dim,
        fusion_hidden_size=fusion_hidden_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_layers=num_layers,
        num_fuse_layers=num_fusion_layers,
        dropout=dropout,
        fiber_config=config
    )

# Keep compatibility with old naming for now
create_authentic_fiber_fusion = create_fiber_fusion
AuthenticFIBERIntegration = FIBERIntegration
