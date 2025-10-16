"""
FIBER-style Cross-Modal Fusion with Queue-Based Contrastive Learning
Adapted from FIBER for BitGen Stage 1 (Vision-Language Pre-training)

Architecture:
- DINOv2-base (frozen) + LoRA adapters (trainable)
- Feature compression (768→128)
- Queue-based contrastive learning (ITC)
- Hard negative sampling for ITM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model


class FIBERCrossModalFusion(nn.Module):
    """
    FIBER-inspired cross-modal fusion with queue-based contrastive learning
    
    Architecture:
    - DINOv2-base (frozen 300M) + LoRA adapters (trainable ~2M)
    - Feature compression: 768→384→128 (trainable ~300K)
    - Queue-based ITC loss (4096 samples)
    - Hard negative ITM loss
    
    Separate transforms for fusion and contrastive learning (FIBER approach)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.vision_embed_dim = config.vision_embed_dim
        
        # DINOv2 Vision Encoder with LoRA
        from transformers import Dinov2Model
        print("🔧 Loading DINOv2-base with LoRA adapters...")
        print("   - Base model: 300M params (frozen)")
        print("   - LoRA adapters: ~2M params (trainable)")
        
        # Load base DINOv2 model
        self.dinov2_model = Dinov2Model.from_pretrained(
            'facebook/dinov2-base',
            cache_dir='./cache'
        )
        
        # Freeze all base parameters
        for param in self.dinov2_model.parameters():
            param.requires_grad = False
        
        # Add LoRA adapters (only to attention layers)
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling
            target_modules=["query", "value"],  # Apply to Q,V in attention
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[]  # Don't train any other modules
        )
        
        # Apply LoRA to DINOv2
        self.dinov2_model = get_peft_model(self.dinov2_model, lora_config)
        self.dinov2_model.print_trainable_parameters()
        
        # Feature Compression: 768 → 384 → 128 (BitMar-style learned compression)
        print("🔧 Adding feature compression layer (768→128)...")
        self.dinov2_dim = 768  # DINOv2-base output dimension
        intermediate_dim = 384
        
        self.feature_compressor = nn.Sequential(
            nn.Linear(self.dinov2_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, config.vision_embed_dim),
            nn.LayerNorm(config.vision_embed_dim)
        )
        
        # Initialize compression with small weights for stability
        for module in self.feature_compressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # FIBER-style cross-modal transforms (for fusion)
        self.cross_modal_text_transform = nn.Linear(config.embed_dim, config.embed_dim)
        self.cross_modal_image_transform = nn.Linear(config.vision_embed_dim, config.embed_dim)
        
        # ITC transforms (for contrastive learning - SEPARATE!)
        self.cross_modal_text_transform_itc = nn.Linear(config.embed_dim, config.embed_dim)
        self.cross_modal_image_transform_itc = nn.Linear(config.vision_embed_dim, config.embed_dim)
        
        # FIBER-style poolers for fusion
        self.cross_modal_text_pooler = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )
        self.cross_modal_image_pooler = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )
        
        # ITC poolers for contrastive learning (SEPARATE!)
        self.cross_modal_text_pooler_itc = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )
        self.cross_modal_image_pooler_itc = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )
        
        # Queue-based contrastive learning (FIBER approach)
        queue_size = 4096
        self.register_buffer("image_queue", torch.randn(config.embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(config.embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Normalize queues
        self.image_queue = F.normalize(self.image_queue, p=2, dim=0)
        self.text_queue = F.normalize(self.text_queue, p=2, dim=0)
        
        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.ffn_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.ffn_dim, config.embed_dim)
        )
        
        # Average pooling for image features
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_keys: torch.Tensor, text_keys: torch.Tensor):
        """
        Update queues with new keys (FIBER approach)
        
        Args:
            image_keys: [batch_size, embed_dim]
            text_keys: [batch_size, embed_dim]
        """
        batch_size = image_keys.shape[0]
        
        ptr = int(self.queue_ptr)
        queue_size = self.image_queue.shape[1]
        
        # Replace oldest entries
        if ptr + batch_size <= queue_size:
            self.image_queue[:, ptr:ptr + batch_size] = image_keys.T
            self.text_queue[:, ptr:ptr + batch_size] = text_keys.T
            ptr = (ptr + batch_size) % queue_size
        else:
            # Wrap around
            remaining = queue_size - ptr
            self.image_queue[:, ptr:] = image_keys[:remaining].T
            self.text_queue[:, ptr:] = text_keys[:remaining].T
            self.image_queue[:, :(batch_size - remaining)] = image_keys[remaining:].T
            self.text_queue[:, :(batch_size - remaining)] = text_keys[remaining:].T
            ptr = batch_size - remaining
        
        self.queue_ptr[0] = ptr
    
    def encode_vision_dinov2(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using DINOv2 + LoRA + Compression
        
        Args:
            images: [batch_size, 3, H, W] - Any size RGB images
        
        Returns:
            vision_features: [batch_size, num_patches, vision_embed_dim]
        """
        batch_size, channels, height, width = images.shape
        
        # Resize to 224x224 (DINOv2 standard)
        if height != 224 or width != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # ImageNet normalization (DINOv2 standard)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        images_normalized = (images - mean) / std
        
        # DINOv2 forward (base frozen + LoRA trainable)
        outputs = self.dinov2_model(pixel_values=images_normalized)
        
        # Get patch embeddings (exclude CLS token)
        # DINOv2-base: 256 patches (16x16) for 224x224 image
        patch_features = outputs.last_hidden_state[:, 1:, :]  # [B, 256, 768]
        
        # Compress: 768 → 128 (trainable compression layer)
        vision_features = self.feature_compressor(patch_features)  # [B, 256, vision_embed_dim]
        
        return vision_features
    
    def freeze_dinov2_base(self):
        """Freeze DINOv2 base (keep LoRA trainable)"""
        for name, param in self.dinov2_model.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False
        print("✓ DINOv2 base frozen (LoRA adapters remain trainable)")
    
    def unfreeze_dinov2_lora(self):
        """Unfreeze LoRA adapters (for main training phase)"""
        for name, param in self.dinov2_model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        print("✓ LoRA adapters unfrozen for training")
    
    def get_contrastive_features(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get contrastive features for ITC loss
        
        Args:
            text_embeddings: [batch_size, seq_len, embed_dim]
            image_embeddings: [batch_size, num_patches, vision_embed_dim]
        
        Returns:
            text_cls: [batch_size, embed_dim] - normalized text features
            image_cls: [batch_size, embed_dim] - normalized image features
        """
        batch_size = text_embeddings.shape[0]
        
        # Text features for contrastive learning
        text_itc = self.cross_modal_text_transform_itc(text_embeddings)  # [B, L, D]
        text_cls = self.cross_modal_text_pooler_itc(text_itc[:, 0:1])  # Use first token as CLS
        text_cls = text_cls.squeeze(1)  # [B, D]
        text_cls = F.normalize(text_cls, p=2, dim=-1, eps=1e-8)
        
        # Image features for contrastive learning
        image_itc = self.cross_modal_image_transform_itc(image_embeddings)  # [B, P, D]
        image_avg = self.avgpool(image_itc.transpose(1, 2)).view(batch_size, 1, -1)  # [B, 1, D]
        image_cls = self.cross_modal_image_pooler_itc(image_avg)
        image_cls = image_cls.squeeze(1)  # [B, D]
        image_cls = F.normalize(image_cls, p=2, dim=-1, eps=1e-8)
        
        return text_cls, image_cls
    
    def sample_hard_negatives(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hard negatives from queue for ITM loss (FIBER approach)
        
        Args:
            image_features: [batch_size, embed_dim] - normalized image features
            text_features: [batch_size, embed_dim] - normalized text features
        
        Returns:
            hard_neg_images: [batch_size, embed_dim] - hard negative images
            hard_neg_texts: [batch_size, embed_dim] - hard negative texts
        """
        batch_size = image_features.shape[0]
        
        # Compute similarities with queue
        sim_i2t = torch.matmul(image_features, self.text_queue)  # [B, queue_size]
        sim_t2i = torch.matmul(text_features, self.image_queue)  # [B, queue_size]
        
        # Sample hard negatives (highest similarity but wrong match)
        with torch.no_grad():
            # Softmax weights for sampling
            weights_i2t = F.softmax(sim_i2t, dim=1)  # [B, queue_size]
            weights_t2i = F.softmax(sim_t2i, dim=1)  # [B, queue_size]
            
            # Sample indices
            hard_neg_text_indices = torch.multinomial(weights_i2t, 1).squeeze(1)  # [B]
            hard_neg_image_indices = torch.multinomial(weights_t2i, 1).squeeze(1)  # [B]
            
            # Get hard negatives from queue
            hard_neg_texts = self.text_queue[:, hard_neg_text_indices].T  # [B, D]
            hard_neg_images = self.image_queue[:, hard_neg_image_indices].T  # [B, D]
        
        return hard_neg_images, hard_neg_texts
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        return_contrastive_features: bool = False,
        return_itm_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with FIBER-style fusion and optional contrastive/ITM features
        
        Args:
            text_embeddings: [batch_size, seq_len, embed_dim]
            images: [batch_size, 3, H, W] or None
            return_contrastive_features: Whether to return features for ITC loss
            return_itm_features: Whether to return features for ITM loss
        
        Returns:
            If return_contrastive_features=False and return_itm_features=False:
                fused_output: [batch_size, seq_len, embed_dim]
            Otherwise:
                fused_output, features_dict
        """
        if images is None:
            if return_contrastive_features or return_itm_features:
                return text_embeddings, {}
            return text_embeddings
        
        batch_size, seq_len, embed_dim = text_embeddings.shape
        
        # Encode vision with DINOv2 + LoRA + Compression
        image_embeds = self.encode_vision_dinov2(images)  # [B, num_patches, vision_embed_dim]
        
        # Transform to common embedding space (for fusion)
        image_embeds_common = self.cross_modal_image_transform(image_embeds)  # [B, P, D]
        text_embeds_common = self.cross_modal_text_transform(text_embeddings)  # [B, L, D]
        
        # Simple fusion: average image features and concatenate
        avg_image_features = self.avgpool(image_embeds_common.transpose(1, 2)).view(batch_size, 1, embed_dim)
        avg_image_features = avg_image_features.expand(-1, seq_len, -1)
        
        # Concatenate and fuse
        concatenated = torch.cat([text_embeds_common, avg_image_features], dim=-1)
        fused_output = self.fusion_mlp(concatenated)  # [B, L, D]
        
        # Get contrastive/ITM features if requested
        if return_contrastive_features or return_itm_features:
            text_cls, image_cls = self.get_contrastive_features(text_embeddings, image_embeds)
            
            features_dict = {
                'text_features': text_cls,  # [B, D]
                'image_features': image_cls,  # [B, D]
                'temperature': self.temperature
            }
            
            # Add ITC-specific features
            if return_contrastive_features:
                features_dict.update({
                    'text_queue': self.text_queue.clone(),  # [D, queue_size]
                    'image_queue': self.image_queue.clone(),  # [D, queue_size]
                })
            
            # Add ITM-specific features (hard negatives)
            if return_itm_features and self.training:
                hard_neg_images, hard_neg_texts = self.sample_hard_negatives(image_cls, text_cls)
                features_dict.update({
                    'hard_neg_images': hard_neg_images,  # [B, D]
                    'hard_neg_texts': hard_neg_texts,  # [B, D]
                })
            
            # Update queues (only during training)
            if self.training:
                self._dequeue_and_enqueue(image_cls, text_cls)
            
            return fused_output, features_dict
        
        return fused_output
