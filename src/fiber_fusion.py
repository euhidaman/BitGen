"""
FIBER-style Cross-Modal Fusion with Queue-Based Contrastive Learning
Adapted from FIBER for BitGen Stage 1 (Vision-Language Pre-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, Union


class FIBERCrossModalFusion(nn.Module):
    """
    FIBER-inspired cross-modal fusion with queue-based contrastive learning
    Separate transforms for fusion and contrastive learning
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.vision_embed_dim = config.vision_embed_dim
        
        # Vision Encoder: Lightweight CNN (BitMar-style) or DINOv2
        self.use_lightweight = getattr(config, 'use_lightweight_vision', False)
        
        if self.use_lightweight:
            # Tiny CNN vision encoder (<5M params) - BitMar style
            print("Using lightweight CNN vision encoder (<5M params)...")
            self.vision_encoder = nn.Sequential(
                # Conv block 1: 3→32
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                # Conv block 2: 32→64
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # Conv block 3: 64→128
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((7, 7))  # Output: 7x7 spatial grid
            )
            self.dinov2_dim = 128  # Output channels
            self.vision_to_embed = nn.Linear(128, config.vision_embed_dim)
        else:
            # DINOv2 Vision Encoder (300M params)
            from transformers import Dinov2Model
            print("Loading facebook/dinov2-base for FIBER fusion (300M params)...")
            self.dinov2_model = Dinov2Model.from_pretrained('facebook/dinov2-base')
            self.dinov2_model.train()
            for param in self.dinov2_model.parameters():
                param.requires_grad = True
            self.dinov2_dim = 768
            self.vision_to_embed = nn.Linear(self.dinov2_dim, config.vision_embed_dim)
        
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
        Encode images using lightweight CNN or DINOv2
        
        Args:
            images: [batch_size, 3, H, W]
        
        Returns:
            vision_features: [batch_size, num_patches, vision_embed_dim]
        """
        batch_size, channels, height, width = images.shape
        
        if self.use_lightweight:
            # Lightweight CNN encoder
            # Resize to 224x224
            if height != 224 or width != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images_normalized = (images - mean) / std
            
            # CNN forward: [B, 3, 224, 224] → [B, 128, 7, 7]
            features = self.vision_encoder(images_normalized)
            
            # Reshape to patches: [B, 128, 7, 7] → [B, 49, 128]
            features = features.flatten(2).transpose(1, 2)  # [B, 49, 128]
            
            # Project to vision_embed_dim
            vision_features = self.vision_to_embed(features)  # [B, 49, vision_embed_dim]
        else:
            # DINOv2 encoder (original)
            if height != 224 or width != 224:
                images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images_normalized = (images - mean) / std
            
            outputs = self.dinov2_model(pixel_values=images_normalized)
            
            # Get patch embeddings (exclude CLS token)
            patch_features = outputs.last_hidden_state[:, 1:, :]  # [B, num_patches, 768]
            
            # Project to vision embedding dimension
            vision_features = self.vision_to_embed(patch_features)  # [B, num_patches, vision_embed_dim]
        
        return vision_features
    
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
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        return_contrastive_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with FIBER-style fusion and optional contrastive features
        
        Args:
            text_embeddings: [batch_size, seq_len, embed_dim]
            images: [batch_size, 3, H, W] or None
            return_contrastive_features: Whether to return features for contrastive loss
        
        Returns:
            If return_contrastive_features=False:
                fused_output: [batch_size, seq_len, embed_dim]
            If return_contrastive_features=True:
                fused_output, contrastive_dict
        """
        if images is None:
            if return_contrastive_features:
                return text_embeddings, {}
            return text_embeddings
        
        batch_size, seq_len, embed_dim = text_embeddings.shape
        
        # Encode vision with DINOv2
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
        
        # Get contrastive features if requested
        if return_contrastive_features:
            text_cls, image_cls = self.get_contrastive_features(text_embeddings, image_embeds)
            
            # Update queues (only during training)
            if self.training:
                self._dequeue_and_enqueue(image_cls, text_cls)
            
            contrastive_dict = {
                'text_features': text_cls,  # [B, D]
                'image_features': image_cls,  # [B, D]
                'text_queue': self.text_queue.clone(),  # [D, queue_size]
                'image_queue': self.image_queue.clone(),  # [D, queue_size]
                'temperature': self.temperature
            }
            
            return fused_output, contrastive_dict
        
        return fused_output
