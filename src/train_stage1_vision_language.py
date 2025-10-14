"""
Stage 1: Vision-Language Pre-training (FIBER-style)
Train stable vision-language representations using COCO dataset
NO reasoning module - pure contrastive learning

After this stage, model should have:
- Aligned vision-language representations (image ↔ text matching)
- Larimar GPM episodic memory (trained)
- FIBER-style cross-modal fusion (with queue-based contrastive)
"""

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*pynvml.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*UnsupportedFieldAttributeWarning.*')

from huggingface_integration import HuggingFaceIntegration
from wandb_integration import setup_wandb_integration
from data_loader import COCODataset
from fiber_fusion import FIBERCrossModalFusion
from larima_memory import BitGenMemory
from bitgen_model import BitGenConfig, BitNetLinear, BitNetTextDecoder
import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import json
from tqdm import tqdm
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path as PathlibPath

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class Stage1Config:
    """Configuration for Stage 1 training - TINY model for edge devices (BitMar-sized)"""
    # Model config - REDUCED to match BitMar (suitable for tiny devices!)
    embed_dim: int = 128  # Was 256 → 128 (match BitMar)
    num_layers: int = 4  # Was 6 → 4 (match BitMar)
    num_heads: int = 4  # Was 8 → 4 (match BitMar)
    head_dim: int = 32
    ffn_dim: int = 256  # Was 512 → 256 (match BitMar)
    vocab_size: int = 50257  # Was 8192 → 50257 (GPT-2 vocabulary for compatibility)

    # Memory config (Larimar GPM) - REDUCED for tiny model
    memory_size: int = 32  # Was 1000 → 32 (match BitMar)
    memory_dim: int = 128  # Match embed_dim
    direct_writing: bool = True

    # Vision config
    vision_embed_dim: int = 128  # Match embed_dim for consistency
    fusion_layers: int = 2

    # Training config (BitMar-style - KISS!)
    batch_size: int = 128  # Balanced for memory with 2-layer decoder
    grad_accum_steps: int = 2  # Effective batch: 256 (close to original 320)
    learning_rate: float = 2e-4  # Match BitMar exactly
    weight_decay: float = 0.02  # Match BitMar exactly
    num_epochs: int = 50  # Max epochs, but early stopping will kick in
    warmup_steps: int = 1000  # Match BitMar exactly

    # Early stopping config
    early_stopping_patience: int = 5  # Stop if no improvement for 5 epochs
    min_delta: float = 0.001  # Minimum improvement to consider
    validation_split: float = 0.05  # 5% for validation

    # Contrastive learning config
    contrastive_weight: float = 1.0
    text_loss_weight: float = 0.5  # Auxiliary loss for text reconstruction
    memory_kl_weight: float = 0.1  # Episodic memory regularization
    itm_weight: float = 0.5  # Image-Text Matching loss (hard negatives)
    queue_size: int = 4096
    temperature: float = 0.07  # FIBER default

    # Optimization
    max_seq_len: int = 256  # Was 512 → 256 (match BitMar)
    max_grad_norm: float = 1.0
    use_amp: bool = True
    
    # HuggingFace Hub config
    push_to_hub_every_epoch: bool = True  # Push checkpoint at end of each epoch
    max_checkpoints_to_keep: int = 5  # Keep last N checkpoints only

    # Paths
    data_file: str = "data/coco/validated_coco.json"
    checkpoint_dir: str = "checkpoints/stage1"
    log_dir: str = "logs/stage1"


class BitGenVisionLanguageModel(nn.Module):
    """
    BitGen Stage 1 Model: Vision-Language Pre-training ONLY
    No reasoning module - will be added in Stage 2
    """

    def __init__(self, config: Stage1Config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # FIBER cross-modal fusion (with DINOv2)
        bitgen_config = self._create_bitgen_config(config)
        self.cross_modal_fusion = FIBERCrossModalFusion(bitgen_config)

        # Larimar GPM episodic memory
        self.episodic_memory = BitGenMemory(bitgen_config)

        # Attention layers (simplified for Stage 1)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ffn_dim,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])

        # Output layers
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(0.1)

        # Text decoder for reconstruction loss (BitMar-style)
        # Use 2 decoder layers to keep model tiny (encoder has 4 layers)
        self.text_decoder = BitNetTextDecoder(bitgen_config, num_decoder_layers=2)

        # Initialize weights
        self.apply(self._init_weights)

    def _create_bitgen_config(self, config: Stage1Config):
        """Create BitGenConfig from Stage1Config"""
        return BitGenConfig(
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            ffn_dim=config.ffn_dim,
            vocab_size=config.vocab_size,
            memory_size=config.memory_size,
            memory_dim=config.memory_dim,
            direct_writing=config.direct_writing,
            vision_embed_dim=config.vision_embed_dim,
            fusion_layers=config.fusion_layers,
            max_seq_len=config.max_seq_len
        )

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        return_contrastive_features: bool = False,
        target_ids: Optional[torch.Tensor] = None,
        use_decoder: bool = False
    ) -> Dict:
        """
        Forward pass - Vision-Language fusion only

        Args:
            input_ids: [batch_size, seq_len]
            images: [batch_size, 3, H, W]
            return_contrastive_features: Whether to return features for contrastive loss
            target_ids: Target token IDs for text reconstruction (optional, unused in Stage 1)
            use_decoder: Whether to use decoder (optional, unused in Stage 1)

        Returns:
            outputs: Dictionary with embeddings and contrastive features
        """
        batch_size, seq_len = input_ids.shape

        # Token + position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        x = self.dropout(token_emb + pos_emb)

        # FIBER cross-modal fusion
        if images is not None and return_contrastive_features:
            x, contrastive_dict = self.cross_modal_fusion(
                x, images, return_contrastive_features=True
            )
        elif images is not None:
            x = self.cross_modal_fusion(
                x, images, return_contrastive_features=False)
            contrastive_dict = {}
        else:
            x = self.cross_modal_fusion(x, None)
            contrastive_dict = {}

        # Larimar GPM episodic memory
        x, memory_info = self.episodic_memory(x)

        # Attention layers
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        # Layer norm
        x = self.layer_norm(x)

        # Text decoder for reconstruction loss (BitMar-style)
        decoder_logits = None
        if use_decoder and target_ids is not None:
            # Get target embeddings
            target_emb = self.token_embedding(target_ids)
            target_pos_ids = torch.arange(target_ids.size(1), device=target_ids.device).unsqueeze(0)
            target_pos_emb = self.pos_embedding(target_pos_ids)
            target_inputs = self.dropout(target_emb + target_pos_emb)
            
            # Use encoder output (x) as context for decoder
            decoder_logits = self.text_decoder(
                x=target_inputs,
                encoder_output=x,
                attention_mask=None,
                causal_mask=None
            )

        outputs = {
            'embeddings': x,
            'contrastive_features': contrastive_dict,
            'memory_info': memory_info,
            'decoder_logits': decoder_logits
        }

        return outputs

    def compute_loss(self, outputs, target_ids, contrastive_dict=None,
                     text_loss_weight=0.5, contrastive_loss_weight=1.0, memory_kl_weight=0.1, itm_weight=0.5):
        """
        Compute multi-component loss with proper vision-language alignment
        
        Loss Components:
        1. Image-Text Contrastive (ITC) Loss: Primary signal for vision-language alignment
        2. Image-Text Matching (ITM) Loss: Binary classification with hard negatives
        3. Text Reconstruction Loss: Auxiliary loss for language understanding
        4. Memory KL Loss: Episodic memory regularization

        Args:
            outputs: Model outputs dictionary
            target_ids: Target token IDs for text reconstruction
            contrastive_dict: Dictionary with text_features, image_features, queues, temperature
            text_loss_weight: Weight for text reconstruction loss (auxiliary)
            contrastive_loss_weight: Weight for ITC loss (PRIMARY)
            memory_kl_weight: Weight for memory KL divergence
            itm_weight: Weight for image-text matching loss

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. Image-Text Contrastive (ITC) Loss - PRIMARY learning signal
        if contrastive_dict is not None:
            text_features = contrastive_dict['text_features']  # [B, D]
            image_features = contrastive_dict['image_features']  # [B, D]
            text_queue = contrastive_dict['text_queue']  # [D, Q]
            image_queue = contrastive_dict['image_queue']  # [D, Q]
            temperature = contrastive_dict['temperature']
            
            # Compute FIBER-style queue-based contrastive loss
            contrastive_result = compute_contrastive_loss(
                text_features, image_features,
                text_queue, image_queue, temperature
            )
            
            contrastive_loss = contrastive_result['contrastive_loss']
            loss_dict['contrastive_loss'] = contrastive_loss.item()
            loss_dict['acc_t2i'] = contrastive_result['acc_t2i'].item()
            loss_dict['acc_i2t'] = contrastive_result['acc_i2t'].item()
            total_loss += contrastive_loss_weight * contrastive_loss
        else:
            loss_dict['contrastive_loss'] = 0.0
            loss_dict['acc_t2i'] = 0.0
            loss_dict['acc_i2t'] = 0.0

        # 2. Image-Text Matching (ITM) Loss - Hard negative mining
        if contrastive_dict is not None:
            text_features = contrastive_dict['text_features']
            image_features = contrastive_dict['image_features']
            batch_size = text_features.shape[0]
            
            # Create hard negatives by shuffling within batch
            # Positive pairs: (text[i], image[i]) → label = 1
            # Negative pairs: (text[i], image[j where j≠i]) → label = 0
            
            # Positive pairs
            pos_similarity = (text_features * image_features).sum(dim=-1)  # [B]
            pos_labels = torch.ones(batch_size, device=text_features.device)
            
            # Hard negative pairs (use most confusing negatives)
            with torch.no_grad():
                sim_matrix = text_features @ image_features.T  # [B, B]
                # Mask diagonal (positive pairs)
                mask = torch.eye(batch_size, device=sim_matrix.device).bool()
                sim_matrix.masked_fill_(mask, float('-inf'))
                # Get hardest negative for each sample
                hard_neg_idx = sim_matrix.argmax(dim=1)  # [B]
            
            neg_image_features = image_features[hard_neg_idx]
            neg_similarity = (text_features * neg_image_features).sum(dim=-1)  # [B]
            neg_labels = torch.zeros(batch_size, device=text_features.device)
            
            # Combine positive and negative pairs
            all_similarities = torch.cat([pos_similarity, neg_similarity], dim=0)
            all_labels = torch.cat([pos_labels, neg_labels], dim=0)
            
            # Binary classification loss
            itm_loss = F.binary_cross_entropy_with_logits(
                all_similarities, all_labels
            )
            
            # Compute ITM accuracy
            with torch.no_grad():
                itm_preds = (torch.sigmoid(all_similarities) > 0.5).float()
                itm_acc = (itm_preds == all_labels).float().mean()
            
            loss_dict['itm_loss'] = itm_loss.item()
            loss_dict['itm_acc'] = itm_acc.item()
            total_loss += itm_weight * itm_loss
        else:
            loss_dict['itm_loss'] = 0.0
            loss_dict['itm_acc'] = 0.0

        # 3. Text Reconstruction Loss (AUXILIARY - for language understanding)
        if outputs.get('decoder_logits') is not None and target_ids is not None:
            decoder_logits = outputs['decoder_logits']  # [B, seq_len, vocab_size]
            
            # Reshape for loss computation
            batch_size, seq_len, vocab_size = decoder_logits.shape
            flat_logits = decoder_logits.reshape(-1, vocab_size)
            flat_targets = target_ids.reshape(-1)
            
            # Cross-entropy loss with label smoothing
            text_loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=-100,  # Ignore padding tokens
                label_smoothing=0.1  # Prevent mode collapse
            )
            
            # Compute perplexity (clip to prevent inf)
            perplexity = torch.exp(torch.clamp(text_loss, max=10.0))
            
            # Compute token accuracy
            with torch.no_grad():
                predictions = flat_logits.argmax(dim=-1)
                mask = (flat_targets != -100)
                if mask.any():
                    correct = (predictions == flat_targets) & mask
                    token_accuracy = correct.float().sum() / mask.float().sum()
                else:
                    token_accuracy = torch.tensor(0.0)
            
            loss_dict['text_loss'] = text_loss.item()
            loss_dict['perplexity'] = perplexity.item()
            loss_dict['token_accuracy'] = token_accuracy.item()
            total_loss += text_loss_weight * text_loss
        else:
            loss_dict['text_loss'] = 0.0
            loss_dict['perplexity'] = 0.0
            loss_dict['token_accuracy'] = 0.0

        # 4. Memory KL Divergence Loss (Episodic memory regularization)
        if 'memory_kl' in outputs:
            memory_kl_loss = outputs['memory_kl']
            loss_dict['memory_kl_loss'] = memory_kl_loss.item()
            total_loss += memory_kl_weight * memory_kl_loss
        else:
            loss_dict['memory_kl_loss'] = 0.0
        
        # Add memory utilization metrics
        if 'memory_info' in outputs:
            memory_info = outputs['memory_info']
            if 'memory_usage_rate' in memory_info:
                loss_dict['memory_utilization'] = memory_info['memory_usage_rate']
            if 'memory_quality_avg' in memory_info:
                loss_dict['memory_quality'] = memory_info['memory_quality_avg']

        loss_dict['total_loss'] = total_loss.item() if isinstance(
            total_loss, torch.Tensor) else total_loss

        return total_loss, loss_dict


def compute_contrastive_loss(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    text_queue: torch.Tensor,
    image_queue: torch.Tensor,
    temperature: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute queue-based contrastive loss (FIBER approach)

    Args:
        text_features: [batch_size, embed_dim] - normalized
        image_features: [batch_size, embed_dim] - normalized
        text_queue: [embed_dim, queue_size] - normalized
        image_queue: [embed_dim, queue_size] - normalized
        temperature: Scalar temperature parameter

    Returns:
        loss_dict: Dictionary with contrastive losses
    """
    batch_size = text_features.shape[0]

    # Image-to-text contrastive
    # Positives: diagonal (matching pairs)
    # Negatives: off-diagonal (batch) + queue

    # Text-to-image similarity (with queue negatives)
    sim_t2i_batch = torch.matmul(
        text_features, image_features.T) / temperature  # [B, B]
    sim_t2i_queue = torch.matmul(
        text_features, image_queue) / temperature  # [B, Q]
    sim_t2i = torch.cat([sim_t2i_batch, sim_t2i_queue], dim=1)  # [B, B+Q]

    # Image-to-text similarity (with queue negatives)
    sim_i2t_batch = torch.matmul(
        image_features, text_features.T) / temperature  # [B, B]
    sim_i2t_queue = torch.matmul(
        image_features, text_queue) / temperature  # [B, Q]
    sim_i2t = torch.cat([sim_i2t_batch, sim_i2t_queue], dim=1)  # [B, B+Q]

    # Labels (diagonal is positive)
    labels = torch.arange(batch_size, device=text_features.device)

    # Cross-entropy loss
    loss_t2i = F.cross_entropy(sim_t2i, labels)
    loss_i2t = F.cross_entropy(sim_i2t, labels)

    # Total contrastive loss
    contrastive_loss = (loss_t2i + loss_i2t) / 2.0

    # Compute accuracy (for monitoring)
    with torch.no_grad():
        acc_t2i = (sim_t2i_batch.argmax(dim=1) == labels).float().mean()
        acc_i2t = (sim_i2t_batch.argmax(dim=1) == labels).float().mean()

    return {
        'contrastive_loss': contrastive_loss,
        'loss_t2i': loss_t2i,
        'loss_i2t': loss_i2t,
        'acc_t2i': acc_t2i,
        'acc_i2t': acc_i2t
    }


class Stage1Trainer:
    """Trainer for Stage 1: Vision-Language Pre-training"""

    def __init__(self, config: Stage1Config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize model
        print("Initializing BitGen Vision-Language model...")
        self.model = BitGenVisionLanguageModel(config).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Initialize WandB
        config_dict = {
            'stage': 'stage1',
            'embed_dim': config.embed_dim,
            'num_layers': config.num_layers,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'memory_size': config.memory_size,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
        self.wandb = setup_wandb_integration(
            project_name="bitgen-training",
            entity="babylm-ntust",
            run_name=f"stage1-vision-language-{time.strftime('%Y%m%d-%H%M%S')}",
            config=config_dict,
            stage="stage1"
        )

        # Initialize HuggingFace Hub - Stage 1: PreReasoning (Vision-Language)
        self.hf_integration = HuggingFaceIntegration(
            repo_name="BitGen-PreReasoning",
            stage="stage1"
        )
        self.hf_integration.create_repo()

        # Early stopping tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0

        # Optimizer - start with target LR (scheduler will handle warmup)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,  # Start with target LR
            weight_decay=config.weight_decay
        )

        # Scheduler with warmup - will be initialized after dataset is loaded
        self.scheduler = None
        self.warmup_scheduler = None

        # AMP scaler (using new API)
        self.scaler = GradScaler('cuda') if config.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.total_steps = 0  # Will be set in train()

    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler with linear warmup then cosine decay"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            # Linear warmup for first warmup_steps
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            
            # Cosine decay after warmup
            progress = float(current_step - self.config.warmup_steps) / float(max(1, num_training_steps - self.config.warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Cosine from 1.0 to 0.1
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_memory_kl_loss = 0.0
        total_text_loss = 0.0
        total_perplexity = 0.0
        total_token_accuracy = 0.0
        num_batches = 0

        # Log current learning rate at start of epoch
        current_lr = self.optimizer.param_groups[0]['lr']
        print(
            f"\n📈 Starting Epoch {self.epoch+1} - Learning Rate: {current_lr:.2e}")

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            images = batch['images'].to(self.device)

            # Forward pass with AMP
            if self.config.use_amp:
                with autocast('cuda'):
                    # Forward pass with decoder (BitMar-style)
                    outputs = self.model(
                        input_ids=input_ids,
                        images=images,
                        target_ids=input_ids,  # Use same sequence as target for reconstruction
                        use_decoder=True,
                        return_contrastive_features=True
                    )

                    # Extract features for contrastive loss
                    contrastive_dict = outputs.get('contrastive_features', {})
                    text_features = contrastive_dict.get(
                        'text_features') if contrastive_dict else None
                    image_features = contrastive_dict.get(
                        'image_features') if contrastive_dict else None

                    # Get memory KL loss
                    memory_kl = self.model.episodic_memory.get_memory_kl_loss()
                    outputs['memory_kl'] = memory_kl

                    # Compute multi-component loss with proper alignment learning
                    loss, loss_dict = self.model.compute_loss(
                        outputs=outputs,
                        target_ids=input_ids,
                        contrastive_dict=contrastive_dict,  # Pass full dict with queues
                        text_loss_weight=self.config.text_loss_weight,  # 0.5 - AUXILIARY
                        contrastive_loss_weight=self.config.contrastive_weight,  # 1.0 - PRIMARY
                        memory_kl_weight=self.config.memory_kl_weight,  # 0.1
                        itm_weight=self.config.itm_weight  # 0.5 - Hard negative mining
                    )

                    # Extract individual losses for logging
                    contrastive_loss = loss_dict.get('contrastive_loss', 0.0)
                    itm_loss = loss_dict.get('itm_loss', 0.0)
                    itm_acc = loss_dict.get('itm_acc', 0.0)
                    acc_t2i = loss_dict.get('acc_t2i', 0.0)
                    acc_i2t = loss_dict.get('acc_i2t', 0.0)
                    memory_kl_loss = loss_dict.get('memory_kl_loss', 0.0)
                    text_loss = loss_dict.get('text_loss', 0.0)
                    perplexity = loss_dict.get('perplexity', 0.0)
                    token_accuracy = loss_dict.get('token_accuracy', 0.0)

                # Backward pass with gradient scaling
                self.scaler.scale(
                    loss / self.config.grad_accum_steps).backward()
            else:
                # Forward pass with decoder (BitMar-style, no AMP)
                outputs = self.model(
                    input_ids=input_ids,
                    images=images,
                    target_ids=input_ids,
                    use_decoder=True,
                    return_contrastive_features=True
                )

                # Extract features for contrastive loss
                contrastive_dict = outputs.get('contrastive_features', {})

                # Get memory KL loss
                memory_kl = self.model.episodic_memory.get_memory_kl_loss()
                outputs['memory_kl'] = memory_kl

                # Compute multi-component loss with proper alignment learning
                loss, loss_dict = self.model.compute_loss(
                    outputs=outputs,
                    target_ids=input_ids,
                    contrastive_dict=contrastive_dict,  # Pass full dict with queues
                    text_loss_weight=self.config.text_loss_weight,  # 0.5 - AUXILIARY
                    contrastive_loss_weight=self.config.contrastive_weight,  # 1.0 - PRIMARY
                    memory_kl_weight=self.config.memory_kl_weight,  # 0.1
                    itm_weight=self.config.itm_weight  # 0.5 - Hard negative mining
                )

                # Extract individual losses for logging
                contrastive_loss = loss_dict.get('contrastive_loss', 0.0)
                itm_loss = loss_dict.get('itm_loss', 0.0)
                itm_acc = loss_dict.get('itm_acc', 0.0)
                acc_t2i = loss_dict.get('acc_t2i', 0.0)
                acc_i2t = loss_dict.get('acc_i2t', 0.0)
                memory_kl_loss = loss_dict.get('memory_kl_loss', 0.0)
                text_loss = loss_dict.get('text_loss', 0.0)
                perplexity = loss_dict.get('perplexity', 0.0)
                token_accuracy = loss_dict.get('token_accuracy', 0.0)

                (loss / self.config.grad_accum_steps).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                # Unscale gradients ONCE before any operations (AMP)
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Log gradient statistics BEFORE clipping (every 100 steps)
                if (self.global_step + 1) % 100 == 0:
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(
                        f"\n🔍 Step {self.global_step + 1}: Gradient norm = {total_norm:.6f} (before clip), LR = {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Clip gradients (after unscaling, before stepping)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                
                # Step scheduler every iteration (BitMar-style - KISS!)
                self.scheduler.step()
                
                self.global_step += 1
                
                # Clear CUDA cache periodically to avoid fragmentation
                if self.global_step % 50 == 0:
                    torch.cuda.empty_cache()

            # Accumulate metrics
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss if isinstance(
                contrastive_loss, float) else contrastive_loss.item() if hasattr(contrastive_loss, 'item') else 0.0
            total_memory_kl_loss += memory_kl_loss if isinstance(
                memory_kl_loss, float) else memory_kl_loss.item() if hasattr(memory_kl_loss, 'item') else 0.0
            total_text_loss += text_loss if isinstance(
                text_loss, float) else text_loss.item() if hasattr(text_loss, 'item') else 0.0
            total_perplexity += perplexity if isinstance(
                perplexity, float) else perplexity.item() if hasattr(perplexity, 'item') else 0.0
            total_token_accuracy += token_accuracy if isinstance(
                token_accuracy, float) else token_accuracy.item() if hasattr(token_accuracy, 'item') else 0.0
            num_batches += 1

            # Update progress bar with all metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'itc': f"{contrastive_loss if isinstance(contrastive_loss, float) else contrastive_loss.item() if hasattr(contrastive_loss, 'item') else 0.0:.4f}",
                'itm': f"{itm_loss if isinstance(itm_loss, float) else itm_loss.item() if hasattr(itm_loss, 'item') else 0.0:.4f}",
                'txt': f"{text_loss if isinstance(text_loss, float) else text_loss.item() if hasattr(text_loss, 'item') else 0.0:.4f}",
                't2i': f"{acc_t2i if isinstance(acc_t2i, float) else acc_t2i.item() if hasattr(acc_t2i, 'item') else 0.0:.2f}",
                'i2t': f"{acc_i2t if isinstance(acc_i2t, float) else acc_i2t.item() if hasattr(acc_i2t, 'item') else 0.0:.2f}",
                'lr': f"{current_lr:.2e}"
            })

            # Log to wandb every 10 steps
            if self.global_step % 10 == 0:
                self.wandb.log_stage1_metrics(
                    epoch=self.epoch,
                    loss=loss.item(),
                    text_loss=text_loss if isinstance(
                        text_loss, float) else text_loss.item() if hasattr(text_loss, 'item') else 0.0,
                    contrastive_loss=contrastive_loss if isinstance(
                        contrastive_loss, float) else contrastive_loss.item() if hasattr(contrastive_loss, 'item') else 0.0,
                    memory_kl_loss=memory_kl_loss if isinstance(
                        memory_kl_loss, float) else memory_kl_loss.item() if hasattr(memory_kl_loss, 'item') else 0.0,
                    perplexity=perplexity if isinstance(perplexity, float) else perplexity.item(
                    ) if hasattr(perplexity, 'item') else 0.0,
                    token_accuracy=token_accuracy if isinstance(
                        token_accuracy, float) else token_accuracy.item() if hasattr(token_accuracy, 'item') else 0.0,
                    lr=current_lr
                )

                # Log loss components (new multi-component loss)
                self.wandb.log_loss_components_stacked(
                    text_loss=text_loss if isinstance(
                        text_loss, float) else text_loss.item() if hasattr(text_loss, 'item') else 0.0,
                    contrastive_loss=contrastive_loss if isinstance(
                        contrastive_loss, float) else contrastive_loss.item() if hasattr(contrastive_loss, 'item') else 0.0,
                    memory_kl=memory_kl_loss if isinstance(memory_kl_loss, float) else memory_kl_loss.item(
                    ) if hasattr(memory_kl_loss, 'item') else 0.0,
                    epoch=self.epoch,
                    step=self.global_step
                )

                self.wandb.step = self.global_step

            # Comprehensive visualizations every 50 steps
            if self.global_step % 50 == 0 and contrastive_dict:
                # Similarity matrix heatmap
                self.wandb.log_similarity_matrix(
                    text_features=contrastive_dict['text_features'],
                    image_features=contrastive_dict['image_features'],
                    epoch=self.epoch,
                    step=self.global_step
                )

                # Queue quality heatmap
                self.wandb.log_queue_quality_heatmap(
                    current_text_features=contrastive_dict['text_features'],
                    current_image_features=contrastive_dict['image_features'],
                    text_queue=contrastive_dict['text_queue'],
                    image_queue=contrastive_dict['image_queue'],
                    epoch=self.epoch,
                    step=self.global_step
                )

                # Retrieval Precision@K
                self.wandb.log_retrieval_precision_at_k(
                    text_features=contrastive_dict['text_features'],
                    image_features=contrastive_dict['image_features'],
                    epoch=self.epoch,
                    step=self.global_step
                )

                # Gradient flow heatmap
                self.wandb.log_gradient_flow_heatmap(
                    model=self.model,
                    epoch=self.epoch,
                    step=self.global_step
                )

        # Average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches,
            'memory_kl_loss': total_memory_kl_loss / num_batches,
            'text_loss': total_text_loss / num_batches,
            'perplexity': total_perplexity / num_batches,
            'token_accuracy': total_token_accuracy / num_batches
        }

        return avg_metrics

    def validate(self, dataloader: DataLoader) -> Dict:
        """Validation loop to check model convergence"""
        self.model.eval()

        total_loss = 0.0
        total_text_loss = 0.0
        total_contrastive_loss = 0.0
        total_memory_kl_loss = 0.0
        total_perplexity = 0.0
        total_token_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
                input_ids = batch['input_ids'].to(self.device)
                images = batch['images'].to(self.device)

                # Forward pass with decoder
                outputs = self.model(
                    input_ids=input_ids,
                    images=images,
                    target_ids=input_ids,
                    use_decoder=True,
                    return_contrastive_features=True
                )

                # Extract features for contrastive loss
                contrastive_dict = outputs.get('contrastive_features', {})
                text_features = contrastive_dict.get(
                    'text_features') if contrastive_dict else None
                image_features = contrastive_dict.get(
                    'image_features') if contrastive_dict else None

                # Get memory KL loss
                memory_kl = self.model.episodic_memory.get_memory_kl_loss()
                outputs['memory_kl'] = memory_kl

                # Compute multi-component loss with proper alignment learning
                loss, loss_dict = self.model.compute_loss(
                    outputs=outputs,
                    target_ids=input_ids,
                    contrastive_dict=contrastive_dict,  # Pass full dict with queues
                    text_loss_weight=self.config.text_loss_weight,  # 0.5 - AUXILIARY
                    contrastive_loss_weight=self.config.contrastive_weight,  # 1.0 - PRIMARY
                    memory_kl_weight=self.config.memory_kl_weight,  # 0.1
                    itm_weight=self.config.itm_weight  # 0.5 - Hard negative mining
                )

                # Extract individual losses for logging
                contrastive_loss = loss_dict.get('contrastive_loss', 0.0)
                itm_loss = loss_dict.get('itm_loss', 0.0)
                memory_kl_loss = loss_dict.get('memory_kl_loss', 0.0)
                text_loss = loss_dict.get('text_loss', 0.0)
                perplexity = loss_dict.get('perplexity', 0.0)
                token_accuracy = loss_dict.get('token_accuracy', 0.0)

                total_loss += loss.item()
                total_text_loss += text_loss
                total_contrastive_loss += contrastive_loss
                total_memory_kl_loss += memory_kl_loss
                total_perplexity += perplexity
                total_token_accuracy += token_accuracy
                num_batches += 1

        # Average metrics
        avg_metrics = {
            'val_loss': total_loss / num_batches,
            'val_text_loss': total_text_loss / num_batches,
            'val_contrastive_loss': total_contrastive_loss / num_batches,
            'val_memory_kl_loss': total_memory_kl_loss / num_batches,
            'val_perplexity': total_perplexity / num_batches,
            'val_token_accuracy': total_token_accuracy / num_batches
        }

        self.model.train()
        return avg_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"stage1_epoch{epoch+1}.pt"
        )

        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model separately
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                "stage1_best.pt"
            )
            torch.save({
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': metrics,
                'config': self.config
            }, best_path)
            print(f"✅ New best model saved: {best_path}")
    
    def _push_checkpoint_to_hub(self, epoch: int):
        """Push checkpoint to HuggingFace Hub at end of epoch"""
        try:
            # Create checkpoint folder name
            checkpoint_name = f"checkpoint-epoch-{epoch+1}"
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{checkpoint_name}.pt")
            
            # Save checkpoint
            torch.save({
                'global_step': self.global_step,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config
            }, checkpoint_path)
            
            # Push to hub
            self.hf_integration.push_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
                metrics={
                    'step': self.global_step,
                    'epoch': epoch
                }
            )
            
            # Keep only last N checkpoints to save space
            self._cleanup_old_checkpoints(keep_last=self.config.max_checkpoints_to_keep)
            
            print(f"✅ Pushed {checkpoint_name} to HuggingFace Hub")
        except Exception as e:
            print(f"⚠️ Failed to push checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """Keep only last N checkpoints to save disk space"""
        try:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            # Get all checkpoint files (excluding best model)
            checkpoints = sorted(
                [f for f in checkpoint_dir.glob("checkpoint-epoch-*.pt")],
                key=lambda x: int(x.stem.split('-')[-1])  # Sort by epoch number
            )
            
            # Remove old checkpoints
            if len(checkpoints) > keep_last:
                for old_checkpoint in checkpoints[:-keep_last]:
                    old_checkpoint.unlink()
                    print(f"🗑️ Removed old checkpoint: {old_checkpoint.name}")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop with validation and early stopping"""
        print("Starting Stage 1 training: Vision-Language Pre-training")
        print(f"Device: {self.device}")
        print(
            f"Batch size: {self.config.batch_size} (effective: {self.config.batch_size * self.config.grad_accum_steps})")
        print(f"Max epochs: {self.config.num_epochs}")
        print(
            f"Early stopping patience: {self.config.early_stopping_patience} epochs")

        # Calculate total training steps and setup scheduler
        steps_per_epoch = len(train_loader) // self.config.grad_accum_steps
        total_training_steps = steps_per_epoch * self.config.num_epochs
        self.total_steps = total_training_steps

        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_training_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")
        print(f"Initial learning rate: 0 (warmup)")
        print(f"Target learning rate: {self.config.learning_rate}")

        # Setup scheduler with warmup
        self._setup_scheduler(total_training_steps)
        
        # Create and push model card to HuggingFace Hub
        print("\n📝 Creating model card on HuggingFace Hub...")
        config_dict = {
            'embed_dim': self.config.embed_dim,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'ffn_dim': self.config.ffn_dim,
            'vocab_size': self.config.vocab_size,
            'memory_size': self.config.memory_size,
            'max_seq_len': self.config.max_seq_len
        }
        training_args = {
            'batch_size': self.config.batch_size,
            'grad_accum_steps': self.config.grad_accum_steps,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'warmup_steps': self.config.warmup_steps,
            'max_grad_norm': self.config.max_grad_norm,
            'contrastive_weight': self.config.contrastive_weight,
            'text_loss_weight': self.config.text_loss_weight,
            'memory_kl_weight': self.config.memory_kl_weight,
            'itm_weight': self.config.itm_weight,
            'temperature': self.config.temperature,
            'queue_size': self.config.queue_size,
            'early_stopping_patience': self.config.early_stopping_patience
        }
        self.hf_integration.create_model_card(config_dict, training_args)

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validation epoch
            val_metrics = self.validate(val_loader)

            # Print metrics
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs} Summary:")
            print(f"{'='*60}")
            print(f"Training Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Text Loss: {train_metrics.get('text_loss', 0.0):.4f}")
            print(
                f"  Contrastive Loss: {train_metrics['contrastive_loss']:.4f}")
            print(f"  Memory KL Loss: {train_metrics['memory_kl_loss']:.4f}")
            if 'perplexity' in train_metrics:
                print(f"  Perplexity: {train_metrics['perplexity']:.2f}")
            if 'token_accuracy' in train_metrics:
                print(
                    f"  Token Accuracy: {train_metrics['token_accuracy']:.3f}")
            print(f"\nValidation Metrics:")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(
                f"  Val Text Loss: {val_metrics.get('val_text_loss', 0.0):.4f}")
            print(
                f"  Val Contrastive Loss: {val_metrics['val_contrastive_loss']:.4f}")
            print(
                f"  Val Memory KL Loss: {val_metrics['val_memory_kl_loss']:.4f}")
            if 'val_perplexity' in val_metrics:
                print(f"  Val Perplexity: {val_metrics['val_perplexity']:.2f}")
            if 'val_token_accuracy' in val_metrics:
                print(
                    f"  Val Token Accuracy: {val_metrics['val_token_accuracy']:.3f}")

            # Log all metrics to WandB
            combined_metrics = {**train_metrics, **val_metrics}
            combined_metrics['epoch'] = epoch + 1
            combined_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            self.wandb.log_stage1_metrics(
                epoch=epoch,
                loss=train_metrics['loss'],
                text_loss=train_metrics.get('text_loss', 0.0),
                contrastive_loss=train_metrics['contrastive_loss'],
                memory_kl_loss=train_metrics['memory_kl_loss'],
                perplexity=train_metrics.get('perplexity', 0.0),
                token_accuracy=train_metrics.get('token_accuracy', 0.0),
                lr=combined_metrics['learning_rate']
            )

            # Log validation metrics in organized sections
            log_dict = {
                'validation/loss_total': val_metrics['val_loss'],
                'validation/loss_contrastive': val_metrics['val_contrastive_loss'],
                'validation/loss_memory_kl': val_metrics['val_memory_kl_loss'],
            }

            # Add text-specific metrics if available
            if 'val_text_loss' in val_metrics:
                log_dict['validation/loss_text'] = val_metrics['val_text_loss']
            if 'val_perplexity' in val_metrics:
                log_dict['validation/perplexity'] = val_metrics['val_perplexity']
            if 'val_token_accuracy' in val_metrics:
                log_dict['validation/token_accuracy'] = val_metrics['val_token_accuracy']

            import wandb
            wandb.log(log_dict, step=self.global_step)

            # Epoch-level visualizations (using validation data for clean vis)
            print(f"📊 Generating epoch visualizations...")

            # Get a validation batch for visualizations
            val_batch = next(iter(val_loader))
            val_input_ids = val_batch['input_ids'].to(self.device)
            val_images = val_batch['images'].to(self.device)

            with torch.no_grad():
                val_outputs = self.model(
                    input_ids=val_input_ids,
                    images=val_images,
                    return_contrastive_features=True
                )
                val_contrastive_dict = val_outputs['contrastive_features']

            if val_contrastive_dict:
                # UMAP embedding space visualization
                self.wandb.log_embedding_space_umap(
                    text_embeddings=val_contrastive_dict['text_features'],
                    image_embeddings=val_contrastive_dict['image_features'],
                    epoch=epoch,
                    step=self.global_step,
                    sample_size=200
                )

            # Memory activation heatmap
            memory_mean = self.model.episodic_memory.gpm.memory_mean
            # Approximate retrieval counts (would need tracking in actual training)
            retrieval_counts = torch.ones(
                memory_mean.shape[0], device=memory_mean.device)

            self.wandb.log_memory_activation_heatmap(
                memory_mean=memory_mean,
                retrieval_counts=retrieval_counts,
                epoch=epoch,
                step=self.global_step
            )

            print(f"✅ Visualizations complete")

            # Cosine scheduler handles LR automatically (BitMar-style - KISS!)
            # No manual LR reduction needed

            # Early stopping check
            val_loss = val_metrics['val_loss']
            is_best = False

            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_epoch = epoch
                is_best = True
                print(f"\n✅ New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(
                    f"\n⏸️  No improvement. Patience: {self.patience_counter}/{self.config.early_stopping_patience}")

            # Save checkpoint
            self.save_checkpoint(epoch, combined_metrics, is_best=is_best)

            # Push to HuggingFace Hub after every epoch
            print(
                f"📤 Pushing Stage 1 checkpoint to HuggingFace Hub (BitGen-PreReasoning)...")
            self.hf_integration.push_model_checkpoint(
                model=self.model,
                config=self.config,
                epoch=epoch,
                metrics=combined_metrics
            )

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                print(
                    f"   Best epoch: {self.best_epoch+1} with val_loss: {self.best_val_loss:.4f}")
                break

        print("\n" + "="*60)
        print("Stage 1 training complete!")
        print(f"Best epoch: {self.best_epoch+1}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)


def main():
    """Main training script"""
    config = Stage1Config()

    # Load COCO dataset
    print("Loading COCO dataset...")
    full_dataset = COCODataset(
        data_file=config.data_file,
        max_seq_len=config.max_seq_len,
        vocab_size=config.vocab_size
    )

    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * config.validation_split)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Total dataset size: {dataset_size:,}")
    print(f"Training set: {train_size:,} samples")
    print(f"Validation set: {val_size:,} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize trainer
    trainer = Stage1Trainer(config)

    # Train with validation and early stopping
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
