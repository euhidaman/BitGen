"""
Stage 1: Vision-Language Pre-training (FIBER-style)
Train stable vision-language representations using COCO dataset
NO reasoning module - pure contrastive learning

After this stage, model should have:
- Aligned vision-language representations (image ↔ text matching)
- Larima GPM episodic memory (trained)
- FIBER-style cross-modal fusion (with queue-based contrastive)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import json
from tqdm import tqdm
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from bitgen_model import BitGenConfig, BitNetLinear
from larima_memory import BitGenMemory
from fiber_fusion import FIBERCrossModalFusion
from data_loader import COCODataset
from wandb_integration import setup_wandb_integration
from huggingface_integration import HuggingFaceIntegration


@dataclass
class Stage1Config:
    """Configuration for Stage 1 training"""
    # Model config
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 32
    ffn_dim: int = 512
    vocab_size: int = 8192
    
    # Memory config (Larima GPM)
    memory_size: int = 1000
    memory_dim: int = 256
    direct_writing: bool = True
    
    # Vision config
    vision_embed_dim: int = 128
    fusion_layers: int = 2
    
    # Training config
    batch_size: int = 160
    grad_accum_steps: int = 2  # Effective batch: 320
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500
    
    # Contrastive learning config
    contrastive_weight: float = 1.0
    memory_kl_weight: float = 0.01
    queue_size: int = 4096
    temperature: float = 0.07
    
    # Optimization
    max_seq_len: int = 512
    max_grad_norm: float = 1.0
    use_amp: bool = True
    
    # Paths
    data_file: str = "data/coco/coco_dataset.json"
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
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # FIBER cross-modal fusion (with DINOv2)
        bitgen_config = self._create_bitgen_config(config)
        self.cross_modal_fusion = FIBERCrossModalFusion(bitgen_config)
        
        # Larima GPM episodic memory
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
        return_contrastive_features: bool = False
    ) -> Dict:
        """
        Forward pass - Vision-Language fusion only
        
        Args:
            input_ids: [batch_size, seq_len]
            images: [batch_size, 3, H, W]
            return_contrastive_features: Whether to return features for contrastive loss
        
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
            x = self.cross_modal_fusion(x, images, return_contrastive_features=False)
            contrastive_dict = {}
        else:
            x = self.cross_modal_fusion(x, None)
            contrastive_dict = {}
        
        # Larima GPM episodic memory
        x, memory_info = self.episodic_memory(x)
        
        # Attention layers
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        
        # Layer norm
        x = self.layer_norm(x)
        
        outputs = {
            'embeddings': x,
            'contrastive_features': contrastive_dict,
            'memory_info': memory_info
        }
        
        return outputs


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
    sim_t2i_batch = torch.matmul(text_features, image_features.T) / temperature  # [B, B]
    sim_t2i_queue = torch.matmul(text_features, image_queue) / temperature  # [B, Q]
    sim_t2i = torch.cat([sim_t2i_batch, sim_t2i_queue], dim=1)  # [B, B+Q]
    
    # Image-to-text similarity (with queue negatives)
    sim_i2t_batch = torch.matmul(image_features, text_features.T) / temperature  # [B, B]
    sim_i2t_queue = torch.matmul(image_features, text_queue) / temperature  # [B, Q]
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize model
        print("Initializing BitGen Vision-Language model...")
        self.model = BitGenVisionLanguageModel(config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
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
        
        # Initialize HuggingFace Hub
        self.hf_integration = HuggingFaceIntegration(
            repo_name="BitGen-Reasoning",
            stage="stage1"
        )
        self.hf_integration.create_repo()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler (cosine with warmup)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * 1000,  # Approximate steps
            eta_min=1e-6
        )
        
        # AMP scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_memory_kl_loss = 0.0
        total_acc_t2i = 0.0
        total_acc_i2t = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            images = batch['images'].to(self.device)
            
            # Forward pass with AMP
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        images=images,
                        return_contrastive_features=True
                    )
                    
                    # Compute contrastive loss
                    contrastive_dict = outputs['contrastive_features']
                    if contrastive_dict:
                        loss_dict = compute_contrastive_loss(
                            text_features=contrastive_dict['text_features'],
                            image_features=contrastive_dict['image_features'],
                            text_queue=contrastive_dict['text_queue'],
                            image_queue=contrastive_dict['image_queue'],
                            temperature=contrastive_dict['temperature']
                        )
                        contrastive_loss = loss_dict['contrastive_loss']
                    else:
                        contrastive_loss = torch.tensor(0.0, device=self.device)
                    
                    # Compute memory KL loss
                    memory_kl_loss = self.model.episodic_memory.get_memory_kl_loss()
                    
                    # Total loss
                    loss = (
                        self.config.contrastive_weight * contrastive_loss +
                        self.config.memory_kl_weight * memory_kl_loss
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss / self.config.grad_accum_steps).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    images=images,
                    return_contrastive_features=True
                )
                
                # Compute losses
                contrastive_dict = outputs['contrastive_features']
                if contrastive_dict:
                    loss_dict = compute_contrastive_loss(
                        text_features=contrastive_dict['text_features'],
                        image_features=contrastive_dict['image_features'],
                        text_queue=contrastive_dict['text_queue'],
                        image_queue=contrastive_dict['image_queue'],
                        temperature=contrastive_dict['temperature']
                    )
                    contrastive_loss = loss_dict['contrastive_loss']
                else:
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                
                memory_kl_loss = self.model.episodic_memory.get_memory_kl_loss()
                
                loss = (
                    self.config.contrastive_weight * contrastive_loss +
                    self.config.memory_kl_weight * memory_kl_loss
                )
                
                (loss / self.config.grad_accum_steps).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Accumulate metrics
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_memory_kl_loss += memory_kl_loss.item()
            if contrastive_dict:
                total_acc_t2i += loss_dict['acc_t2i'].item()
                total_acc_i2t += loss_dict['acc_i2t'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cont': f"{contrastive_loss.item():.4f}",
                'acc_t2i': f"{loss_dict['acc_t2i'].item():.3f}" if contrastive_dict else "0.000",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb every 10 steps
            if self.global_step % 10 == 0:
                self.wandb.log_stage1_metrics(
                    epoch=self.epoch,
                    loss=loss.item(),
                    contrastive_loss=contrastive_loss.item(),
                    memory_kl_loss=memory_kl_loss.item(),
                    acc_t2i=loss_dict['acc_t2i'].item() if contrastive_dict else 0.0,
                    acc_i2t=loss_dict['acc_i2t'].item() if contrastive_dict else 0.0,
                    lr=self.scheduler.get_last_lr()[0]
                )
                self.wandb.step = self.global_step
        
        # Average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches,
            'memory_kl_loss': total_memory_kl_loss / num_batches,
            'acc_t2i': total_acc_t2i / num_batches,
            'acc_i2t': total_acc_i2t / num_batches
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
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
    
    def train(self, dataloader: DataLoader):
        """Full training loop"""
        print("Starting Stage 1 training: Vision-Language Pre-training")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size} (effective: {self.config.batch_size * self.config.grad_accum_steps})")
        print(f"Epochs: {self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            metrics = self.train_epoch(dataloader)
            
            # Print metrics
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Contrastive Loss: {metrics['contrastive_loss']:.4f}")
            print(f"  Memory KL Loss: {metrics['memory_kl_loss']:.4f}")
            print(f"  Acc T2I: {metrics['acc_t2i']:.3f}")
            print(f"  Acc I2T: {metrics['acc_i2t']:.3f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics)
            
            # Push to HuggingFace Hub every 2 epochs
            if (epoch + 1) % 2 == 0:
                print(f"Pushing Stage 1 checkpoint to HuggingFace Hub...")
                self.hf_integration.push_model_checkpoint(
                    model=self.model,
                    config=self.config,
                    epoch=epoch,
                    metrics=metrics
                )
            
            # Save best checkpoint if accuracy improved
            if epoch == 0 or metrics['acc_t2i'] + metrics['acc_i2t'] > self.best_acc:
                self.best_acc = metrics['acc_t2i'] + metrics['acc_i2t']
                best_path = os.path.join(self.config.checkpoint_dir, "stage1_best.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'metrics': metrics,
                    'config': self.config
                }, best_path)
                print(f"Saved best checkpoint: {best_path}")
        
        print("\nStage 1 training complete!")
        print(f"Best accuracy: {self.best_acc:.3f}")


def main():
    """Main training script"""
    config = Stage1Config()
    
    # Load COCO dataset
    print("Loading COCO dataset...")
    dataset = COCODataset(
        data_file=config.data_file,
        max_seq_len=config.max_seq_len,
        vocab_size=config.vocab_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize trainer
    trainer = Stage1Trainer(config)
    trainer.best_acc = 0.0
    
    # Train
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
