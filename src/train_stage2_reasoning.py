"""
Stage 2: Reasoning Module Training (tiny-r1 style)
Load Stage 1 checkpoint (frozen vision-language base) and add reasoning
Train on Robot Selection dataset with GRPO (reward-based learning)

This stage adds:
- Reasoning module (chain-of-thought traces)
- Robot selection head (multi-label classifier)
- GRPO training (reward functions for correctness)
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from train_stage1_vision_language import BitGenVisionLanguageModel, Stage1Config
from bitgen_model import ReasoningModule, RobotSelector, BitGenConfig
from data_loader import RobotSelectionDataset


@dataclass
class Stage2Config:
    """Configuration for Stage 2 training"""
    # Inherit from Stage 1
    stage1_checkpoint: str = "checkpoints/stage1/stage1_best.pt"
    
    # Reasoning config
    reasoning_dim: int = 64
    max_reasoning_steps: int = 8
    
    # Robot selection config
    num_robots: int = 5
    top_k_robots: int = 3
    robot_embed_dim: int = 32
    
    # Training config
    batch_size: int = 64
    grad_accum_steps: int = 4  # Effective batch: 256
    learning_rate: float = 1e-4  # Lower LR for fine-tuning
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_steps: int = 200
    
    # Loss weights
    reasoning_weight: float = 1.0
    robot_selection_weight: float = 1.0
    
    # Reward weights (GRPO)
    correctness_reward_weight: float = 1.0
    reasoning_trace_reward_weight: float = 0.5
    format_reward_weight: float = 0.1
    
    # Optimization
    max_seq_len: int = 512
    max_grad_norm: float = 1.0
    use_amp: bool = True
    freeze_vision_language: bool = True  # Freeze Stage 1 base
    
    # Paths
    data_file: str = "robot_selection_data/multi_robot_selection_dataset.json"
    checkpoint_dir: str = "checkpoints/stage2"
    log_dir: str = "logs/stage2"


class BitGenReasoningModel(nn.Module):
    """
    BitGen Stage 2 Model: Adds reasoning on top of Stage 1
    Loads frozen vision-language base, adds reasoning module
    """
    
    def __init__(self, stage1_model: BitGenVisionLanguageModel, config: Stage2Config):
        super().__init__()
        self.config = config
        
        # Load Stage 1 base (vision-language)
        self.vision_language_base = stage1_model
        
        # Freeze Stage 1 if specified
        if config.freeze_vision_language:
            for param in self.vision_language_base.parameters():
                param.requires_grad = False
            print("Froze vision-language base (Stage 1)")
        
        # Create BitGenConfig for reasoning module
        bitgen_config = BitGenConfig(
            embed_dim=stage1_model.config.embed_dim,
            reasoning_dim=config.reasoning_dim,
            max_reasoning_steps=config.max_reasoning_steps,
            num_robots=config.num_robots,
            top_k_robots=config.top_k_robots,
            robot_embed_dim=config.robot_embed_dim
        )
        
        # Reasoning module (tiny-r1 style)
        self.reasoning_module = ReasoningModule(bitgen_config)
        
        # Robot selection head
        self.robot_selector = RobotSelector(bitgen_config)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(stage1_model.config.embed_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        return_reasoning_trace: bool = False
    ) -> Dict:
        """
        Forward pass: Vision-Language base + Reasoning + Robot selection
        
        Args:
            input_ids: [batch_size, seq_len]
            images: [batch_size, 3, H, W]
            return_reasoning_trace: Whether to return chain-of-thought trace
        
        Returns:
            outputs: Dictionary with embeddings, reasoning, and robot selection
        """
        # Stage 1: Vision-Language base (frozen)
        with torch.no_grad() if self.config.freeze_vision_language else torch.enable_grad():
            stage1_outputs = self.vision_language_base(
                input_ids=input_ids,
                images=images,
                return_contrastive_features=False
            )
            x = stage1_outputs['embeddings']  # [B, L, D]
        
        # Stage 2: Reasoning module
        if return_reasoning_trace:
            x, reasoning_info = self.reasoning_module(x, return_reasoning_trace=True)
        else:
            x = self.reasoning_module(x, return_reasoning_trace=False)
            reasoning_info = {}
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Robot selection
        robot_outputs = self.robot_selector(x, return_top_k=True)
        
        outputs = {
            'embeddings': x,
            'reasoning_info': reasoning_info,
            'robot_selection': robot_outputs
        }
        
        return outputs


def compute_robot_selection_loss(
    pred_logits: torch.Tensor,
    target_labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute multi-label robot selection loss (BCE with logits)
    
    Args:
        pred_logits: [batch_size, num_robots]
        target_labels: [batch_size, num_robots] - binary labels
    
    Returns:
        loss: Scalar BCE loss
    """
    return F.binary_cross_entropy_with_logits(pred_logits, target_labels.float())


def compute_correctness_reward(
    pred_indices: torch.Tensor,
    target_labels: torch.Tensor,
    top_k: int = 3
) -> torch.Tensor:
    """
    Compute correctness reward for GRPO
    
    Args:
        pred_indices: [batch_size, top_k] - predicted robot indices
        target_labels: [batch_size, num_robots] - binary target labels
        top_k: Number of top predictions
    
    Returns:
        rewards: [batch_size] - correctness rewards
    """
    batch_size = pred_indices.shape[0]
    rewards = torch.zeros(batch_size, device=pred_indices.device)
    
    for i in range(batch_size):
        # Get ground truth robot indices
        gt_indices = torch.where(target_labels[i] == 1)[0]
        
        # Count correct predictions
        correct = 0
        for pred_idx in pred_indices[i]:
            if pred_idx in gt_indices:
                correct += 1
        
        # Reward = (correct / gt_count) (precision-like)
        if len(gt_indices) > 0:
            rewards[i] = correct / len(gt_indices)
    
    return rewards


def compute_reasoning_trace_reward(
    reasoning_info: Dict,
    target_labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute reward for reasoning trace quality
    
    Args:
        reasoning_info: Dictionary with reasoning traces
        target_labels: [batch_size, num_robots]
    
    Returns:
        rewards: [batch_size] - reasoning quality rewards
    """
    batch_size = target_labels.shape[0]
    
    # Simple reward: check if reasoning steps are diverse (not collapsed)
    if 'reasoning_steps' in reasoning_info:
        reasoning_steps = reasoning_info['reasoning_steps']  # [B, num_steps, D]
        
        # Compute diversity (L2 distance between steps)
        step_diffs = []
        for i in range(reasoning_steps.shape[1] - 1):
            diff = torch.norm(
                reasoning_steps[:, i+1] - reasoning_steps[:, i],
                p=2,
                dim=-1
            )
            step_diffs.append(diff)
        
        if step_diffs:
            avg_diff = torch.stack(step_diffs, dim=0).mean(dim=0)  # [B]
            # Normalize to [0, 1]
            rewards = torch.sigmoid(avg_diff - 0.5)
        else:
            rewards = torch.zeros(batch_size, device=target_labels.device)
    else:
        rewards = torch.zeros(batch_size, device=target_labels.device)
    
    return rewards


class Stage2Trainer:
    """Trainer for Stage 2: Reasoning Module Training"""
    
    def __init__(self, config: Stage2Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Load Stage 1 checkpoint
        print(f"Loading Stage 1 checkpoint: {config.stage1_checkpoint}")
        checkpoint = torch.load(config.stage1_checkpoint, map_location=self.device)
        
        # Reconstruct Stage 1 model
        stage1_config = checkpoint['config']
        stage1_model = BitGenVisionLanguageModel(stage1_config)
        stage1_model.load_state_dict(checkpoint['model_state_dict'])
        stage1_model.to(self.device)
        
        print("Loaded Stage 1 model successfully")
        
        # Initialize Stage 2 model (with reasoning)
        print("Initializing Stage 2 reasoning model...")
        self.model = BitGenReasoningModel(stage1_model, config).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Optimizer (only trainable parameters)
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * 500,  # Approximate steps
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
        
        # Set Stage 1 to eval mode if frozen
        if self.config.freeze_vision_language:
            self.model.vision_language_base.eval()
        
        total_loss = 0.0
        total_robot_loss = 0.0
        total_correctness_reward = 0.0
        total_reasoning_reward = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            images = batch['images'].to(self.device)
            robot_labels = batch['robot_labels'].to(self.device)  # [B, num_robots]
            
            # Forward pass with AMP
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        images=images,
                        return_reasoning_trace=True
                    )
                    
                    # Robot selection loss
                    robot_logits = outputs['robot_selection']['all_logits']
                    robot_loss = compute_robot_selection_loss(robot_logits, robot_labels)
                    
                    # Compute rewards
                    pred_indices = outputs['robot_selection']['top_k_indices']
                    correctness_reward = compute_correctness_reward(
                        pred_indices, robot_labels, self.config.top_k_robots
                    )
                    reasoning_reward = compute_reasoning_trace_reward(
                        outputs['reasoning_info'], robot_labels
                    )
                    
                    # GRPO-style loss (negative reward)
                    reward_loss = -(
                        self.config.correctness_reward_weight * correctness_reward.mean() +
                        self.config.reasoning_trace_reward_weight * reasoning_reward.mean()
                    )
                    
                    # Total loss
                    loss = (
                        self.config.robot_selection_weight * robot_loss +
                        self.config.reasoning_weight * reward_loss
                    )
                
                # Backward pass
                self.scaler.scale(loss / self.config.grad_accum_steps).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    images=images,
                    return_reasoning_trace=True
                )
                
                robot_logits = outputs['robot_selection']['all_logits']
                robot_loss = compute_robot_selection_loss(robot_logits, robot_labels)
                
                pred_indices = outputs['robot_selection']['top_k_indices']
                correctness_reward = compute_correctness_reward(
                    pred_indices, robot_labels, self.config.top_k_robots
                )
                reasoning_reward = compute_reasoning_trace_reward(
                    outputs['reasoning_info'], robot_labels
                )
                
                reward_loss = -(
                    self.config.correctness_reward_weight * correctness_reward.mean() +
                    self.config.reasoning_trace_reward_weight * reasoning_reward.mean()
                )
                
                loss = (
                    self.config.robot_selection_weight * robot_loss +
                    self.config.reasoning_weight * reward_loss
                )
                
                (loss / self.config.grad_accum_steps).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Compute accuracy
            with torch.no_grad():
                robot_probs = torch.sigmoid(robot_logits)
                pred_binary = (robot_probs > 0.5).float()
                accuracy = (pred_binary == robot_labels).float().mean()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_robot_loss += robot_loss.item()
            total_correctness_reward += correctness_reward.mean().item()
            total_reasoning_reward += reasoning_reward.mean().item()
            total_accuracy += accuracy.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'robot': f"{robot_loss.item():.4f}",
                'reward': f"{correctness_reward.mean().item():.3f}",
                'acc': f"{accuracy.item():.3f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'robot_loss': total_robot_loss / num_batches,
            'correctness_reward': total_correctness_reward / num_batches,
            'reasoning_reward': total_reasoning_reward / num_batches,
            'accuracy': total_accuracy / num_batches
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"stage2_epoch{epoch+1}.pt"
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
        print("Starting Stage 2 training: Reasoning Module")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size} (effective: {self.config.batch_size * self.config.grad_accum_steps})")
        print(f"Epochs: {self.config.num_epochs}")
        
        best_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            metrics = self.train_epoch(dataloader)
            
            # Print metrics
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Robot Loss: {metrics['robot_loss']:.4f}")
            print(f"  Correctness Reward: {metrics['correctness_reward']:.3f}")
            print(f"  Reasoning Reward: {metrics['reasoning_reward']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics)
            
            # Save best checkpoint
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_path = os.path.join(self.config.checkpoint_dir, "stage2_best.pt")
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
        
        print("\nStage 2 training complete!")
        print(f"Best accuracy: {best_accuracy:.3f}")


def main():
    """Main training script"""
    config = Stage2Config()
    
    # Load Robot Selection dataset
    print("Loading Robot Selection dataset...")
    from data_loader import RobotDataset
    
    dataset = RobotDataset(
        data_file=config.data_file,
        max_seq_len=config.max_seq_len
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
    trainer = Stage2Trainer(config)
    
    # Train
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
