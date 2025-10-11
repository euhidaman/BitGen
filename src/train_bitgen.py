"""
BitGen Training Script
Main training loop for embedded microcontroller deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# Updated PyTorch AMP imports to fix deprecation warnings
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import wandb
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import psutil
import time

# Import BitGen components
from .bitgen_model import BitGenModel, BitGenConfig, create_bitgen_model
from .adaptive_loss import BitGenLoss, AdaptiveLossManager, PerformanceTracker, EmbeddedTrainingUtils
from .data_loader import COCODataset, BitGenDataLoader, RobotSelectionDataset
from .wandb_monitor import WandbMonitor

# CodeCarbon for energy tracking
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("⚠️ CodeCarbon not installed. Install with: pip install codecarbon")

class BitGenTrainer:
    """Complete training system for BitGen models"""
    
    def __init__(self, 
                 config: BitGenConfig,
                 model_size: str = 'tiny',
                 output_dir: str = 'checkpoints',
                 use_wandb: bool = False,
                 wandb_project: str = "bitgen-training",
                 wandb_entity: Optional[str] = None,
                 wandb_run_name: Optional[str] = None,
                 wandb_tags: Optional[List[str]] = None,
                 push_to_hub: bool = False,
                 hf_repo_name: Optional[str] = None,
                 hf_organization: Optional[str] = None,
                 hf_private: bool = False):

        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = create_bitgen_model(model_size)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = BitGenLoss(config, config.vocab_size)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # STABILITY: Track loss moving average to detect divergence
        self.loss_ema = None  # Exponential moving average of loss
        self.loss_ema_alpha = 0.01  # Smoothing factor
        self.divergence_threshold = 2.0  # If current loss > 2x EMA, warn
        
        # Robot selection confusion matrix tracking (5x5 for exact dataset robots)
        self.robot_confusion_matrix = np.zeros((config.num_robots, config.num_robots))
        self.robot_selection_history = []  # Reserved for future use (currently unused)
        self.robot_accuracy_per_epoch = []
        self.MAX_HISTORY_SIZE = 1000  # Safety limit to prevent memory leaks if history gets used
        
        # Memory optimization
        self.memory_utils = EmbeddedTrainingUtils()
        
        # Logging
        self.setup_logging()
        
        # Wandb integration
        self.use_wandb = use_wandb
        self.wandb_monitor = None
        if use_wandb:
            self.setup_wandb(wandb_project, wandb_entity, wandb_run_name, wandb_tags)
            self.wandb_monitor = WandbMonitor(log_freq=100, log_attention_freq=100)
            self.logger.info("📊 Comprehensive WandB monitoring enabled")

        # CodeCarbon integration
        self.emissions_tracker = None
        if CODECARBON_AVAILABLE and use_wandb:
            self.emissions_tracker = EmissionsTracker(
                project_name="BitGen-Training",
                output_dir=str(self.output_dir),
                log_level='warning'
            )
            self.logger.info("🌱 CodeCarbon energy tracking enabled")

        # HuggingFace Hub integration
        self.push_to_hub = push_to_hub
        self.hf_integration = None
        if push_to_hub:
            from .huggingface_integration import HuggingFaceIntegration
            self.hf_integration = HuggingFaceIntegration(
                repo_name=hf_repo_name or "BitGen-Robot-Reasoning",
                organization=hf_organization,
                private=hf_private
            )
            self.hf_integration.create_repo()

    def setup_logging(self):
        """Setup logging for training"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_wandb(self, project: str, entity: Optional[str], run_name: Optional[str], tags: Optional[List[str]]):
        """Setup Weights & Biases logging"""
        wandb.init(
            project=project,
            entity=entity,
            config=self.config.__dict__,
            name=run_name,
            tags=tags
        )
        self.logger.info(f"WandB logging enabled: {project}/{entity}/{run_name}")

    def setup_optimizer(self, learning_rate: float = 1e-4):
        """Setup optimizer with balanced stability and convergence - RADICAL: Use CLIP-style high LR"""
        # CLIP uses 5e-4 for contrastive learning (much higher than language models)
        # Contrastive learning is more stable and can handle aggressive LR
        stable_lr = learning_rate * 10.0  # Increase from 1e-4 to 1e-3 (CLIP-style)

        # Use AdamW with balanced settings for good convergence
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=stable_lr,
            betas=(0.9, 0.999),  # FIXED: Standard Adam betas for better convergence
            weight_decay=0.01,
            eps=1e-8,
            amsgrad=False
        )
        
        # FIXED: Store total training steps for proper scheduler
        self.total_training_steps = None  # Will be set in train() method

        # CLIP-STYLE: Very short warmup, AGGRESSIVE cosine decay with restarts
        def lr_lambda(step):
            warmup_steps = 100  # CLIP uses ~100 steps warmup (very short!)
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing with warm restarts to escape plateaus
                if self.total_training_steps is None or self.total_training_steps <= warmup_steps:
                    # Fallback: use a large number if not set
                    total_steps = 100000
                else:
                    total_steps = self.total_training_steps

                # AGGRESSIVE DECAY: Decay more in early epochs to force learning
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                
                # Use cosine annealing with periodic restarts every 20% of training
                # This helps escape local minima / plateaus
                restart_period = 0.2  # Restart every 20% of training
                cycle_progress = (progress % restart_period) / restart_period
                
                # Cosine decay within each cycle
                cosine_decay = 0.5 * (1.0 + np.cos(np.pi * cycle_progress))
                
                # Overall decay envelope (slower)
                overall_decay = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
                
                # Combine: cycle restarts + overall decay
                # Min LR = 0.1 (10% of peak), Max LR varies with overall decay
                min_factor = 0.1
                max_factor = 0.3 + 0.7 * overall_decay  # Decays from 1.0 to 0.3
                
                return min_factor + (max_factor - min_factor) * cosine_decay

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.logger.info(f"✅ AGGRESSIVE LR: Base LR = {stable_lr} (10x), with cosine restarts every 20% of training")
        self.logger.info(f"   Warmup steps: 100 (CLIP-style)")
        self.logger.info(f"   LR range: {stable_lr * 0.1:.6f} - {stable_lr:.6f}")
        self.logger.info(f"   Periodic restarts help escape plateaus!")

        return optimizer, scheduler
    
    def setup_data_loaders(self, 
                          coco_data_path: str,
                          robot_data_path: Optional[str] = None,
                          batch_size: int = 4,
                          num_workers: int = 2,
                          train_split: float = 0.7):
        """Setup data loaders for multi-modal training with train/val split - OPTIMIZED FOR A4500 GPU"""

        # OPTIMIZED: Detect GPU capability and adjust settings
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            self.logger.info(f"Detected GPU: {gpu_name} with {total_memory:.1f}GB VRAM")

            # Optimize for A4500/A5000 class GPUs
            if total_memory > 15:  # A4500 has 20GB, A5000 has 24GB
                # Increase workers for powerful GPUs
                num_workers = min(16, psutil.cpu_count())  # Use more CPU cores
                self.logger.info(f"GPU-optimized: Using {num_workers} data loading workers")

        # COCO dataset for vision-language training
        full_dataset = COCODataset(
            coco_data_path,
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.config.vocab_size
        )
        
        # FIXED: Split dataset into train (70%) and val (30%)
        dataset_size = len(full_dataset)
        train_size = int(train_split * dataset_size)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )

        self.logger.info(f"📊 DATASET SPLIT:")
        self.logger.info(f"   Total samples: {dataset_size:,}")
        self.logger.info(f"   Training samples: {train_size:,} ({train_split*100:.0f}%)")
        self.logger.info(f"   Validation samples: {val_size:,} ({(1-train_split)*100:.0f}%)")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None
        )
        
        # Robot selection dataset (if available)
        robot_loader = None
        if robot_data_path and Path(robot_data_path).exists():
            robot_dataset = RobotSelectionDataset(robot_data_path)
            
            # CRITICAL: Update model config with actual robot types from dataset
            dataset_robot_types = robot_dataset.robot_types
            if len(dataset_robot_types) != self.config.num_robots:
                self.logger.warning(f"⚠️ Robot count mismatch detected!")
                self.logger.warning(f"   Config expects: {self.config.num_robots} robots")
                self.logger.warning(f"   Dataset has: {len(dataset_robot_types)} robots")
                self.logger.warning(f"   Updating model to match dataset...")
                
                # Update both config and model
                self.config.robot_types = dataset_robot_types
                self.config.num_robots = len(dataset_robot_types)
                self.model.config.robot_types = dataset_robot_types
                self.model.config.num_robots = len(dataset_robot_types)
                
                # Recreate robot selector with correct dimensions
                from .bitgen_model import RobotSelector
                self.model.robot_selector = RobotSelector(self.model.config).to(self.device)
                
                # Update confusion matrix size
                self.robot_confusion_matrix = np.zeros((len(dataset_robot_types), len(dataset_robot_types)))
                
                self.logger.info(f"✅ Model updated: {len(dataset_robot_types)} robots")
                self.logger.info(f"   Robot types: {dataset_robot_types}")
            
            # Custom collate function for robot dataset to handle variable-length lists
            def robot_collate_fn(batch):
                """Custom collate function for robot selection dataset"""
                return {
                    'input_ids': torch.stack([item['input_ids'] for item in batch]),
                    'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                    'labels': torch.stack([item['labels'] for item in batch]),
                    'robot_labels': torch.stack([item['robot_labels'] for item in batch]),
                    'selected_robots': [item['selected_robots'] for item in batch],  # Keep as list
                    'task_description': [item['task_description'] for item in batch],  # Keep as list
                    'num_robots_selected': torch.tensor([item['num_robots_selected'] for item in batch])
                }
            
            robot_loader = DataLoader(
                robot_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
                collate_fn=robot_collate_fn  # Use custom collate function
            )
        
        return train_loader, val_loader, robot_loader

    def train_step(self, batch: Dict, optimizer, grad_accum_steps: int = 1) -> Dict:
        """Single training step with enhanced monitoring"""
        self.model.train()
        step_start_time = time.time()

        # Extract batch data
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        images = batch.get('images')
        if images is not None:
            images = images.to(self.device)
        
        target_robot = batch.get('target_robot')
        if target_robot is not None:
            target_robot = target_robot.to(self.device)

        # Storage for intermediate outputs (for monitoring)
        attention_weights = None
        memory_bank = None
        memory_usage_info = {}
        text_features = None
        image_features = None

        try:
            # Forward pass with autocast for mixed precision
            with autocast('cuda'):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    return_attention_weights=True  # Request attention weights for monitoring
                )

                # Extract monitoring data from outputs
                if isinstance(outputs, dict):
                    attention_weights = outputs.get('attention_weights')
                    memory_bank = outputs.get('memory_bank')
                    memory_usage_info = outputs.get('memory_usage', {})
                    text_features = outputs.get('text_features')
                    image_features = outputs.get('image_features')

                total_loss, loss_dict = self.loss_fn(
                    outputs,
                    labels,
                    images=images,
                    target_robot=target_robot,
                    global_step=self.global_step
                )
                
                # CRITICAL: Scale loss by gradient accumulation steps
                # This ensures correct gradient magnitudes when accumulating
                total_loss = total_loss / grad_accum_steps
                
                # DEBUG: Log zero losses from loss_fn
                if total_loss.item() == 0.0 and self.global_step % 10 == 0:
                    self.logger.warning(f"⚠️ COCO loss_fn returned ZERO at step {self.global_step}")
                    self.logger.warning(f"   loss_dict: {loss_dict}")
                    self.logger.warning(f"   labels shape: {labels.shape}, valid labels: {(labels != -100).sum().item()}")
                    self.logger.warning(f"   images: {images is not None}, outputs keys: {list(outputs.keys())}")

                # FIXED: Only check for actual NaN/Inf, not zero values
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    self.logger.error(f"Invalid loss detected! Loss value: {total_loss.item()}")
                    # Create a small valid loss to maintain training
                    total_loss = torch.tensor(1.0 / grad_accum_steps, device=total_loss.device, requires_grad=True)
                    loss_dict['total_loss'] = total_loss
                else:
                    # DEBUG: Print the actual loss being used for training - only every 5000 steps
                    if self.global_step % 5000 == 0:
                        print(f"TRAINING: Using loss = {total_loss.item():.6f}")

        except RuntimeError as e:
            if "device-side assert" in str(e) or "index" in str(e).lower():
                self.logger.error(f"CUDA indexing error detected. Input shape: {input_ids.shape}")
                # Return a small loss to continue training - handle None optimizer
                lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5
                return {'total_loss': 1.0, 'learning_rate': lr, 'skipped_batch': True}
            else:
                raise

        # Check gradients before backward pass - FIXED: Don't check for zero loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.logger.error(f"Loss is NaN/Inf before backward pass: {total_loss.item()}")
            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5
            return {'total_loss': 1.0, 'learning_rate': lr, 'skipped_batch': True}

        # STABILITY: Check model parameters for NaN before backward
        has_nan_params = False
        for name, param in self.model.named_parameters():
            if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                self.logger.error(f"NaN/Inf detected in parameter {name} before backward!")
                has_nan_params = True
                break
        
        if has_nan_params:
            self.logger.error("Skipping batch due to NaN parameters")
            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5
            return {'total_loss': 1.0, 'learning_rate': lr, 'skipped_batch': True}

        # GRADIENT ACCUMULATION FIX: Always do backward pass first
        # Accumulate gradients regardless of whether we'll step optimizer
        total_loss.backward()

        # Only step optimizer if provided (after gradient accumulation)
        if optimizer is not None:
            # Enhanced gradient clipping with NaN checking
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self.logger.error(f"NaN/Inf gradients detected! Grad norm: {grad_norm}")
                optimizer.zero_grad()  # Clear bad gradients
                return {'total_loss': 0.0, 'learning_rate': optimizer.param_groups[0]['lr'], 'skipped_batch': True}

            if grad_norm > 10.0:  # Very large gradients
                self.logger.warning(f"Large gradient norm detected: {grad_norm:.4f}")

            # DIAGNOSTIC: Check gradient flow to encoders (every 50 steps)
            if self.global_step % 50 == 0:
                text_grad_norm = 0.0
                vision_grad_norm = 0.0
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.norm().item()
                        # FIXED: Match actual parameter names in BitGen model
                        if 'embedding' in name or 'attention_layers' in name:
                            text_grad_norm += param_grad_norm ** 2
                        elif 'dinov2' in name or 'cross_modal' in name:
                            vision_grad_norm += param_grad_norm ** 2
                
                text_grad_norm = text_grad_norm ** 0.5
                vision_grad_norm = vision_grad_norm ** 0.5
                
                self.logger.info(f"🔍 Gradient Flow (Step {self.global_step}): "
                               f"Text encoder={text_grad_norm:.6f}, "
                               f"Vision encoder={vision_grad_norm:.6f}, "
                               f"Total={grad_norm:.6f}")
                
                if text_grad_norm < 1e-6 and vision_grad_norm < 1e-6:
                    self.logger.error("❌ NO GRADIENTS flowing to encoders! Model is NOT learning!")

            # Log gradient flow to wandb (every 100 steps)
            if self.wandb_monitor and self.global_step % 100 == 0:
                self.wandb_monitor.log_gradient_flow(self.model, self.global_step)

            # Step optimizer and zero gradients for next accumulation cycle
            optimizer.step()
            optimizer.zero_grad()
        else:
            # During gradient accumulation (no optimizer step yet)
            grad_norm = 0.0

        # Calculate throughput
        step_time = time.time() - step_start_time
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        samples_per_sec = batch_size / step_time if step_time > 0 else 0
        tokens_per_sec = (batch_size * seq_len) / step_time if step_time > 0 else 0

        # Update metrics
        lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5
        # NOTE: total_loss is scaled for gradient accumulation, so multiply back for logging
        metrics = {
            'total_loss': total_loss.item() * grad_accum_steps,  # Unscale for proper reporting
            'learning_rate': lr,
            'gradient_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            'samples_per_sec': samples_per_sec,
            'tokens_per_sec': tokens_per_sec,
            'gpu_util': self._get_gpu_utilization()
        }
        
        # Add individual loss components
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            elif isinstance(value, dict):
                for k, v in value.items():
                    metrics[f"{key}_{k}"] = v
        
        # Add contrastive learning metrics if available
        if 'contrastive_features' in outputs:
            contrastive_feats = outputs['contrastive_features']
            if 'text_features' in contrastive_feats and 'image_features' in contrastive_feats:
                # Compute similarity matrix for visualization
                text_feat = contrastive_feats['text_features']
                image_feat = contrastive_feats['image_features']
                similarity = torch.matmul(text_feat, image_feat.t())
                
                # Compute retrieval accuracy (text->image and image->text)
                batch_size = text_feat.shape[0]
                labels = torch.arange(batch_size, device=text_feat.device)
                
                # Text-to-image retrieval accuracy (top-1)
                t2i_preds = similarity.argmax(dim=1)
                t2i_acc = (t2i_preds == labels).float().mean().item()
                
                # Image-to-text retrieval accuracy (top-1)
                i2t_preds = similarity.t().argmax(dim=1)
                i2t_acc = (i2t_preds == labels).float().mean().item()
                
                metrics['contrastive/t2i_accuracy'] = t2i_acc
                metrics['contrastive/i2t_accuracy'] = i2t_acc
                metrics['contrastive/avg_accuracy'] = (t2i_acc + i2t_acc) / 2
        
        # Comprehensive wandb logging
        if self.wandb_monitor and optimizer is not None:  # Only log when optimizer steps
            # Basic training metrics (every step)
            self.wandb_monitor.log_training_step(metrics, self.model, self.global_step)

            # Attention patterns (every 100 steps)
            if attention_weights is not None:
                self.wandb_monitor.log_attention_patterns(attention_weights, self.global_step, layer_idx=0)

            # Episodic memory (every 100 steps)
            if memory_bank is not None:
                self.wandb_monitor.log_episodic_memory(memory_bank, memory_usage_info, self.global_step)

            # Weight distributions (every 100 steps)
            self.wandb_monitor.log_weight_distributions(self.model, self.global_step)

            # Cross-modal fusion (every 100 steps)
            if text_features is not None and image_features is not None:
                similarity_matrix = None
                if 'similarity_matrix' in outputs:
                    similarity_matrix = outputs['similarity_matrix']
                self.wandb_monitor.log_cross_modal_fusion(
                    text_features, image_features, similarity_matrix, self.global_step
                )

            # CodeCarbon energy tracking (every 500 steps)
            if self.emissions_tracker and self.global_step % 500 == 0:
                emissions_data = {
                    'energy_consumed': getattr(self.emissions_tracker, '_total_energy', 0),
                    'emissions': getattr(self.emissions_tracker, '_total_emissions', 0),
                    'power': getattr(self.emissions_tracker, '_total_power', 0)
                }
                self.wandb_monitor.log_codecarbon(emissions_data, self.global_step)

        return metrics
    
    def train_robot_selection_step(self, batch: Dict, optimizer, grad_accum_steps: int = 1) -> Dict:
        """
        Robot selection training step with chain-of-thought reasoning (Tiny-R1 style)
        Multi-label classification with top-K robot selection
        """
        self.model.train()
        step_start_time = time.time()

        # Extract batch data
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        robot_labels = batch['robot_labels'].to(self.device)  # Multi-hot vector [B, num_robots]
        selected_robots = batch.get('selected_robots', [])  # For logging
        task_description = batch.get('task_description', [])

        try:
            # Forward pass with robot selection and reasoning traces enabled
            with autocast('cuda'):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_robot_selection=True,  # Enable robot selector
                    return_analysis_data=True  # Get reasoning traces
                )

                # Extract robot selection outputs
                robot_selection = outputs.get('robot_selection')
                if robot_selection is None or 'all_logits' not in robot_selection:
                    self.logger.error("Robot selection output not found!")
                    return {'total_loss': 0.0, 'robot_accuracy': 0.0, 'skipped_batch': True}

                robot_logits = robot_selection['all_logits']  # [B, num_robots] - Raw scores
                robot_probs = robot_selection['all_probs']  # [B, num_robots] - Probabilities for accuracy
                top_k_indices = robot_selection['top_k_indices']  # [B, top_k]

                # Check for NaN in logits BEFORE computing loss
                if torch.isnan(robot_logits).any() or torch.isinf(robot_logits).any():
                    self.logger.error(f"NaN/Inf detected in robot_logits!")
                    self.logger.error(f"  robot_logits stats: min={robot_logits.min().item():.4f}, max={robot_logits.max().item():.4f}")
                    self.logger.error(f"  robot_logits nan count: {torch.isnan(robot_logits).sum().item()}")
                    self.logger.error(f"  robot_logits inf count: {torch.isinf(robot_logits).sum().item()}")
                    lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5
                    return {'total_loss': 1.0 / grad_accum_steps, 'learning_rate': lr, 'skipped_batch': True}

                # Multi-label binary cross-entropy loss with logits (autocast-safe)
                robot_loss = F.binary_cross_entropy_with_logits(
                    robot_logits, robot_labels, reduction='mean'
                )

                # Check robot_loss for NaN immediately after computation
                if torch.isnan(robot_loss) or torch.isinf(robot_loss):
                    self.logger.error(f"NaN/Inf in robot_loss after BCE computation!")
                    self.logger.error(f"  robot_logits: min={robot_logits.min().item():.4f}, max={robot_logits.max().item():.4f}")
                    self.logger.error(f"  robot_labels: {robot_labels[0]}")
                    lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5
                    return {'total_loss': 1.0 / grad_accum_steps, 'learning_rate': lr, 'skipped_batch': True}

                # Optional: Language modeling loss on task description
                if 'logits' in outputs and 'labels' in batch:
                    labels = batch['labels'].to(self.device)
                    lm_loss, _ = self.loss_fn(outputs, labels, images=None, global_step=self.global_step)
                    
                    # Check lm_loss for NaN
                    if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                        self.logger.warning(f"NaN/Inf in lm_loss, skipping LM component")
                        total_loss = robot_loss
                    else:
                        total_loss = robot_loss + 0.1 * lm_loss  # Weight LM loss lower
                else:
                    total_loss = robot_loss

                # CRITICAL: Scale loss by gradient accumulation steps
                total_loss = total_loss / grad_accum_steps

                # Check for NaN/Inf in final total_loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    self.logger.error(f"Invalid robot total_loss after scaling: {total_loss.item()}")
                    lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5
                    return {'total_loss': 1.0 / grad_accum_steps, 'learning_rate': lr, 'skipped_batch': True}

        except Exception as e:
            self.logger.error(f"Error in robot selection training: {e}")
            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5
            return {'total_loss': 1.0, 'learning_rate': lr, 'skipped_batch': True}

        # GRADIENT ACCUMULATION FIX: Always do backward pass first
        total_loss.backward()

        # Only step optimizer if provided (after gradient accumulation)
        if optimizer is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self.logger.error(f"NaN/Inf gradients in robot training! Grad norm: {grad_norm}")
                optimizer.zero_grad()
                return {'total_loss': 0.0, 'learning_rate': optimizer.param_groups[0]['lr'], 'skipped_batch': True}

            # Step optimizer and zero gradients for next accumulation cycle
            optimizer.step()
            optimizer.zero_grad()
        else:
            # During gradient accumulation (no optimizer step yet)
            grad_norm = 0.0

        # Compute top-K accuracy
        robot_accuracy = self.compute_top_k_robot_accuracy(top_k_indices, robot_labels)

        # Update confusion matrix
        self.update_robot_confusion_matrix(top_k_indices, robot_labels)

        # Extract reasoning info
        reasoning_info = outputs.get('reasoning_info', {})
        reasoning_steps = reasoning_info.get('num_steps', 0)

        # Calculate metrics
        step_time = time.time() - step_start_time
        batch_size = input_ids.size(0)
        lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5

        # NOTE: total_loss is scaled for gradient accumulation, so multiply back for logging
        metrics = {
            'total_loss': total_loss.item() * grad_accum_steps,  # Unscale for proper reporting
            'robot_loss': robot_loss.item(),
            'robot_accuracy': robot_accuracy,
            'reasoning_steps': reasoning_steps,
            'learning_rate': lr,
            'gradient_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            'samples_per_sec': batch_size / step_time if step_time > 0 else 0,
            'batch_type': 'robot'
        }

        # Log reasoning trace periodically (every 100 robot batches)
        if self.global_step % 100 == 0 and len(task_description) > 0:
            self.log_reasoning_trace(batch, outputs, robot_selection)

        # WandB logging for robot selection
        if self.wandb_monitor and optimizer is not None:
            robot_metrics = {
                'robot/loss': robot_loss.item(),
                'robot/accuracy': robot_accuracy,
                'robot/reasoning_steps': reasoning_steps,
                'robot/top_1_confidence': robot_selection['top_k_probs'][:, 0].mean().item(),
                'robot/top_3_confidence': robot_selection['top_k_probs'].mean().item()
            }
            wandb.log(robot_metrics, step=self.global_step)

        return metrics
    
    def compute_top_k_robot_accuracy(self, top_k_indices, robot_labels):
        """
        Compute accuracy: What percentage of ground truth robots appear in top-K predictions?
        """
        batch_size = robot_labels.size(0)
        correct_predictions = 0

        for i in range(batch_size):
            true_robot_indices = torch.where(robot_labels[i] > 0.5)[0]  # Get ground truth robots
            predicted_indices = top_k_indices[i]  # Top-K predictions

            # Check how many true robots are in top-K predictions
            for true_idx in true_robot_indices:
                if true_idx in predicted_indices:
                    correct_predictions += 1

        # Accuracy = (correct predictions) / (total ground truth robots)
        total_true_robots = (robot_labels > 0.5).sum().item()
        accuracy = correct_predictions / total_true_robots if total_true_robots > 0 else 0.0

        return accuracy
    
    def update_robot_confusion_matrix(self, top_k_indices, robot_labels):
        """Update confusion matrix for robot selection tracking"""
        batch_size = robot_labels.size(0)

        for i in range(batch_size):
            true_robot_indices = torch.where(robot_labels[i] > 0.5)[0]  # Ground truth
            predicted_indices = top_k_indices[i]  # Top-K predictions

            # Update confusion matrix: true_robot x predicted_robot
            for true_idx in true_robot_indices:
                for pred_idx in predicted_indices:
                    self.robot_confusion_matrix[true_idx.item(), pred_idx.item()] += 1
    
    def log_reasoning_trace(self, batch, outputs, robot_selection):
        """Log chain-of-thought reasoning traces for debugging"""
        try:
            # Sample one example from batch
            task_desc = batch['task_description'][0] if len(batch['task_description']) > 0 else "N/A"
            selected_robots = batch['selected_robots'][0] if len(batch['selected_robots']) > 0 else []
            
            predicted_robots = robot_selection['top_k_robots'][0]
            confidences = robot_selection['top_k_probs'][0]
            
            reasoning_info = outputs.get('reasoning_info', {})
            reasoning_steps = reasoning_info.get('num_steps', 0)
            
            trace = f"""
            🤔 CHAIN-OF-THOUGHT REASONING TRACE (Step {self.global_step}):
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            📝 Task Description: {task_desc[:200]}...
            
            💭 Reasoning Process:
               • Reasoning Steps Taken: {reasoning_steps}
               • Multi-step chain-of-thought enabled
            
            🤖 Top-3 Predicted Robots:
               1. {predicted_robots[0]:<20} (confidence: {confidences[0]:.3f})
               2. {predicted_robots[1]:<20} (confidence: {confidences[1]:.3f})
               3. {predicted_robots[2]:<20} (confidence: {confidences[2]:.3f})
            
            ✅ Ground Truth Robots: {', '.join(selected_robots)}
            
            📊 Match Analysis:
               • Correct predictions: {sum(1 for r in predicted_robots if r in selected_robots)}/3
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """
            
            self.logger.info(trace)
            
            # Log to WandB as HTML
            if self.wandb_monitor:
                import wandb
                wandb.log({
                    'reasoning_trace': wandb.Html(trace.replace('\n', '<br>').replace(' ', '&nbsp;'))
                }, step=self.global_step)
                
        except Exception as e:
            self.logger.warning(f"Failed to log reasoning trace: {e}")
    
    def plot_robot_confusion_matrix(self, epoch):
        """Plot and save confusion matrix showing robot selection accuracy"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Normalize confusion matrix by row (true labels)
            row_sums = self.robot_confusion_matrix.sum(axis=1, keepdims=True)
            normalized_cm = np.divide(
                self.robot_confusion_matrix, 
                row_sums, 
                out=np.zeros_like(self.robot_confusion_matrix, dtype=float),
                where=row_sums != 0
            )
            
            # Plot
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                normalized_cm,
                xticklabels=self.config.robot_types,
                yticklabels=self.config.robot_types,
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                cbar_kws={'label': 'Prediction Frequency'}
            )
            plt.title(f'Robot Selection Confusion Matrix - Epoch {epoch+1}\n(Normalized by True Labels)', 
                     fontsize=14, fontweight='bold')
            plt.ylabel('True Robots', fontsize=12)
            plt.xlabel('Predicted Robots (Top-K)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save
            save_path = self.output_dir / f'confusion_matrix_epoch_{epoch+1}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"📊 Saved confusion matrix: {save_path}")
            
            # Log to WandB
            if self.wandb_monitor:
                import wandb
                wandb.log({
                    f'robot_confusion_matrix_epoch_{epoch+1}': wandb.Image(str(save_path))
                }, step=self.global_step)
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to plot confusion matrix: {e}")
    
    def evaluate(self, data_loader: DataLoader, max_eval_samples: int = 1000) -> Dict:
        """Evaluation step - FIXED: Use more samples for stable metrics"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        metrics = defaultdict(float)
        
        with torch.no_grad():
            # FIXED: Increased from 200 to 1000 samples for more stable eval metrics
            eval_batches = min(max_eval_samples // data_loader.batch_size, len(data_loader))

            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating", total=eval_batches)):
                if batch_idx >= eval_batches:
                    break  # Stop after max_eval_samples

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                images = batch.get('images')
                if images is not None:
                    images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, images=images)
                
                # Compute loss (use 0 for eval since we don't do stage transitions during eval)
                loss, loss_dict = self.loss_fn(outputs, labels, images=images, global_step=0)
                
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Accumulate metrics
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        metrics[key] += value.item() * batch_size
        
        # Average metrics
        avg_metrics = {
            'eval_loss': total_loss / total_samples if total_samples > 0 else 0.0
        }
        
        for key, value in metrics.items():
            avg_metrics[f"eval_{key}"] = value / total_samples if total_samples > 0 else 0.0

        return avg_metrics
    
    def save_checkpoint(self, optimizer: optim.Optimizer, scheduler, suffix: str = ""):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'performance_summary': self.performance_tracker.get_summary()
        }
        
        filename = f"bitgen_checkpoint{suffix}.pt"
        torch.save(checkpoint, self.output_dir / filename)
        self.logger.info(f"Saved checkpoint: {filename}")
        
        # Export for embedded deployment
        if suffix == "_best":
            self.model.export_for_embedded(str(self.output_dir / "bitgen_embedded.bin"))
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: optim.Optimizer = None, scheduler = None):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self,
             coco_data_path: str,
             robot_data_path: Optional[str] = None,
             num_epochs: int = 10,
             batch_size: int = 4,
             learning_rate: float = 1e-4,
             eval_steps: int = 500,
             save_steps: int = 1000,
             max_memory_mb: int = 1024):
        """Main training loop - OPTIMIZED FOR A40 GPU"""

        self.logger.info("Starting BitGen training for embedded deployment")
        
        # GPU-OPTIMIZED: Detect and optimize for high-end GPUs (A40, A100, etc.)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"Detected GPU: {gpu_name} with {total_vram_gb:.1f}GB VRAM")

            # OPTIMIZE FOR A40/A100: Maximize utilization for high-end GPUs
            if total_vram_gb > 40:  # A40 has 46GB, A100 has 40-80GB
                # AGGRESSIVE: Increase batch size to use more GPU (currently only 13.5GB/46GB used!)
                # With unfrozen DINOv2 (86M params), we can handle larger batches
                optimized_batch_size = max(batch_size, 256)  # Increased from 192 to 256

                # Target 42GB memory usage (~91% of 46GB, leaving 4GB for safety)
                optimized_memory_mb = min(42000, int(total_vram_gb * 910))  # Use ~91% of VRAM

                self.logger.info(f"🚀 A40/A100 AGGRESSIVE OPTIMIZATION:")
                self.logger.info(f"   Original batch_size: {batch_size} → Optimized: {optimized_batch_size}")
                self.logger.info(f"   Original memory_mb: {max_memory_mb} → Optimized: {optimized_memory_mb}")
                self.logger.info(f"   Previous usage: 13.5GB/46GB (29%) → Target: 42GB (91%)")

                batch_size = optimized_batch_size
                max_memory_mb = optimized_memory_mb

            elif total_vram_gb > 15:  # A4500 has 20GB, A5000 has 24GB
                # Increase batch size significantly for better GPU utilization
                optimized_batch_size = max(batch_size, 32)  # Minimum 32 for A4500

                # Increase memory limit to utilize more GPU memory
                optimized_memory_mb = min(16000, int(total_vram_gb * 800))  # Use ~80% of VRAM

                self.logger.info(f"🚀 A4500/A5000 OPTIMIZATION ENABLED:")
                self.logger.info(f"   Original batch_size: {batch_size} → Optimized: {optimized_batch_size}")
                self.logger.info(f"   Original memory_mb: {max_memory_mb} → Optimized: {optimized_memory_mb}")

                batch_size = optimized_batch_size
                max_memory_mb = optimized_memory_mb
            else:
                self.logger.info(f"Using standard configuration for {gpu_name}")

        # Check memory constraints
        memory_info = self.memory_utils.compute_memory_usage()
        self.logger.info(f"Initial memory usage: {memory_info['rss_mb']:.1f} MB")
        
        # GPU-OPTIMIZED: Smart gradient accumulation for large batch processing
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

            if vram_gb > 40:  # A40/A100 class GPUs
                # OPTIMIZED: Moderate increase to 320 effective batch size (leaves safety margin)
                # Physical batch = 160 per step to avoid OOM
                batch_size = 320  # Effective batch size
                grad_accum_steps = 2  # Keep at 2 for stability
                actual_batch_size = batch_size // grad_accum_steps  # 320 / 2 = 160 per step

                self.logger.info(f"💡 A40 optimization (balanced for 80% VRAM):")
                self.logger.info(f"   Physical batch size: {actual_batch_size} (was 96, +67%)")
                self.logger.info(f"   Gradient accumulation steps: {grad_accum_steps}")
                self.logger.info(f"   Effective batch size: {batch_size} (was 192, +67%)")
                self.logger.info(f"   Expected VRAM usage: ~35GB (was ~23GB, safe 80% of 46GB)")

            elif vram_gb > 15:  # A4500/A5000 class GPUs
                # For A4500+: Use minimal gradient accumulation
                grad_accum_steps = max(1, batch_size // 16)
                actual_batch_size = batch_size // grad_accum_steps

                # Ensure we're using significant batch sizes
                if actual_batch_size < 8:
                    actual_batch_size = 8
                    grad_accum_steps = batch_size // actual_batch_size
            else:
                # Conservative for smaller GPUs
                grad_accum_steps = self.memory_utils.gradient_accumulation_steps(batch_size, max_memory_mb)
                actual_batch_size = batch_size // grad_accum_steps
        else:
            # CPU fallback
            grad_accum_steps = self.memory_utils.gradient_accumulation_steps(batch_size, max_memory_mb)
            actual_batch_size = batch_size // grad_accum_steps

        self.logger.info(f"🎯 FINAL TRAINING CONFIG:")
        self.logger.info(f"   Effective batch size: {batch_size}")
        self.logger.info(f"   Actual batch size: {actual_batch_size}")
        self.logger.info(f"   Gradient accumulation: {grad_accum_steps}")
        self.logger.info(f"   Memory limit: {max_memory_mb}MB")

        # MEMORY OPTIMIZATION: Enable PyTorch memory optimizations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set environment variable for better memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
            
            # STABILITY: Set CUDA matmul precision for stability
            torch.backends.cuda.matmul.allow_tf32 = True  # Faster but still accurate
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Auto-tune kernels for performance
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            # CRITICAL: Reduce memory fragmentation
            torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use max 95% of GPU memory
            
            self.logger.info("✅ Enabled PyTorch CUDA memory optimization (expandable segments + max_split_size)")
            self.logger.info("✅ Enabled TF32 precision for faster training")
            self.logger.info("✅ Set memory fraction to 95% to prevent OOM")

        # CRITICAL FIX: Setup data loaders FIRST to get steps_per_epoch
        # Then set total_training_steps BEFORE creating scheduler
        train_loader, val_loader, robot_loader = self.setup_data_loaders(
            coco_data_path, robot_data_path, actual_batch_size
        )
        
        # Calculate total training steps for LR scheduler
        steps_per_epoch = len(train_loader)
        self.total_training_steps = steps_per_epoch * num_epochs
        
        self.logger.info(f"📈 LEARNING RATE SCHEDULE:")
        self.logger.info(f"   Steps per epoch: {steps_per_epoch:,}")
        self.logger.info(f"   Total training steps: {self.total_training_steps:,}")
        self.logger.info(f"   Warmup steps: 500")
        self.logger.info(f"   LR will decay from {learning_rate * 3.0} to {learning_rate * 3.0 * 0.1}")
        
        # NOW create optimizer and scheduler (after total_training_steps is set)
        optimizer, scheduler = self.setup_optimizer(learning_rate)

        # DIAGNOSTIC: Log data loader details
        self.logger.info(f"📊 DATA LOADER DETAILS:")
        self.logger.info(f"   Dataset samples: {len(train_loader.dataset):,}")
        self.logger.info(f"   Batch size: {train_loader.batch_size}")
        self.logger.info(f"   Expected batches per epoch: {len(train_loader):,}")
        self.logger.info(f"   Num workers: {train_loader.num_workers}")
        self.logger.info(f"   Drop last: {train_loader.drop_last}")

        # Training loop with GPU optimization
        self.model.train()
        scaler = GradScaler('cuda')

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = defaultdict(list)
            
            # DIAGNOSTIC: Start of epoch logging
            import time
            epoch_start_time = time.time()
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🚀 STARTING EPOCH {epoch+1}/{num_epochs}")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"   Dataset size (verified): {len(train_loader.dataset):,} samples")
            self.logger.info(f"   Expected batches: {len(train_loader):,}")
            self.logger.info(f"   Batch size: {actual_batch_size}")
            self.logger.info(f"   Effective batch size (with grad accum): {batch_size}")

            # MEMORY OPTIMIZATION: Clear cache at start of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"   GPU memory at epoch start: {gpu_mem_before:.2f}GB")

            # INTERLEAVED TRAINING: 90% COCO vision-language, 10% robot selection
            robot_iter = iter(robot_loader) if robot_loader else None
            
            if robot_loader:
                desc = f"Epoch {epoch+1}/{num_epochs} (90% COCO + 10% Robot Selection)"
                self.logger.info(f"   📊 Interleaved training enabled: Every 10th batch = Robot Selection")
            else:
                desc = f"Epoch {epoch+1}/{num_epochs} (COCO only)"
            
            # FIXED: Total should be based on optimizer steps, not data loader length
            total_iterations = len(train_loader) // grad_accum_steps

            # FIXED: Progress bar shows optimizer steps, not batch steps
            progress_bar = tqdm(total=total_iterations, desc=desc)

            accumulated_loss = 0.0
            accumulated_count = 0  # Track non-skipped batches
            batch_count = 0
            optimizer_step_count = 0  # FIXED: Track actual optimizer steps

            for step, coco_batch in enumerate(train_loader):
                # INTERLEAVED TRAINING LOGIC:
                # Every 10th batch (steps 0, 10, 20, ...) = Robot selection training
                # Other batches (steps 1-9, 11-19, ...) = COCO vision-language training
                if robot_iter and step % 10 == 0:
                    # Robot selection training with chain-of-thought reasoning
                    try:
                        robot_batch = next(robot_iter)
                    except StopIteration:
                        # Restart robot iterator if exhausted
                        robot_iter = iter(robot_loader)
                        robot_batch = next(robot_iter)
                    
                    # GPU-OPTIMIZED: Gradient accumulation for robot selection
                    with autocast('cuda'):
                        step_metrics = self.train_robot_selection_step(
                            robot_batch, 
                            optimizer if (step + 1) % grad_accum_steps == 0 else None,
                            grad_accum_steps=grad_accum_steps
                        )
                    step_metrics['batch_type'] = 'robot'
                    
                else:
                    # Regular vision-language training (COCO)
                    batch = coco_batch
                    
                    # GPU-OPTIMIZED: Gradient accumulation for large effective batch sizes
                    with autocast('cuda'):
                        step_metrics = self.train_step(
                            batch, 
                            optimizer if (step + 1) % grad_accum_steps == 0 else None,
                            grad_accum_steps=grad_accum_steps
                        )
                    step_metrics['batch_type'] = 'coco'

                # Skip accumulation if batch was skipped
                if not step_metrics.get('skipped_batch', False):
                    current_loss = step_metrics.get('total_loss', 0.0)
                    accumulated_loss += current_loss
                    accumulated_count += 1
                    
                    # DEBUG: Log losses more frequently to diagnose zero loss issue
                    if step % 20 == 0 or current_loss == 0.0:
                        batch_type = step_metrics.get('batch_type', 'unknown')
                        is_optim_step = (step + 1) % grad_accum_steps == 0
                        self.logger.info(f"DEBUG Step {step} (optim={is_optim_step}): "
                                       f"{batch_type} loss={current_loss:.6f}, "
                                       f"accum={accumulated_loss:.6f}, count={accumulated_count}")
                else:
                    # Log skipped batches
                    self.logger.warning(f"⚠️ Step {step}: Batch SKIPPED (type={step_metrics.get('batch_type', 'unknown')})")

                # Only step optimizer after accumulating gradients
                if (step + 1) % grad_accum_steps == 0:
                    # NOTE: step_metrics already has the correct loss (from last batch)
                    # We DON'T average again because train_step already returned unscaled loss for logging
                    # Just use the last batch's metrics (which represents the optimizer step)
                    
                    if accumulated_count == 0:
                        # All batches were skipped - use safe default
                        step_metrics['total_loss'] = 1.0
                        self.logger.warning(f"⚠️ Step {step}: All {grad_accum_steps} batches were skipped!")
                    
                    # STABILITY: Check for loss divergence
                    current_loss_val = step_metrics.get('total_loss', 0.0)
                    if current_loss_val > 0.0:
                        if self.loss_ema is None:
                            self.loss_ema = current_loss_val
                        else:
                            self.loss_ema = self.loss_ema_alpha * current_loss_val + (1 - self.loss_ema_alpha) * self.loss_ema
                            
                            # Check for divergence
                            if current_loss_val > self.loss_ema * self.divergence_threshold:
                                self.logger.warning(f"⚠️ LOSS DIVERGENCE DETECTED at step {step}!")
                                self.logger.warning(f"   Current loss: {current_loss_val:.4f}")
                                self.logger.warning(f"   EMA loss: {self.loss_ema:.4f}")
                                self.logger.warning(f"   Ratio: {current_loss_val / self.loss_ema:.2f}x")
                    
                    # DEBUG: Log the loss on optimizer steps
                    if step % 20 == 0:
                        ema_str = f", EMA={self.loss_ema:.4f}" if self.loss_ema else ""
                        self.logger.info(f"✓ OPTIMIZER STEP {step}: "
                                       f"Loss = {current_loss_val:.6f}{ema_str}")
                    
                    accumulated_loss = 0.0
                    accumulated_count = 0

                    # Update metrics
                    for key, value in step_metrics.items():
                        epoch_metrics[key].append(value)

                    # Update performance tracker (filter out non-numeric values)
                    numeric_metrics = {k: v for k, v in step_metrics.items() 
                                      if isinstance(v, (int, float, torch.Tensor)) and k != 'batch_type'}
                    self.performance_tracker.update(numeric_metrics)

                    # CRITICAL FIX: Step scheduler ONLY on optimizer steps (not on every batch!)
                    # This was causing LR to advance too fast (2x speed with grad_accum=2)
                    scheduler.step()
                    
                    # Increment global step counter for proper LR scheduling
                    self.global_step += 1

                    # FIXED: Update progress bar only on optimizer steps
                    optimizer_step_count += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'loss': f"{step_metrics['total_loss']:.4f}",
                        'lr': f"{step_metrics['learning_rate']:.6f}",
                        'gpu_util': f"{self._get_gpu_utilization():.0f}%"
                    })

                    # Evaluation
                    if self.global_step % eval_steps == 0:
                        eval_metrics = self.evaluate(val_loader)  # FIXED: Use validation loader

                        avg_eval_loss = eval_metrics['eval_loss']
                        avg_train_loss = step_metrics['total_loss']

                        self.logger.info(f"Step {self.global_step}: Eval Loss = {avg_eval_loss:.4f}")

                        # FIXED: Check for massive overfitting (avoid division by zero)
                        if avg_train_loss > 0 and avg_eval_loss > avg_train_loss * 10:
                            self.logger.warning(f"⚠️⚠️⚠️ SEVERE OVERFITTING DETECTED!")
                            self.logger.warning(f"   Train Loss: {avg_train_loss:.4f}")
                            self.logger.warning(f"   Eval Loss: {avg_eval_loss:.4f}")
                            self.logger.warning(f"   Gap: {avg_eval_loss/avg_train_loss:.1f}x")
                            self.logger.warning(f"   Consider: reducing model size, adding dropout, or reducing learning rate")
                        elif avg_train_loss == 0:
                            self.logger.warning(f"⚠️ WARNING: Training loss is zero!")
                            self.logger.warning(f"   This indicates batches are being skipped or loss is not being calculated.")
                            self.logger.warning(f"   Eval Loss: {avg_eval_loss:.4f}")

                        # Save best model
                        if avg_eval_loss < self.best_loss:
                            self.best_loss = avg_eval_loss
                            self.save_checkpoint(optimizer, scheduler, "_best")

                        # Log to wandb
                        if wandb.run:
                            wandb.log(eval_metrics, step=self.global_step)

                    # Regular checkpointing
                    if self.global_step % save_steps == 0:
                        self.save_checkpoint(optimizer, scheduler, f"_step_{self.global_step}")

                    # GPU memory monitoring - MORE AGGRESSIVE
                    if self.global_step % 50 == 0:
                        if torch.cuda.is_available():
                            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB

                            # Clear cache if memory usage is high
                            if gpu_memory_used > total_vram_gb * 0.7:  # 70% threshold (lowered from 90%)
                                self.logger.warning(f"High GPU memory: {gpu_memory_used:.1f}GB/{total_vram_gb:.1f}GB - clearing cache")
                                torch.cuda.empty_cache()

            # FIXED: Close progress bar properly
            progress_bar.close()

            # End of epoch - DIAGNOSTIC LOGGING
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # Comprehensive epoch summary
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"✅ COMPLETED EPOCH {epoch+1}/{num_epochs}")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"   Data batches processed: {step+1:,}")
            self.logger.info(f"   Optimizer steps: {optimizer_step_count:,}")
            self.logger.info(f"   Expected optimizer steps: {total_iterations:,}")
            if optimizer_step_count != total_iterations:
                self.logger.warning(f"   ⚠️ MISMATCH: {optimizer_step_count:,} optimizer steps but expected {total_iterations:,}!")
            self.logger.info(f"   Epoch duration: {epoch_duration/3600:.2f} hours ({epoch_duration/60:.1f} minutes)")
            self.logger.info(f"   Samples per second: {len(train_loader.dataset)/epoch_duration:.2f}")
            self.logger.info(f"   Avg seconds per optimizer step: {epoch_duration/optimizer_step_count:.2f}")

            # Log epoch metrics - FILTER OUT NON-NUMERIC VALUES
            epoch_avg = {}
            for key, values in epoch_metrics.items():
                # Skip non-numeric keys like 'batch_type'
                if len(values) > 0 and isinstance(values[0], (int, float, np.number)):
                    epoch_avg[key] = np.mean(values)
                elif len(values) > 0 and not isinstance(values[0], str):
                    # Try to convert to float if possible
                    try:
                        epoch_avg[key] = np.mean([float(v) for v in values])
                    except (ValueError, TypeError):
                        # Skip if can't convert to numeric
                        pass
            
            self.logger.info(f"   Avg Loss: {epoch_avg.get('total_loss', 0.0):.4f}")
            self.logger.info(f"   Current LR: {scheduler.get_last_lr()[0]:.6f}")

            # Log GPU utilization summary
            if torch.cuda.is_available():
                gpu_util = self._get_gpu_utilization()
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"   GPU Utilization: {gpu_util:.1f}%")
                self.logger.info(f"   GPU Memory: {gpu_memory_used:.1f}GB")

            self.logger.info(f"{'='*80}\n")

            if wandb.run:
                wandb.log(epoch_avg, step=self.global_step)
            
            # Push to HuggingFace after each epoch
            if self.hf_integration:
                try:
                    self.logger.info(f"🤗 Pushing checkpoint to HuggingFace for epoch {epoch+1}...")
                    self.hf_integration.push_model_checkpoint(
                        model=self.model,
                        config=self.config,
                        epoch=epoch + 1,
                        metrics=epoch_avg
                    )
                    self.logger.info(f"✅ Model pushed to HuggingFace: epoch {epoch+1}")
                except Exception as e:
                    self.logger.error(f"❌ HuggingFace push failed for epoch {epoch+1}: {e}")
            
            # Plot robot selection confusion matrix at end of each epoch
            if robot_loader:
                self.logger.info(f"📊 Generating robot selection confusion matrix for epoch {epoch+1}...")
                self.plot_robot_confusion_matrix(epoch)
                
                # Calculate and log robot accuracy for this epoch
                total_predictions = self.robot_confusion_matrix.sum()
                if total_predictions > 0:
                    # Accuracy = correct predictions / total predictions
                    correct_predictions = np.trace(self.robot_confusion_matrix)  # Diagonal sum
                    epoch_robot_accuracy = correct_predictions / total_predictions
                    self.robot_accuracy_per_epoch.append(epoch_robot_accuracy)
                    self.logger.info(f"   Robot Selection Accuracy: {epoch_robot_accuracy:.2%}")
                    
                    if self.wandb_monitor:
                        wandb.log({
                            'robot/epoch_accuracy': epoch_robot_accuracy
                        }, step=self.global_step)
        
        # Final checkpoint
        self.save_checkpoint(optimizer, scheduler, "_final")
        self.logger.info("Training completed!")
        
        # Performance summary
        summary = self.performance_tracker.get_summary()
        self.logger.info("Training Summary:")
        for metric, stats in summary.items():
            self.logger.info(f"  {metric}: {stats['current']:.4f} (trend: {stats['trend']})")
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0

    def merge_batches(self, coco_batch: Dict, robot_batch: Dict) -> Dict:
        """Merge COCO and robot selection batches"""
        merged = coco_batch.copy()
        
        # Add robot selection targets
        if 'target_robot' in robot_batch:
            merged['target_robot'] = robot_batch['target_robot']
        
        return merged
    
    def inference_benchmark(self, test_data_path: str, max_samples: int = 100):
        """Benchmark inference performance for embedded deployment"""
        self.logger.info("Running inference benchmark...")
        
        # Load test data
        test_dataset = COCODataset(test_data_path, max_seq_len=self.config.max_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        self.model.eval()
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Inference benchmark")):
                if i >= max_samples:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                images = batch.get('images')
                if images is not None:
                    images = images.to(self.device)
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                # Generate text (embedded-style)
                generated, cache = self.model.generate_embedded(
                    input_ids, max_length=50, cache=None
                )
                
                end_time.record()
                torch.cuda.synchronize()
                
                inference_time = start_time.elapsed_time(end_time)
                inference_times.append(inference_time)
                
                # Monitor memory
                memory_info = self.memory_utils.compute_memory_usage()
                memory_usage.append(memory_info['rss_mb'])
        
        # Benchmark results
        avg_inference_time = np.mean(inference_times)
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        
        benchmark_results = {
            'avg_inference_time_ms': avg_inference_time,
            'max_inference_time_ms': np.max(inference_times),
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'samples_tested': len(inference_times)
        }
        
        self.logger.info("Inference Benchmark Results:")
        for key, value in benchmark_results.items():
            self.logger.info(f"  {key}: {value:.2f}")
        
        # Save benchmark results
        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        return benchmark_results

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train BitGen model for embedded deployment")
    parser.add_argument("--coco_data", type=str, required=True, help="Path to COCO dataset")
    parser.add_argument("--robot_data", type=str, help="Path to robot selection dataset")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["nano", "tiny", "small"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--max_memory_mb", type=int, default=1024)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Create model configuration
    config = BitGenConfig()
    
    # Initialize trainer
    trainer = BitGenTrainer(
        config=config,
        model_size=args.model_size,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        optimizer, scheduler = trainer.setup_optimizer(args.learning_rate)
        trainer.load_checkpoint(args.checkpoint, optimizer, scheduler)
    
    # Start training
    trainer.train(
        coco_data_path=args.coco_data,
        robot_data_path=args.robot_data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_memory_mb=args.max_memory_mb
    )

if __name__ == "__main__":
    main()
