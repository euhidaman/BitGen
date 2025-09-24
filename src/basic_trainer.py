"""
Basic BitGen Training System
Cross-platform training without Raspberry Pi dependencies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict

class BasicBitGenTrainer:
    """Basic BitGen trainer for any platform without Pi dependencies"""

    def __init__(self,
                 config,
                 model_size: str = 'tiny',
                 output_dir: str = 'checkpoints',
                 # HuggingFace integration
                 hf_repo_name: str = "BitGen-Reasoning",
                 hf_organization: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 push_to_hub: bool = False,
                 # WandB integration
                 wandb_project: str = "bitgen-training",
                 wandb_entity: str = "babylm-ntust",
                 use_wandb: bool = False):

        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize model
        from src.bitgen_model import create_bitgen_model
        self.model = create_bitgen_model(model_size)

        # Multi-GPU setup - detect and use all available GPUs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_multi_gpu = self.num_gpus > 1

        if torch.cuda.is_available():
            print(f"🚀 CUDA available with {self.num_gpus} GPU(s)")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

        # Setup device and multi-GPU training
        if self.use_multi_gpu:
            print(f"✅ Enabling DataParallel training across {self.num_gpus} GPUs")
            self.device = torch.device('cuda:0')
            self.model = self.model.to(self.device)
            # Wrap model with DataParallel for multi-GPU training
            self.model = nn.DataParallel(self.model)
            self.effective_batch_size_multiplier = self.num_gpus
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            self.effective_batch_size_multiplier = 1
            print("✅ Using single GPU training")
        else:
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
            self.effective_batch_size_multiplier = 1
            print("⚠️  Using CPU training (no CUDA available)")

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Initialize integrations if requested
        self.hf_integration = None
        self.wandb_integration = None

        if push_to_hub:
            try:
                from src.huggingface_integration import setup_huggingface_integration
                self.hf_integration = setup_huggingface_integration(
                    model_name=hf_repo_name,
                    organization=hf_organization,
                    token=hf_token
                )
                print(f"✅ HuggingFace integration: {self.hf_integration.repo_id}")
            except Exception as e:
                print(f"❌ HuggingFace setup failed: {e}")

        if use_wandb:
            try:
                from src.wandb_integration import setup_wandb_integration
                self.wandb_integration = setup_wandb_integration(
                    project_name=wandb_project,
                    entity=wandb_entity,
                    config=config.__dict__,
                    tags=["bitgen", "basic-training", model_size]
                )
                print(f"✅ WandB integration: {wandb_entity}/{wandb_project}")
            except Exception as e:
                print(f"❌ WandB setup failed: {e}")

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train(self, coco_data_path: str, num_epochs: int = 10, batch_size: int = 8, learning_rate: float = 1e-4):
        """Enhanced training loop with multi-GPU optimization"""

        # Adjust batch size for multi-GPU training
        if self.use_multi_gpu:
            # Scale batch size with number of GPUs for better utilization
            effective_batch_size = batch_size * self.effective_batch_size_multiplier
            actual_batch_size = batch_size  # Per-GPU batch size
            self.logger.info(f"🚀 Multi-GPU training with {self.num_gpus} GPUs")
            self.logger.info(f"   Per-GPU batch size: {actual_batch_size}")
            self.logger.info(f"   Effective total batch size: {effective_batch_size}")
        else:
            actual_batch_size = batch_size
            effective_batch_size = batch_size

        self.logger.info(f"Starting BitGen {self.config.__class__.__name__} training")
        self.logger.info(f"Device: {self.device}, Multi-GPU: {self.use_multi_gpu}")

        # Setup mixed precision training for faster training on modern GPUs
        use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        if use_amp:
            self.logger.info("✅ Using Automatic Mixed Precision (AMP) for faster training")

        # Setup optimizer with learning rate scaling for multi-GPU
        if self.use_multi_gpu:
            # Scale learning rate with effective batch size
            scaled_lr = learning_rate * np.sqrt(self.effective_batch_size_multiplier)
            self.logger.info(f"📈 Scaling learning rate: {learning_rate:.2e} → {scaled_lr:.2e}")
        else:
            scaled_lr = learning_rate

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=scaled_lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            eps=1e-8
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Setup data loader with optimized settings for multi-GPU
        train_loader = self._setup_data_loader(coco_data_path, actual_batch_size)

        # Initialize loss function
        from src.adaptive_loss import BitGenLoss
        loss_fn = BitGenLoss(self.config, self.config.vocab_size)

        # Training metrics tracking
        total_steps = 0
        best_loss = float('inf')

        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = defaultdict(list)
            epoch_start_time = time.time()

            self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # Set model to training mode
            self.model.train()

            for step, batch in enumerate(train_loader):
                step_start_time = time.time()

                # Move to device (automatically distributed across GPUs with DataParallel)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                images = batch.get('images')
                if images is not None:
                    images = images.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast() if use_amp else torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        images=images,
                        return_robot_selection=True
                    )

                    # Compute loss
                    total_loss, loss_dict = loss_fn(outputs, labels, images=images)

                # Backward pass with gradient scaling
                optimizer.zero_grad()

                if use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                self.global_step += 1
                total_steps += 1
                step_time = time.time() - step_start_time

                # Calculate throughput metrics
                tokens_processed = input_ids.numel()
                if self.use_multi_gpu:
                    tokens_processed *= self.num_gpus  # Account for all GPUs

                throughput = tokens_processed / step_time

                # Collect metrics
                step_metrics = {
                    'loss': total_loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'throughput_tokens_per_sec': throughput,
                    'step_time_ms': step_time * 1000,
                    'gpu_count': self.num_gpus,
                    'effective_batch_size': effective_batch_size
                }

                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)

                # Log metrics with multi-GPU info
                if step % 50 == 0:
                    gpu_memory_info = ""
                    if torch.cuda.is_available():
                        if self.use_multi_gpu:
                            memory_mb = sum(torch.cuda.memory_allocated(i) / 1e6 for i in range(self.num_gpus))
                            gpu_memory_info = f", GPU Memory: {memory_mb:.1f}MB total"
                        else:
                            memory_mb = torch.cuda.memory_allocated() / 1e6
                            gpu_memory_info = f", GPU Memory: {memory_mb:.1f}MB"

                    self.logger.info(
                        f"Epoch {epoch+1}, Step {step}: "
                        f"Loss={total_loss.item():.4f}, "
                        f"Throughput={throughput:.0f} tokens/s"
                        f"{gpu_memory_info}"
                    )

                # Log to WandB with detailed multi-GPU metrics
                if self.wandb_integration and step % 10 == 0:
                    wandb_metrics = {
                        'train/loss': total_loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'performance/throughput_tokens_per_sec': throughput,
                        'performance/step_time_ms': step_time * 1000,
                        'system/gpu_count': self.num_gpus,
                        'system/effective_batch_size': effective_batch_size,
                    }

                    # Add GPU memory metrics
                    if torch.cuda.is_available():
                        for i in range(self.num_gpus):
                            memory_mb = torch.cuda.memory_allocated(i) / 1e6
                            wandb_metrics[f'system/gpu_{i}_memory_mb'] = memory_mb

                    self.wandb_integration.run.log(wandb_metrics, step=self.global_step)

            # End of epoch processing
            scheduler.step()
            epoch_time = time.time() - epoch_start_time
            epoch_avg_loss = np.mean(epoch_metrics['loss'])
            epoch_avg_throughput = np.mean(epoch_metrics['throughput_tokens_per_sec'])

            # Save best model
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                self._save_best_checkpoint(optimizer, scheduler)

            # Log epoch summary with multi-GPU performance
            epoch_summary = {
                'epoch': epoch + 1,
                'avg_loss': epoch_avg_loss,
                'avg_throughput_tokens_per_sec': epoch_avg_throughput,
                'epoch_time_minutes': epoch_time / 60,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'gpu_count': self.num_gpus,
                'effective_batch_size': effective_batch_size
            }

            # Log to WandB
            if self.wandb_integration:
                self.wandb_integration.log_training_metrics({
                    'epoch/loss': epoch_avg_loss,
                    'epoch/learning_rate': epoch_summary['learning_rate'],
                    'epoch/throughput_tokens_per_sec': epoch_avg_throughput,
                    'epoch/time_minutes': epoch_time / 60,
                    'epoch/gpu_utilization': self.num_gpus,
                }, epoch=epoch)

            # Push to HuggingFace
            if self.hf_integration:
                try:
                    # For DataParallel, need to access the underlying module
                    model_to_save = self.model.module if self.use_multi_gpu else self.model

                    self.hf_integration.push_model_checkpoint(
                        model=model_to_save,
                        config=self.config,
                        epoch=epoch + 1,
                        metrics=epoch_summary
                    )
                    self.logger.info(f"✅ Model pushed to HuggingFace for epoch {epoch+1}")
                except Exception as e:
                    self.logger.error(f"❌ HuggingFace push failed: {e}")

            # Save checkpoint
            self._save_checkpoint(optimizer, scheduler)

            self.logger.info(f"Epoch {epoch+1} completed in {epoch_time/60:.1f} minutes")
            self.logger.info(f"  Average loss: {epoch_avg_loss:.4f}")
            self.logger.info(f"  Average throughput: {epoch_avg_throughput:.0f} tokens/s")
            if self.use_multi_gpu:
                self.logger.info(f"  Multi-GPU efficiency: {epoch_avg_throughput / self.num_gpus:.0f} tokens/s per GPU")

        # Training completed
        # Finish WandB run with final summary
        if self.wandb_integration:
            final_summary = {
                'final/best_loss': best_loss,
                'final/total_epochs': num_epochs,
                'final/gpu_count': self.num_gpus,
                'final/effective_batch_size': effective_batch_size,
                'final/avg_throughput': epoch_avg_throughput,
            }
            self.wandb_integration.finish_run(final_summary)

        self.logger.info("🎉 Training completed!")
        self.logger.info(f"   Best loss achieved: {best_loss:.4f}")
        self.logger.info(f"   GPUs utilized: {self.num_gpus}")
        self.logger.info(f"   Final throughput: {epoch_avg_throughput:.0f} tokens/s")

    def _save_best_checkpoint(self, optimizer, scheduler):
        """Save best model checkpoint"""
        model_to_save = self.model.module if self.use_multi_gpu else self.model

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss,
            'multi_gpu_training': self.use_multi_gpu,
            'num_gpus': self.num_gpus
        }

        filename = f"bitgen_best_model.pt"
        torch.save(checkpoint, self.output_dir / filename)
        self.logger.info(f"💾 Saved best checkpoint: {filename}")

    def _save_checkpoint(self, optimizer, scheduler):
        """Save model checkpoint"""
        model_to_save = self.model.module if self.use_multi_gpu else self.model

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss,
            'multi_gpu_training': self.use_multi_gpu,
            'num_gpus': self.num_gpus
        }

        filename = f"bitgen_checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, self.output_dir / filename)
        self.logger.info(f"Saved checkpoint: {filename}")

    def _setup_data_loader(self, coco_data_path: str, batch_size: int):
        """Setup data loader optimized for multi-GPU training"""
        # Create simple dataset
        try:
            from src.data_loader import COCODataset
            dataset = COCODataset(
                coco_data_path,
                max_seq_len=self.config.max_seq_len,
                vocab_size=self.config.vocab_size
            )
        except:
            # Fallback to dummy dataset if COCO loader not available
            dataset = DummyDataset(self.config)

        # Optimize DataLoader for multi-GPU training
        num_workers = min(4 * self.num_gpus, 8) if self.use_multi_gpu else 2

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True,
            persistent_workers=num_workers > 0
        )

class DummyDataset:
    """Dummy dataset for testing when COCO is not available"""

    def __init__(self, config):
        self.config = config
        self.length = 1000

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate dummy data
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.max_seq_len,))
        labels = torch.randint(0, self.config.vocab_size, (self.config.max_seq_len,))

        return {
            'input_ids': input_ids,
            'labels': labels,
            'images': None
        }

def create_basic_trainer(model_size: str = 'tiny', **kwargs):
    """Create basic trainer with default config"""

    # Create appropriate config
    if model_size == 'nano':
        from src.bitgen_model import BitGenConfig
        config = BitGenConfig(
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            memory_size=32,
            vocab_size=4096
        )
    elif model_size == 'tiny':
        from src.bitgen_model import BitGenConfig
        config = BitGenConfig()
    elif model_size == 'small':
        from src.bitgen_model import BitGenConfig
        config = BitGenConfig(
            embed_dim=256,
            num_layers=6,
            num_heads=8,
            memory_size=128,
            ffn_dim=512
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    return BasicBitGenTrainer(config=config, model_size=model_size, **kwargs)
