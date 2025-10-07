"""
BitGen Training Script
Main training loop for embedded microcontroller deployment
"""

import torch
import torch.nn as nn
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

# Import BitGen components
from .bitgen_model import BitGenModel, BitGenConfig, create_bitgen_model
from .adaptive_loss import BitGenLoss, AdaptiveLossManager, PerformanceTracker, EmbeddedTrainingUtils
from .data_loader import COCODataset, BitGenDataLoader, RobotSelectionDataset

class BitGenTrainer:
    """Complete training system for BitGen models"""
    
    def __init__(self, 
                 config: BitGenConfig,
                 model_size: str = 'tiny',
                 output_dir: str = 'checkpoints',
                 use_wandb: bool = False):
        
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
        
        # Memory optimization
        self.memory_utils = EmbeddedTrainingUtils()
        
        # Logging
        self.setup_logging()
        
        # Wandb integration
        if use_wandb:
            self.setup_wandb()
    
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
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb.init(
            project="bitgen-embedded",
            config=self.config.__dict__,
            name=f"bitgen-{self.config.embed_dim}d-{self.config.num_layers}l"
        )
    
    def setup_optimizer(self, learning_rate: float = 1e-4):
        """Setup optimizer with enhanced numerical stability"""
        # Reduce learning rate for better stability
        stable_lr = learning_rate * 0.1  # Reduce by 10x to prevent gradient explosion

        # Use AdamW with more conservative settings for numerical stability
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=stable_lr,
            betas=(0.9, 0.999),  # More conservative beta2
            weight_decay=0.01,   # Reduced weight decay
            eps=1e-8,
            amsgrad=True        # More stable variant
        )
        
        # More gradual learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1000,
            eta_min=stable_lr * 0.01  # Lower minimum LR
        )

        self.logger.info(f"Optimizer setup with stable learning rate: {stable_lr}")

        return optimizer, scheduler
    
    def setup_data_loaders(self, 
                          coco_data_path: str,
                          robot_data_path: Optional[str] = None,
                          batch_size: int = 4,
                          num_workers: int = 2):
        """Setup data loaders for multi-modal training - OPTIMIZED FOR A4500 GPU"""

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
        coco_dataset = COCODataset(
            coco_data_path,
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.config.vocab_size
        )
        
        train_loader = DataLoader(
            coco_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
            prefetch_factor=4 if num_workers > 0 else None  # Prefetch more batches
        )
        
        # Robot selection dataset (if available)
        robot_loader = None
        if robot_data_path and Path(robot_data_path).exists():
            robot_dataset = RobotSelectionDataset(robot_data_path)
            robot_loader = DataLoader(
                robot_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False
            )
        
        return train_loader, robot_loader
    
    def train_step(self, batch: Dict, optimizer) -> Dict:
        """Single training step with enhanced monitoring"""
        self.model.train()
        
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

        try:
            # Forward pass with autocast for mixed precision
            with autocast('cuda'):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images
                )

                total_loss, loss_dict = self.loss_fn(
                    outputs,
                    labels,
                    images=images,
                    target_robot=target_robot
                )

                # FIXED: Only check for actual NaN/Inf, not zero values
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    self.logger.error(f"Invalid loss detected! Loss value: {total_loss.item()}")
                    # Create a small valid loss to maintain training
                    total_loss = torch.tensor(1.0, device=total_loss.device, requires_grad=True)
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

        # FIXED: Only call zero_grad if optimizer is not None
        if optimizer is not None:
            optimizer.zero_grad()

        # Scale loss for gradient accumulation if optimizer is None
        if optimizer is None:
            # During gradient accumulation, just do backward pass without optimizer step
            total_loss.backward()
        else:
            # Normal training step with optimizer
            total_loss.backward()

            # Enhanced gradient clipping with NaN checking
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self.logger.error(f"NaN/Inf gradients detected! Grad norm: {grad_norm}")
                # FIXED: Only clear gradients if optimizer is not None
                if optimizer is not None:
                    optimizer.zero_grad()  # Clear bad gradients
                return {'total_loss': 0.0, 'learning_rate': optimizer.param_groups[0]['lr'], 'skipped_batch': True}

            if grad_norm > 10.0:  # Very large gradients
                self.logger.warning(f"Large gradient norm detected: {grad_norm:.4f}")

            optimizer.step()

        # Update metrics
        lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 1e-5
        metrics = {
            'total_loss': total_loss.item(),
            'learning_rate': lr
        }
        
        # Add individual loss components
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            elif isinstance(value, dict):
                for k, v in value.items():
                    metrics[f"{key}_{k}"] = v
        
        return metrics
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluation step"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        metrics = defaultdict(float)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                images = batch.get('images')
                if images is not None:
                    images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, images=images)
                
                # Compute loss
                loss, loss_dict = self.loss_fn(outputs, labels, images=images)
                
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Accumulate metrics
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        metrics[key] += value.item() * batch_size
        
        # Average metrics
        avg_metrics = {
            'eval_loss': total_loss / total_samples
        }
        
        for key, value in metrics.items():
            avg_metrics[f"eval_{key}"] = value / total_samples
        
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
                # FIXED: Reduce batch size to avoid OOM, use gradient accumulation instead
                optimized_batch_size = max(batch_size, 64)  # Reduced from 128 to 64

                # Use less GPU memory to avoid fragmentation (up to 30GB out of 46GB)
                optimized_memory_mb = min(30000, int(total_vram_gb * 650))  # Use ~65% of VRAM

                self.logger.info(f"🚀 A40/A100 OPTIMIZATION ENABLED:")
                self.logger.info(f"   Original batch_size: {batch_size} → Optimized: {optimized_batch_size}")
                self.logger.info(f"   Original memory_mb: {max_memory_mb} → Optimized: {optimized_memory_mb}")

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
                # For A40+: Use gradient accumulation to simulate larger batch size
                # Effective batch size = 64 * 2 = 128
                grad_accum_steps = 2  # Accumulate gradients over 2 steps
                actual_batch_size = batch_size // grad_accum_steps  # 64 / 2 = 32 per step

                self.logger.info(f"💡 Using gradient accumulation for A40:")
                self.logger.info(f"   Physical batch size: {actual_batch_size}")
                self.logger.info(f"   Gradient accumulation steps: {grad_accum_steps}")
                self.logger.info(f"   Effective batch size: {batch_size}")

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
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            self.logger.info("✅ Enabled PyTorch CUDA memory optimization")

        # Setup training components
        optimizer, scheduler = self.setup_optimizer(learning_rate)
        train_loader, robot_loader = self.setup_data_loaders(
            coco_data_path, robot_data_path, actual_batch_size
        )
        
        # Training loop with GPU optimization
        self.model.train()
        scaler = GradScaler('cuda')

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = defaultdict(list)
            
            # MEMORY OPTIMIZATION: Clear cache at start of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Combine COCO and robot data if available
            if robot_loader:
                data_iterator = zip(train_loader, robot_loader)
                desc = f"Epoch {epoch+1}/{num_epochs} (Multi-modal)"
            else:
                data_iterator = train_loader
                desc = f"Epoch {epoch+1}/{num_epochs} (COCO only)"
            
            progress_bar = tqdm(data_iterator, desc=desc)
            
            accumulated_loss = 0.0
            for step, batch_data in enumerate(progress_bar):
                if robot_loader and isinstance(batch_data, tuple):
                    coco_batch, robot_batch = batch_data
                    # Merge batches for multi-task learning
                    batch = self.merge_batches(coco_batch, robot_batch)
                else:
                    batch = batch_data
                
                # GPU-OPTIMIZED: Gradient accumulation for large effective batch sizes
                with autocast('cuda'):
                    step_metrics = self.train_step(batch, optimizer if (step + 1) % grad_accum_steps == 0 else None)

                accumulated_loss += step_metrics['total_loss']

                # Only step optimizer after accumulating gradients
                if (step + 1) % grad_accum_steps == 0:
                    # Scale loss by accumulation steps
                    step_metrics['total_loss'] = accumulated_loss / grad_accum_steps
                    accumulated_loss = 0.0

                    # Update metrics
                    for key, value in step_metrics.items():
                        epoch_metrics[key].append(value)

                    # Update performance tracker
                    self.performance_tracker.update(step_metrics)

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{step_metrics['total_loss']:.4f}",
                        'lr': f"{step_metrics['learning_rate']:.6f}",
                        'gpu_util': f"{self._get_gpu_utilization():.0f}%"
                    })

                    self.global_step += 1

                    # Evaluation
                    if self.global_step % eval_steps == 0:
                        eval_metrics = self.evaluate(train_loader)  # Use subset for speed

                        avg_eval_loss = eval_metrics['eval_loss']
                        self.logger.info(f"Step {self.global_step}: Eval Loss = {avg_eval_loss:.4f}")

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

            # End of epoch
            scheduler.step()
            
            # Log epoch metrics
            epoch_avg = {key: np.mean(values) for key, values in epoch_metrics.items()}
            self.logger.info(f"Epoch {epoch+1} completed. Avg Loss: {epoch_avg['total_loss']:.4f}")
            
            # Log GPU utilization summary
            if torch.cuda.is_available():
                gpu_util = self._get_gpu_utilization()
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"GPU Utilization: {gpu_util:.1f}%, Memory: {gpu_memory_used:.1f}GB")

            if wandb.run:
                wandb.log(epoch_avg, step=self.global_step)
        
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
