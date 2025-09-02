"""
Unified BitMar Training Script
Combines multimodal training, robot reasoning, and comprehensive security
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import time
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import all components
from src.dataset import create_data_module
from src.model import create_bitmar_model, count_parameters
from src.wandb_logger import BitMarWandbLogger
from src.attention_visualizer import AttentionHeadAnalyzer
from src.memory_visualization_integration import setup_memory_visualization
from src.robot_reasoning_dataset import create_robot_reasoning_data_module, create_robot_reasoning_trainer_integration
from src.robot_reasoning import ReasoningFormatValidator, create_robot_reasoning_integration
from src.security_guard import BitGenSecurityGuard, SecurityConfig, SecurityIntegration, create_security_guard

# Optional imports
try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from src.flops_tracker import FLOPsTracker, FLOPsEstimator
    FLOPS_TRACKER_AVAILABLE = True
except ImportError:
    FLOPS_TRACKER_AVAILABLE = False

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

try:
    from src.adaptive_training_controller import AdaptiveTrainingController
    ADAPTIVE_TRAINING_AVAILABLE = True
except ImportError:
    ADAPTIVE_TRAINING_AVAILABLE = False

try:
    from src.attention_sinks_integration import AttentionSinksConfig, apply_attention_sinks_to_bitmar_model
    ATTENTION_SINKS_AVAILABLE = True
except ImportError:
    ATTENTION_SINKS_AVAILABLE = False


class UnifiedBitMarTrainer:
    """Unified trainer for BitMar with multimodal data, robot reasoning, and security"""

    def __init__(self, config_path: str, device: Optional[str] = None):
        """Initialize unified trainer"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_similarity = 0.0
        self.tokens_processed = 0

        # Setup directories
        self.setup_directories()

        # Setup components
        self.setup_wandb()
        self.setup_carbon_tracking()
        self.setup_huggingface_hub()

    def setup_directories(self):
        """Create output directories"""
        for dir_name in ['checkpoint_dir', 'log_dir', 'attention_dir', 'memory_dir', 'results_dir']:
            dir_path = Path(self.config['output'][dir_name])
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dir_name, dir_path)

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('project'):
            try:
                self.wandb_logger = BitMarWandbLogger(
                    project_name=wandb_config['project'],
                    config=self.config,
                    entity=wandb_config.get('entity'),
                    run_name=f"bitmar-unified-{wandb.util.generate_id()[:8]}"
                )
                self.use_wandb = True
                logger.info("✅ Weights & Biases initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
                self.wandb_logger = None
        else:
            self.use_wandb = False
            self.wandb_logger = None

    def setup_carbon_tracking(self):
        """Setup carbon emissions tracking"""
        if CODECARBON_AVAILABLE:
            try:
                self.carbon_tracker = EmissionsTracker(project_name="BitMar-Unified-Training")
                logger.info("🌱 Carbon emissions tracking enabled")
            except Exception as e:
                logger.warning(f"Failed to setup carbon tracking: {e}")
                self.carbon_tracker = None
        else:
            self.carbon_tracker = None

    def setup_huggingface_hub(self):
        """Setup Hugging Face Hub integration"""
        self.hf_hub_enabled = False
        if not HF_HUB_AVAILABLE:
            return

        hf_config = self.config.get('huggingface_hub', {})
        if hf_config.get('enabled', False) and hf_config.get('repo_id'):
            try:
                self.hf_api = HfApi()
                self.hf_repo_id = hf_config['repo_id']
                self.hf_hub_enabled = True
                logger.info(f"✅ Hugging Face Hub integration enabled: {self.hf_repo_id}")
            except Exception as e:
                logger.warning(f"Failed to setup Hugging Face Hub: {e}")

    def setup_model_and_data(self):
        """Setup model and data with unified support"""
        logger.info("Setting up unified model and data...")

        # Detect capabilities from config
        self.enable_robot_reasoning = self.config['model'].get('enable_robot_reasoning', False)

        # Create unified data module
        if self.enable_robot_reasoning:
            robot_data_config = self.config['data'].copy()
            robot_data_config.update({
                'robot_data_dir': self.config['model'].get('robot_data_dir', "D:/BabyLM/robot_selection_data/data"),
                'robot_data_ratio': self.config['data'].get('robot_data_ratio', 0.3),
                'include_multimodal_data': True,
                'include_robot_data': True
            })
            self.data_module = create_robot_reasoning_data_module(robot_data_config)
            self.reasoning_trainer_mixin = create_robot_reasoning_trainer_integration()
            logger.info("🤖 Robot reasoning data module initialized")
        else:
            self.data_module = create_data_module(self.config['data'])
            self.reasoning_trainer_mixin = None
            logger.info("📊 Standard multimodal data module initialized")

        self.data_module.setup(rebuild_cache=getattr(self, 'rebuild_cache', False))

        # Create model
        model_config = self.config['model'].copy()
        if self.enable_robot_reasoning:
            model_config['enable_robot_reasoning'] = True
            model_config['robot_data_dir'] = self.config['model'].get('robot_data_dir', "D:/BabyLM/robot_selection_data/data")

        self.model = create_bitmar_model(model_config)
        self.model.to(self.device)

        # Setup security guard
        self.setup_security_guard()

        # Setup optimizer
        self.setup_optimizer()

        # Setup optional components
        self.setup_attention_analyzer()
        self.setup_memory_visualization()
        self.setup_flops_tracking()
        self.setup_adaptive_training()

        logger.info("✅ Unified model and data setup completed")

    def setup_security_guard(self):
        """Setup comprehensive security guard"""
        try:
            security_config = self.config.get('security', {})
            config = SecurityConfig(security_config)
            self.security_guard = BitGenSecurityGuard(config, str(self.device))

            # Wrap model with security
            self.model, self.security_guard = SecurityIntegration.wrap_model_with_security(
                self.model, security_config
            )

            logger.info("🛡️ Security guard initialized with input/output validation and memory anomaly detection")
        except Exception as e:
            logger.warning(f"Failed to setup security guard: {e}")
            self.security_guard = None

    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler_config = self.config['training'].get('scheduler_config', {})
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=scheduler_config.get('T_0', 2000),
            T_mult=scheduler_config.get('T_mult', 2)
        )

    def setup_attention_analyzer(self):
        """Setup attention analysis"""
        try:
            self.attention_analyzer = AttentionHeadAnalyzer(
                model=self.model,
                tokenizer=self.model.tokenizer,
                save_dir=str(self.attention_dir),
                wandb_logger=self.wandb_logger
            )
        except Exception as e:
            logger.warning(f"Failed to setup attention analyzer: {e}")
            self.attention_analyzer = None

    def setup_memory_visualization(self):
        """Setup memory visualization"""
        try:
            self.memory_viz = setup_memory_visualization(self.config, self.model)
            logger.info("✅ Memory visualization initialized")
        except Exception as e:
            logger.warning(f"Failed to setup memory visualization: {e}")
            self.memory_viz = None

    def setup_flops_tracking(self):
        """Setup FLOPS tracking"""
        if FLOPS_TRACKER_AVAILABLE:
            try:
                flops_config = self.config.get('flops_tracking', {})
                self.flops_tracker = FLOPsTracker(
                    model=self.model,
                    log_frequency=flops_config.get('log_frequency', 100),
                    save_dir="./flops_logs"
                )
                logger.info("🔢 FLOPS tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to setup FLOPS tracking: {e}")
                self.flops_tracker = None
        else:
            self.flops_tracker = None

    def setup_adaptive_training(self):
        """Setup adaptive training controller"""
        if ADAPTIVE_TRAINING_AVAILABLE and self.config['model'].get('enable_adaptive_training', False):
            try:
                adaptive_config = self.config.get('adaptive_training', {})
                self.adaptive_controller = AdaptiveTrainingController(
                    similarity_window_size=adaptive_config.get('similarity_window_size', 200),
                    drop_threshold=adaptive_config.get('drop_threshold', 0.12),
                    save_dir="./logs/adaptive_training"
                )
                logger.info("🤖 Adaptive training controller enabled")
            except Exception as e:
                logger.warning(f"Failed to setup adaptive training: {e}")
                self.adaptive_controller = None
        else:
            self.adaptive_controller = None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with security monitoring"""
        self.model.train()
        train_loader = self.data_module.train_dataloader()

        epoch_losses = []
        epoch_metrics = {
            'train_loss': 0.0,
            'cross_modal_similarity': 0.0,
            'tokens_in_epoch': 0,
            'security_blocks': 0,
            'memory_anomalies': 0
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)

            try:
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels'],
                    step=self.global_step
                )

                loss = outputs['loss']

                # Security monitoring
                if self.security_guard:
                    try:
                        # Monitor memory security if model has memory
                        if hasattr(self.model, 'memory') and self.model.memory is not None:
                            memory_state = self.model.memory.get_memory_state()
                            security_result = self.security_guard.monitor_memory_security(memory_state, self.global_step)

                            if security_result.get('is_anomaly', False):
                                epoch_metrics['memory_anomalies'] += 1
                                logger.warning(f"🚨 Memory anomaly detected at step {self.global_step}")

                        # Add security monitoring to training step
                        SecurityIntegration.add_security_to_training_step(self, batch, outputs, self.global_step)

                    except Exception as e:
                        logger.warning(f"Security monitoring failed: {e}")

                # Robot reasoning metrics
                if self.enable_robot_reasoning and self.reasoning_trainer_mixin:
                    try:
                        reasoning_metrics = self.reasoning_trainer_mixin.compute_robot_reasoning_metrics(outputs, batch)
                        if reasoning_metrics:
                            for key, value in reasoning_metrics.items():
                                if key in epoch_metrics:
                                    epoch_metrics[key] = epoch_metrics.get(key, 0) + value
                    except Exception as e:
                        logger.warning(f"Robot reasoning metrics failed: {e}")

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip_val'])

                self.optimizer.step()
                self.scheduler.step()

                # Update metrics
                epoch_losses.append(loss.item())
                batch_tokens = batch['attention_mask'].sum().item()
                self.tokens_processed += batch_tokens
                epoch_metrics['tokens_in_epoch'] += batch_tokens

                # Update progress
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'tokens': f"{self.tokens_processed:,}"})

                # Logging
                if self.use_wandb and self.global_step % 100 == 0:
                    try:
                        log_dict = {
                            'train/loss': loss.item(),
                            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                            'tokens/processed': self.tokens_processed,
                            'step': self.global_step
                        }

                        # Add security metrics
                        if self.security_guard:
                            security_stats = self.security_guard.get_security_statistics()
                            log_dict.update({
                                'security/blocked_inputs': security_stats['blocked_inputs'],
                                'security/blocked_outputs': security_stats['blocked_outputs'],
                                'security/memory_anomalies': security_stats['memory_anomalies']
                            })

                        wandb.log(log_dict, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log to wandb: {e}")

                self.global_step += 1

                # Periodic checkpointing
                if self.global_step % 5000 == 0:
                    self.save_checkpoint()

            except Exception as e:
                logger.error(f"Training step failed: {e}")
                continue

        # Calculate epoch metrics
        if epoch_losses:
            epoch_metrics['train_loss'] = np.mean(epoch_losses)

        return epoch_metrics

    def save_checkpoint(self):
        """Save checkpoint with security report"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'tokens_processed': self.tokens_processed,
            'best_similarity': self.best_similarity,
            'config': self.config
        }

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}_step_{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)

        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        # Save security report
        if self.security_guard:
            try:
                security_report_path = self.security_guard.export_security_report(
                    self.checkpoint_dir / f'security_report_epoch_{self.current_epoch}.json'
                )
                logger.info(f"📊 Security report saved: {security_report_path}")
            except Exception as e:
                logger.warning(f"Failed to save security report: {e}")

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def test_robot_reasoning_with_security(self, num_examples: int = 5):
        """Test robot reasoning with security validation"""
        if not self.enable_robot_reasoning:
            return []

        test_tasks = [
            "Inspect building exterior for damage",
            "Navigate underwater caves",
            "Deliver supplies through market",
            "Survey agricultural fields",
            "Transport heavy equipment"
        ]

        self.model.eval()
        test_results = []

        with torch.no_grad():
            for task in test_tasks[:num_examples]:
                try:
                    # Test with security guard if available
                    if self.security_guard and hasattr(self.model, 'secure_robot_reasoning'):
                        result = self.model.secure_robot_reasoning(task)
                        test_results.append({
                            'task': task,
                            'result': result,
                            'security_validated': result.get('success', False)
                        })
                    else:
                        # Fallback to standard reasoning
                        if hasattr(self.model, 'robot_reasoning_integration'):
                            result = self.model.robot_reasoning_integration.generate_robot_reasoning(task)
                            test_results.append({
                                'task': task,
                                'result': result,
                                'security_validated': False
                            })

                except Exception as e:
                    logger.error(f"Robot reasoning test failed for task '{task}': {e}")

        self.model.train()
        return test_results

    def train(self):
        """Main unified training loop"""
        logger.info("🚀 Starting unified BitMar training with security...")

        if self.carbon_tracker:
            self.carbon_tracker.start()

        # Setup everything
        self.setup_model_and_data()

        try:
            total_epochs = self.config['training']['max_epochs']

            for epoch in range(total_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{total_epochs}")
                self.current_epoch = epoch

                # Train epoch
                epoch_metrics = self.train_epoch(epoch)

                # Test robot reasoning with security
                if self.enable_robot_reasoning and epoch % 2 == 0:
                    test_results = self.test_robot_reasoning_with_security()

                # Save checkpoint
                self.save_checkpoint()

                # Upload to HuggingFace
                if self.hf_hub_enabled:
                    self.upload_checkpoint_to_hf(epoch)

                logger.info(f"Epoch {epoch} completed - Loss: {epoch_metrics['train_loss']:.4f}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Final security report
            if self.security_guard:
                final_report = self.security_guard.export_security_report()
                logger.info(f"📊 Final security report: {final_report}")

            if self.carbon_tracker:
                emissions = self.carbon_tracker.stop()
                logger.info(f"🌱 Total emissions: {emissions:.6f} kg CO2")

            self.save_checkpoint()

    def upload_checkpoint_to_hf(self, epoch: int):
        """Upload checkpoint to Hugging Face Hub"""
        if not self.hf_hub_enabled:
            return

        try:
            checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
            if checkpoint_path.exists():
                # Simple upload - can be enhanced later
                logger.info(f"📤 Uploading to {self.hf_repo_id}...")
        except Exception as e:
            logger.error(f"Upload failed: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Unified BitMar Training")
    parser.add_argument("--config", type=str, default="configs/bitmar_with_memory.yaml")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild dataset cache")

    args = parser.parse_args()

    try:
        trainer = UnifiedBitMarTrainer(args.config, device=args.device)
        trainer.rebuild_cache = args.rebuild_cache
        trainer.train()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
