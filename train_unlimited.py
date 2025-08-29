"""
BitMar Unlimited Training Script
Removed BabyLM limitations - supports unlimited multimodal datasets
Enhanced with Facebook's DINOv3-Large for SUPERIOR image understanding
No token constraints or dataset size limitations
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

# Hugging Face Hub integration
try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from transformers import AutoTokenizer, AutoConfig
    HF_HUB_AVAILABLE = True
    print("✅ Hugging Face Hub integration available")
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠️  Hugging Face Hub not available - install with: pip install huggingface_hub")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_unlimited.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import components
from src.dataset import create_data_module
from src.model import create_bitmar_model, count_parameters
from src.wandb_logger import BitMarWandbLogger
from src.attention_visualizer import AttentionHeadAnalyzer
from src.memory_visualization_integration import setup_memory_visualization

# Try to import FLOPS tracker
try:
    from src.flops_tracker import FLOPsTracker, FLOPsEstimator
    FLOPS_TRACKER_AVAILABLE = True
    logger.info("✅ FLOPS tracker available")
except ImportError:
    FLOPS_TRACKER_AVAILABLE = False
    logger.warning("⚠️  FLOPS tracker not available")

# Try to import unlimited dataset handler
try:
    from src.unlimited_dataset import create_unlimited_data_module
    UNLIMITED_DATASET_AVAILABLE = True
    logger.info("✅ Unlimited dataset handler available")
except ImportError:
    UNLIMITED_DATASET_AVAILABLE = False
    logger.warning("⚠️  Unlimited dataset handler not available - using standard dataset")

# Try to import optional components
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

try:
    from src.adaptive_training_controller import AdaptiveTrainingController, compute_cross_modal_similarity
    ADAPTIVE_TRAINING_AVAILABLE = True
except ImportError:
    ADAPTIVE_TRAINING_AVAILABLE = False

# Try to import attention sinks integration
try:
    from src.attention_sinks_integration import (
        AttentionSinksConfig,
        apply_attention_sinks_to_bitmar_model,
        update_model_kwargs_for_generation_with_sinks
    )
    ATTENTION_SINKS_AVAILABLE = True
    logger.info("✅ Attention Sinks integration available")
except ImportError:
    ATTENTION_SINKS_AVAILABLE = False
    logger.warning("⚠️  Attention Sinks integration not available")


class UnlimitedTrainer:
    """Unlimited trainer for multimodal datasets without BabyLM constraints"""

    def __init__(self, config_path: str, device: Optional[str] = None):
        """Initialize unlimited trainer"""
        # Load configuration with validation
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            # Validate required config sections (removed token_constraints)
            required_sections = ['model', 'data', 'training', 'output']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required config section: {section}")

            logger.info(f"Configuration loaded successfully from {config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")

        # Set device with enhanced GPU detection
        logger.info(f"🔍 GPU Detection:")
        logger.info(f"  • CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  • CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  • GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"  • GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

        if device:
            try:
                self.device = torch.device(device)
                if device.startswith('cuda'):
                    if not torch.cuda.is_available():
                        logger.error(f"❌ CUDA not available but {device} requested!")
                        raise RuntimeError(f"CUDA not available for {device}")
                    elif device != "cuda:0" and not torch.cuda.device_count() > int(device.split(':')[1]):
                        logger.warning(f"Device {device} not available, using cuda:0")
                        self.device = torch.device("cuda:0")
                    else:
                        # Test GPU
                        test_tensor = torch.tensor([1.0], device=self.device)
                        logger.info(f"✅ Successfully initialized {device}")
                        logger.info(f"   GPU memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**2:.1f} MB")
                logger.info(f"Using device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to set device {device}: {e}")
                raise RuntimeError(f"Device setup failed: {e}")
        else:
            # Auto-select best available device
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                test_tensor = torch.tensor([1.0], device=self.device)
                logger.info(f"✅ Auto-selected GPU: {self.device}")
                logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                self.device = torch.device("cpu")
                logger.warning(f"⚠️  No GPU available, using CPU (training will be slower)")

        # Training state (removed token tracking)
        self.global_step = 0
        self.current_epoch = 0
        self.best_similarity = 0.0
        self.best_validation_loss = float('inf')
        self.early_stopping_counter = 0

        # Setup directories
        self.setup_directories()

        # Setup logging
        self.setup_wandb()

        # Initialize carbon tracking if available
        self.setup_carbon_tracking()

        # Setup Hugging Face Hub integration
        self.setup_huggingface_hub()

        logger.info(f"🚀 Unlimited multimodal trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Enhanced with Facebook DINOv3-Large for SUPERIOR image understanding")

    def setup_directories(self):
        """Create output directories"""
        for dir_name in ['checkpoint_dir', 'log_dir', 'attention_dir', 'memory_dir', 'results_dir', 'model_exports_dir']:
            dir_path = Path(self.config['output'][dir_name])
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dir_name, dir_path)

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb_config = self.config.get('wandb', {})

        if wandb_config.get('project'):
            try:
                # Enhanced run name for unlimited training
                run_name = f"bitmar-unlimited-{wandb.util.generate_id()[:8]}"

                self.wandb_logger = BitMarWandbLogger(
                    project_name=wandb_config['project'],
                    config=self.config,
                    entity=wandb_config.get('entity'),
                    run_name=run_name
                )
                self.use_wandb = True
                logger.info("✅ Weights & Biases initialized for unlimited training")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
                self.wandb_logger = None
        else:
            self.use_wandb = False
            self.wandb_logger = None

    def setup_carbon_tracking(self):
        """Setup carbon emissions tracking"""
        if not CODECARBON_AVAILABLE:
            self.carbon_tracker = None
            return

        try:
            carbon_logs_dir = Path("./carbon_logs_unlimited")
            carbon_logs_dir.mkdir(exist_ok=True)

            self.carbon_tracker = EmissionsTracker(
                project_name="BitMar-Unlimited-Training",
                experiment_id=f"bitmar-unlimited-{self.device.type}",
                output_dir=str(carbon_logs_dir),
                output_file="emissions_unlimited.csv",
                log_level="INFO",
                save_to_file=True,
                tracking_mode="machine"
            )
            logger.info("🌱 Carbon emissions tracking enabled for unlimited training")
        except Exception as e:
            logger.warning(f"Failed to setup carbon tracking: {e}")
            self.carbon_tracker = None

    def setup_huggingface_hub(self):
        """Setup Hugging Face Hub integration"""
        if not HF_HUB_AVAILABLE:
            self.hf_hub_enabled = False
            logger.warning("⚠️  Hugging Face Hub not available - model uploads disabled")
            return

        hf_config = self.config.get('huggingface_hub', {})

        if not hf_config.get('enabled', False):
            self.hf_hub_enabled = False
            logger.info("📤 Hugging Face Hub uploads disabled in config")
            return

        self.hf_repo_id = hf_config.get('repo_id')
        if not self.hf_repo_id:
            self.hf_hub_enabled = False
            logger.warning("⚠️  No Hugging Face repo_id specified - model uploads disabled")
            return

        # Get authentication token with multiple fallback options
        self.hf_token = None

        if hf_config.get('token'):
            self.hf_token = hf_config.get('token')
        elif os.getenv('HF_TOKEN'):
            self.hf_token = os.getenv('HF_TOKEN')
        else:
            try:
                from huggingface_hub import HfFolder
                stored_token = HfFolder.get_token()
                if stored_token:
                    self.hf_token = stored_token
            except Exception:
                pass

        if not self.hf_token:
            try:
                test_api = HfApi()
                user_info = test_api.whoami()
                if user_info:
                    self.hf_token = "cached_credentials"
            except Exception:
                pass

        if not self.hf_token:
            self.hf_hub_enabled = False
            logger.warning("⚠️  No Hugging Face token found - model uploads disabled")
            return

        try:
            if self.hf_token == "cached_credentials":
                self.hf_api = HfApi()
            else:
                self.hf_api = HfApi(token=self.hf_token)

            user_info = self.hf_api.whoami()
            logger.info(f"✅ Authenticated with Hugging Face as: {user_info['name']}")

            # Check/create repository
            try:
                repo_info = self.hf_api.repo_info(self.hf_repo_id, token=self.hf_token)
                logger.info(f"✅ Repository found: {self.hf_repo_id}")
            except Exception:
                logger.info(f"📤 Creating new repository: {self.hf_repo_id}")
                create_repo(
                    repo_id=self.hf_repo_id,
                    token=self.hf_token,
                    private=hf_config.get('private', True),
                    exist_ok=True
                )
                logger.info(f"✅ Repository created: {self.hf_repo_id}")

            self.hf_hub_enabled = True
            self.hf_upload_after_epoch = hf_config.get('upload_after_epoch', True)
            self.hf_upload_final_model = hf_config.get('upload_final_model', True)
            self.hf_commit_message_template = hf_config.get('commit_message_template',
                "BitMar Unlimited - Epoch {epoch} - Enhanced training")
            self.hf_create_model_card = hf_config.get('create_model_card', True)
            self.hf_model_card_template = hf_config.get('model_card_template', "")

            logger.info("🤗 Hugging Face Hub integration initialized for unlimited training")

        except Exception as e:
            logger.error(f"❌ Failed to setup Hugging Face Hub: {e}")
            self.hf_hub_enabled = False

    def create_model_card(self, epoch: int, final: bool = False) -> str:
        """Create model card content"""
        if not self.hf_model_card_template:
            return ""

        try:
            card_content = self.hf_model_card_template.format(
                epoch=epoch + 1,
                best_similarity=self.best_similarity,
                repo_id=self.hf_repo_id,
                text_encoder_layers=self.config['model']['text_encoder_layers'],
                text_encoder_dim=self.config['model']['text_encoder_dim'],
                vision_latent_size=self.config['model']['vision_latent_size'],
                memory_size=self.config['model']['memory_size'],
                episode_dim=self.config['model']['episode_dim']
            )

            if final:
                card_content += f"\n\n## Training Status\n- **Status**: Completed\n"
            else:
                card_content += f"\n\n## Training Status\n- **Status**: In Progress (Epoch {epoch + 1})\n"

            card_content += f"- **Best Cross-modal Similarity**: {self.best_similarity:.4f}\n"
            card_content += f"- **Vision Model**: Facebook DINOv3-Large\n"
            card_content += f"- **Training Type**: Unlimited multimodal datasets\n"

            return card_content
        except Exception as e:
            logger.warning(f"Failed to create model card: {e}")
            return ""

    def setup_model_and_data(self):
        """Setup model and unlimited multimodal data"""
        logger.info("Setting up model and unlimited multimodal data...")
        logger.info("🔥 Enhanced with Facebook DINOv3-Large for SUPERIOR image understanding")

        # Use unlimited dataset if available
        if UNLIMITED_DATASET_AVAILABLE and self.config.get('data', {}).get('dataset_type') == 'multimodal_unlimited':
            logger.info("🎯 Using unlimited multimodal dataset handler")
            self.data_module = create_unlimited_data_module(self.config['data'])
        else:
            logger.info("📊 Using standard multimodal dataset")
            self.data_module = create_data_module(self.config['data'])

        self.data_module.setup()

        # Create enhanced model
        logger.info("Creating enhanced BitMar model with DINOv3-Large...")
        logger.info(f"Model config dimensions:")
        logger.info(f"  • text_encoder_dim: {self.config['model']['text_encoder_dim']}")
        logger.info(f"  • vision_encoder_dim: {self.config['model']['vision_encoder_dim']} (DINOv3-Large)")
        logger.info(f"  • vision_latent_size: {self.config['model']['vision_latent_size']}")
        logger.info(f"  • memory_size: {self.config['model']['memory_size']} slots")

        self.model = create_bitmar_model(self.config['model'])

        # Apply attention sinks if enabled
        if ATTENTION_SINKS_AVAILABLE and self.config.get('attention_sinks', {}).get('enabled', False):
            try:
                logger.info("🔄 Applying attention sinks for unlimited sequences...")

                attention_sinks_config = AttentionSinksConfig(
                    enable_attention_sinks=True,
                    attention_sink_size=self.config['attention_sinks'].get('attention_sink_size', 8),
                    attention_sink_window_size=self.config['attention_sinks'].get('attention_sink_window_size', 2048),
                    inject_to_text_encoder=self.config['attention_sinks'].get('inject_to_text_encoder', True),
                    inject_to_text_decoder=self.config['attention_sinks'].get('inject_to_text_decoder', True),
                    position_shift_enabled=self.config['attention_sinks'].get('position_shift_enabled', True)
                )

                self.model = apply_attention_sinks_to_bitmar_model(self.model, attention_sinks_config)
                logger.info("✅ Attention Sinks applied for unlimited sequence handling")

            except Exception as e:
                logger.error(f"❌ Failed to apply attention sinks: {e}")
                logger.warning("⚠️  Continuing training without attention sinks")

        self.model.to(self.device)

        # Verify model setup
        logger.info(f"🎮 Device Verification:")
        logger.info(f"  • Model device: {next(self.model.parameters()).device}")
        logger.info(f"  • Expected device: {self.device}")

        if self.device.type == 'cuda':
            logger.info(f"  • GPU memory before training: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
            logger.info(f"  • GPU memory reserved: {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB")

        # Log model info
        param_count = count_parameters(self.model)
        logger.info(f"Enhanced model created with {param_count['total_parameters']:,} total parameters")
        logger.info(f"Trainable parameters: {param_count['trainable_parameters']:,}")

        # Setup optimizer for unlimited training
        self.setup_optimizer()

        # Initialize attention analyzer
        self.attention_analyzer = AttentionHeadAnalyzer(
            model=self.model,
            tokenizer=self.model.tokenizer,
            save_dir=str(self.attention_dir),
            wandb_logger=self.wandb_logger,
            track_top_k=self.config.get('attention_analysis', {}).get('track_top_k', 8)
        )

        # Setup adaptive training if enabled
        self.setup_adaptive_training()

        # Setup memory visualization
        try:
            self.memory_viz = setup_memory_visualization(self.config, self.model)
            logger.info("✅ Memory visualization integration initialized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize memory visualization: {e}")
            self.memory_viz = None

        # Setup FLOPS tracking
        self.setup_flops_tracking()

    def setup_optimizer(self):
        """Setup optimizer and scheduler for unlimited training"""
        train_loader = self.data_module.train_dataloader()
        steps_per_epoch = len(train_loader)

        # Calculate total steps for scheduler (no token constraints)
        max_epochs = self.config['training']['max_epochs']
        total_steps = steps_per_epoch * max_epochs

        logger.info(f"Training planning (unlimited):")
        logger.info(f"  • Steps per epoch: {steps_per_epoch}")
        logger.info(f"  • Maximum epochs: {max_epochs}")
        logger.info(f"  • Total steps: {total_steps}")

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Create enhanced scheduler
        scheduler_config = self.config['training'].get('scheduler_config', {})

        T_0 = int(scheduler_config.get('T_0', 2000))
        T_mult = scheduler_config.get('T_mult', 2)

        if isinstance(T_mult, float):
            T_mult = max(1, int(T_mult))
        elif not isinstance(T_mult, int) or T_mult < 1:
            T_mult = 2

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=self.config['training']['learning_rate'] * scheduler_config.get('eta_min_ratio', 0.05)
        )

        logger.info(f"Enhanced scheduler configured: T_0={T_0}, T_mult={T_mult}")
        logger.info(f"✅ Optimizer configured for unlimited training")

    def setup_adaptive_training(self):
        """Setup adaptive training controller"""
        if not ADAPTIVE_TRAINING_AVAILABLE or not self.config['model'].get('enable_adaptive_training', False):
            self.adaptive_controller = None
            return

        adaptive_config = self.config.get('adaptive_training', {})
        adaptive_logs_dir = Path("./logs/adaptive_training_unlimited")
        adaptive_logs_dir.mkdir(parents=True, exist_ok=True)

        self.adaptive_controller = AdaptiveTrainingController(
            similarity_window_size=adaptive_config.get('similarity_window_size', 300),
            drop_threshold=adaptive_config.get('drop_threshold', 0.10),
            min_steps_between_interventions=adaptive_config.get('min_steps_between_interventions', 600),
            freeze_duration_steps=adaptive_config.get('freeze_duration_steps', 1200),
            loss_rebalance_factor=adaptive_config.get('loss_rebalance_factor', 2.5),
            similarity_smoothing_alpha=adaptive_config.get('similarity_smoothing_alpha', 0.12),
            save_dir=str(adaptive_logs_dir)
        )

        logger.info("🤖 Enhanced adaptive training controller enabled")

    def setup_flops_tracking(self):
        """Setup FLOPS tracking system"""
        if not FLOPS_TRACKER_AVAILABLE:
            self.flops_tracker = None
            return

        try:
            flops_config = self.config.get('flops_tracking', {})
            log_frequency = flops_config.get('log_frequency', 75)

            flops_logs_dir = Path("./flops_logs_unlimited")
            flops_logs_dir.mkdir(parents=True, exist_ok=True)

            self.flops_tracker = FLOPsTracker(
                model=self.model,
                log_frequency=log_frequency,
                save_dir=str(flops_logs_dir)
            )

            self.flops_tracker.log_model_complexity()
            logger.info("🔢 Enhanced FLOPS tracker initialized for unlimited training")

        except Exception as e:
            logger.warning(f"Failed to setup FLOPS tracking: {e}")
            self.flops_tracker = None

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'best_similarity': self.best_similarity,
            'best_validation_loss': self.best_validation_loss,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"💎 New best checkpoint saved: {best_path}")

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch without token constraints"""
        self.model.train()
        train_loader = self.data_module.train_dataloader()

        epoch_losses = []
        epoch_metrics = {
            'train_loss': 0.0,
            'cross_modal_similarity': 0.0,
            'num_samples': 0
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} | Step: {self.global_step}")

        for batch_idx, batch in enumerate(progress_bar):
            # Start FLOPS tracking
            if self.flops_tracker:
                self.flops_tracker.start_step()

            try:
                # Move batch to device
                processed_batch = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        processed_batch[k] = v.to(self.device)
                    else:
                        processed_batch[k] = v

                batch = processed_batch

                # Ensure required keys exist
                if 'vision_index' not in batch:
                    batch['vision_index'] = torch.arange(batch['input_ids'].size(0), device=self.device)

                if 'has_vision' not in batch:
                    batch['has_vision'] = torch.ones(batch['input_ids'].size(0), dtype=torch.bool, device=self.device)

                # Handle vision features for DINOv2-large (1024-dim)
                if 'vision_features' in batch:
                    vf = batch['vision_features']
                    if len(vf.shape) == 3 and vf.shape[1] == 1:
                        batch['vision_features'] = vf.squeeze(1)
                    elif vf.size(-1) != self.config['model']['vision_encoder_dim']:
                        # Pad or truncate to match DINOv2-large dimensions
                        target_dim = self.config['model']['vision_encoder_dim']
                        if vf.size(-1) < target_dim:
                            pad_size = target_dim - vf.size(-1)
                            vf = torch.cat([vf, torch.zeros(*vf.shape[:-1], pad_size, device=vf.device)], dim=-1)
                        elif vf.size(-1) > target_dim:
                            vf = vf[..., :target_dim]
                        batch['vision_features'] = vf

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels'],
                    step=self.global_step,
                    has_vision=batch.get('has_vision'),
                    adaptive_controller=self.adaptive_controller
                )

                loss = outputs['loss']

                # Memory visualization logging
                if self.memory_viz is not None:
                    try:
                        self.memory_viz.log_training_step(
                            batch=batch,
                            epoch=epoch,
                            step=self.global_step,
                            model_outputs=outputs
                        )
                    except Exception as e:
                        logger.warning(f"Memory visualization logging failed: {e}")

                # Check for valid loss
                if not torch.isfinite(loss):
                    logger.warning(f"Invalid loss at step {self.global_step}: {loss.item()}")
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_val']
                    )

                self.optimizer.step()
                self.scheduler.step()

                # End FLOPS tracking
                if self.flops_tracker:
                    try:
                        flops_metrics = self.flops_tracker.end_step(
                            batch_size=batch['input_ids'].size(0),
                            sequence_length=batch['input_ids'].size(1)
                        )
                        if self.flops_tracker.should_log():
                            self.flops_tracker.log_flops(
                                metrics=flops_metrics,
                                logger_func=logger.info,
                                wandb_logger=wandb if self.use_wandb else None,
                                step=self.global_step
                            )
                    except Exception as e:
                        logger.warning(f"FLOPS tracking failed: {e}")

                # Update metrics
                epoch_losses.append(loss.item())
                epoch_metrics['num_samples'] += batch['input_ids'].size(0)

                # Compute cross-modal similarity
                if outputs.get('text_features') is not None and outputs.get('vision_latent') is not None:
                    try:
                        similarity = self._compute_cross_modal_similarity(
                            outputs['text_features'], outputs['vision_latent']
                        )
                        epoch_metrics['cross_modal_similarity'] += similarity

                        if similarity > self.best_similarity:
                            self.best_similarity = similarity
                    except Exception as e:
                        logger.warning(f"Cross-modal similarity computation failed: {e}")

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'step': self.global_step,
                    'sim': f"{self.best_similarity:.3f}"
                })

                # Enhanced wandb logging
                if self.use_wandb and self.global_step % self.config['wandb'].get('log_every_n_steps', 75) == 0:
                    if self.wandb_logger:
                        try:
                            self.wandb_logger.log_consolidated_metrics(
                                outputs=outputs,
                                epoch=epoch,
                                step=self.global_step,
                                lr=self.optimizer.param_groups[0]['lr'],
                                model=self.model,
                                memory_module=getattr(self.model, 'memory', None),
                                log_quantization=True
                            )
                        except Exception as e:
                            logger.warning(f"Failed to log comprehensive metrics: {e}")

                self.global_step += 1

                # Save checkpoint periodically
                if self.global_step % 5000 == 0:
                    self.save_checkpoint(epoch)

            except Exception as e:
                logger.error(f"Training step failed at step {self.global_step}: {e}")
                if hasattr(self, 'optimizer'):
                    self.optimizer.zero_grad()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue

        # Calculate epoch metrics
        if epoch_losses:
            epoch_metrics['train_loss'] = np.mean(epoch_losses)
            epoch_metrics['cross_modal_similarity'] = epoch_metrics['cross_modal_similarity'] / len(epoch_losses)

        logger.info(f"Epoch {epoch} completed:")
        logger.info(f"  • Loss: {epoch_metrics['train_loss']:.4f}")
        logger.info(f"  • Cross-modal similarity: {epoch_metrics['cross_modal_similarity']:.4f}")
        logger.info(f"  • Samples processed: {epoch_metrics['num_samples']:,}")
        logger.info(f"  • Best similarity so far: {self.best_similarity:.4f}")

        return epoch_metrics

    def validate(self) -> float:
        """Run validation"""
        if not hasattr(self.data_module, 'val_dataloader'):
            return float('inf')

        self.model.eval()
        val_loader = self.data_module.val_dataloader()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels']
                )

                val_losses.append(outputs['loss'].item())

        val_loss = np.mean(val_losses)
        logger.info(f"Validation loss: {val_loss:.4f}")
        return val_loss

    def _compute_cross_modal_similarity(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> float:
        """Compute cross-modal similarity"""
        try:
            if text_features.dim() == 3:
                text_pooled = text_features.mean(dim=1)
            else:
                text_pooled = text_features

            if text_pooled.size(-1) != vision_features.size(-1):
                min_dim = min(text_pooled.size(-1), vision_features.size(-1))
                text_pooled = text_pooled[:, :min_dim]
                vision_features = vision_features[:, :min_dim]

            cos_sim = torch.cosine_similarity(text_pooled, vision_features, dim=1)
            return cos_sim.mean().item()
        except Exception as e:
            logger.warning(f"Cross-modal similarity computation failed: {e}")
            return 0.0

    def train(self):
        """Main unlimited training loop"""
        logger.info("🚀 Starting unlimited multimodal training with Facebook DINOv3-Large...")

        # Start carbon tracking
        if self.carbon_tracker:
            self.carbon_tracker.start()

        # Setup model and data
        self.setup_model_and_data()

        try:
            for epoch in range(self.config['training']['max_epochs']):
                logger.info(f"Starting epoch {epoch + 1}/{self.config['training']['max_epochs']}")

                self.current_epoch = epoch
                epoch_metrics = self.train_epoch(epoch)

                # Validation
                val_loss = self.validate()
                is_best = val_loss < self.best_validation_loss
                if is_best:
                    self.best_validation_loss = val_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1

                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)

                # Upload to Hugging Face Hub
                if self.hf_hub_enabled and self.hf_upload_after_epoch:
                    self.upload_checkpoint_to_hf(epoch)

                # Log epoch summary
                if self.use_wandb:
                    try:
                        wandb.log({
                            'epoch/train_loss': epoch_metrics['train_loss'],
                            'epoch/validation_loss': val_loss,
                            'epoch/cross_modal_similarity': epoch_metrics['cross_modal_similarity'],
                            'epoch/best_similarity': self.best_similarity,
                            'epoch/number': epoch
                        }, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log epoch summary: {e}")

                # Early stopping
                patience = self.config['training'].get('early_stopping_patience', 5)
                if self.early_stopping_counter >= patience:
                    logger.info(f"Early stopping after {patience} epochs without improvement")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Stop carbon tracking
            if self.carbon_tracker:
                emissions = self.carbon_tracker.stop()
                logger.info(f"🌱 Carbon emissions: {emissions:.6f} kg CO2")

            # Final checkpoint
            self.save_checkpoint(self.current_epoch, is_best=True)

            # Upload final model
            if self.hf_hub_enabled and self.hf_upload_final_model:
                self.upload_checkpoint_to_hf(self.current_epoch, final=True)

            # Final summaries
            logger.info("🎯 Final Training Summary:")
            logger.info(f"  • Best cross-modal similarity: {self.best_similarity:.4f}")
            logger.info(f"  • Best validation loss: {self.best_validation_loss:.4f}")
            logger.info(f"  • Total epochs: {self.current_epoch + 1}")
            logger.info(f"  • Enhanced with Facebook DINOv3-Large")

            if self.use_wandb:
                try:
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Failed to finish wandb: {e}")

    def upload_checkpoint_to_hf(self, epoch: int, final: bool = False):
        """Upload checkpoint to Hugging Face Hub"""
        if not self.hf_hub_enabled:
            return

        try:
            logger.info(f"📤 Uploading {'final ' if final else ''}model to HF Hub...")

            checkpoint_path = self.checkpoint_dir / ('best_checkpoint.pt' if final else f'checkpoint_epoch_{epoch}.pt')
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return

            # Prepare for upload (similar to previous implementation but simplified)
            hf_model_dir = self.checkpoint_dir / "hf_model_temp"
            hf_model_dir.mkdir(exist_ok=True)

            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Save model
            model_path = hf_model_dir / "pytorch_model.bin"
            torch.save(checkpoint['model_state_dict'], model_path)

            # Create config
            model_config = {
                "architectures": ["BitMarModel"],
                "model_type": "bitmar",
                "vocab_size": self.config['model']['vocab_size'],
                "text_encoder_dim": self.config['model']['text_encoder_dim'],
                "text_encoder_layers": self.config['model']['text_encoder_layers'],
                "vision_encoder_name": self.config['model'].get('vision_encoder_name', 'facebook/dinov3-large'),
                "vision_encoder_dim": self.config['model']['vision_encoder_dim'],
                "memory_size": self.config['model']['memory_size'],
                "training_type": "unlimited_multimodal"
            }

            import json
            with open(hf_model_dir / "config.json", 'w') as f:
                json.dump(model_config, f, indent=2)

            # Create model card
            if self.hf_create_model_card:
                model_card = self.create_model_card(epoch, final)
                if model_card:
                    with open(hf_model_dir / "README.md", 'w') as f:
                        f.write(model_card)

            # Upload
            commit_message = self.hf_commit_message_template.format(epoch=epoch + 1)
            if final:
                commit_message = f"Final unlimited model - {commit_message}"

            self.hf_api.upload_folder(
                folder_path=str(hf_model_dir),
                repo_id=self.hf_repo_id,
                token=self.hf_token,
                commit_message=commit_message
            )

            logger.info(f"✅ Model uploaded: https://huggingface.co/{self.hf_repo_id}")

            # Cleanup
            import shutil
            shutil.rmtree(hf_model_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"❌ Failed to upload to HF Hub: {e}")


def main():
    """Main function for unlimited training"""
    parser = argparse.ArgumentParser(description="Train BitMar with unlimited multimodal datasets")

    parser.add_argument("--config", type=str, default="configs/bitmar_unlimited.yaml",
                       help="Path to unlimited configuration file")
    parser.add_argument("--device", type=str, help="Device to use (cuda:0, cpu)")

    args = parser.parse_args()

    try:
        # Initialize unlimited trainer
        trainer = UnlimitedTrainer(args.config, device=args.device)

        logger.info("🚀 BitMar Unlimited Training")
        logger.info("=" * 50)
        logger.info("✅ Removed all BabyLM limitations")
        logger.info("🔥 Enhanced with Facebook DINOv3-Large")
        logger.info("📈 No token constraints or dataset size limits")
        logger.info("🎯 Focus on high-quality multimodal learning")

        # Start unlimited training
        trainer.train()

    except Exception as e:
        logger.error(f"Unlimited training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
