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

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

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
        """Basic training loop"""

        self.logger.info(f"Starting BitGen {self.config.__class__.__name__} training")

        # Setup optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

        # Setup data loader
        train_loader = self._setup_data_loader(coco_data_path, batch_size)

        # Initialize loss function
        from src.adaptive_loss import BitGenLoss
        loss_fn = BitGenLoss(self.config, self.config.vocab_size)

        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = defaultdict(list)

            self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                images = batch.get('images')
                if images is not None:
                    images = images.to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    images=images,
                    return_robot_selection=True
                )

                # Compute loss
                total_loss, loss_dict = loss_fn(outputs, labels, images=images)

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                self.global_step += 1

                # Log metrics
                if step % 50 == 0:
                    self.logger.info(f"Epoch {epoch+1}, Step {step}: Loss={total_loss.item():.4f}")

                # Collect metrics
                epoch_metrics['loss'].append(total_loss.item())
                epoch_metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # End of epoch
            scheduler.step()
            epoch_avg_loss = np.mean(epoch_metrics['loss'])

            # Log to WandB if available
            if self.wandb_integration:
                self.wandb_integration.log_training_metrics({
                    'epoch/loss': epoch_avg_loss,
                    'epoch/learning_rate': epoch_metrics['learning_rate'][-1]
                }, epoch=epoch)

            # Push to HuggingFace if available
            if self.hf_integration:
                try:
                    self.hf_integration.push_model_checkpoint(
                        model=self.model,
                        config=self.config,
                        epoch=epoch + 1,
                        metrics={'loss': epoch_avg_loss}
                    )
                    self.logger.info(f"✅ Model pushed to HuggingFace for epoch {epoch+1}")
                except Exception as e:
                    self.logger.error(f"❌ HuggingFace push failed: {e}")

            # Save checkpoint
            self._save_checkpoint(optimizer, scheduler)

            self.logger.info(f"Epoch {epoch+1} completed. Average loss: {epoch_avg_loss:.4f}")

        # Finish WandB run
        if self.wandb_integration:
            self.wandb_integration.finish_run()

        self.logger.info("Training completed!")

    def _setup_data_loader(self, coco_data_path: str, batch_size: int):
        """Setup basic data loader"""
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

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

    def _save_checkpoint(self, optimizer, scheduler):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss
        }

        filename = f"bitgen_checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, self.output_dir / filename)
        self.logger.info(f"Saved checkpoint: {filename}")

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
