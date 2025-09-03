"""
Enhanced GRPO Training Script for BitGen Robot Reasoning
Integrates GRPO policy optimization with existing BitGen multimodal training
"""

import os
import sys
import yaml
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import wandb

# TRL imports for GRPO
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
from datasets import Dataset as HFDataset

# BitGen imports
from src.bitgen_model import BitGenModel
from src.multimodal_fusion import MultiModalFusion
from src.vision_encoder import VisionEncoder
from src.grpo_robot_reasoning import GRPORobotReasoningIntegration, PolicyOptimizedRobotHead
from src.robot_reasoning import RobotReasoningProcessor, create_robot_reasoning_datasets

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grpo_robot_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IntegratedGRPODataset(Dataset):
    """Enhanced dataset that combines multimodal and robot reasoning for GRPO training"""

    def __init__(
        self,
        robot_data: List[Dict],
        multimodal_data: List[Dict] = None,
        tokenizer=None,
        max_length: int = 512,
        vision_processor=None,
        config: Dict = None
    ):
        self.robot_data = robot_data
        self.multimodal_data = multimodal_data or []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vision_processor = vision_processor
        self.config = config or {}

        # Combine datasets with proper weighting
        self.combined_data = self._combine_datasets()

        logger.info(f"✅ IntegratedGRPODataset initialized:")
        logger.info(f"   • Robot reasoning samples: {len(robot_data):,}")
        logger.info(f"   • Multimodal samples: {len(self.multimodal_data):,}")
        logger.info(f"   • Combined samples: {len(self.combined_data):,}")

    def _combine_datasets(self) -> List[Dict]:
        """Combine multimodal and robot datasets with proper sampling weights"""

        # Weight robot reasoning higher since it's our focus
        robot_weight = self.config.get('robot_reasoning_weight', 0.7)
        multimodal_weight = 1.0 - robot_weight

        combined = []

        # Add robot reasoning data (weighted)
        robot_samples = int(len(self.robot_data) *
                            robot_weight / (robot_weight + multimodal_weight))
        for i in range(min(robot_samples, len(self.robot_data))):
            item = self.robot_data[i].copy()
            item['data_type'] = 'robot_reasoning'
            combined.append(item)

        # Add multimodal data (weighted)
        if self.multimodal_data:
            multimodal_samples = int(len(
                self.multimodal_data) * multimodal_weight / (robot_weight + multimodal_weight))
            for i in range(min(multimodal_samples, len(self.multimodal_data))):
                item = self.multimodal_data[i].copy()
                item['data_type'] = 'multimodal'
                combined.append(item)

        # Shuffle for better training dynamics
        np.random.shuffle(combined)
        return combined

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.combined_data[idx]

        # Common tokenization
        prompt = item.get('prompt', item.get('instruction', ''))
        response = item.get('response', item.get('answer', ''))

        # Tokenize inputs
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors="pt",
            padding=False
        )

        response_tokens = self.tokenizer(
            response,
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors="pt",
            padding=False
        )

        # Prepare output
        result = {
            'input_ids': prompt_tokens['input_ids'].squeeze(0),
            'attention_mask': prompt_tokens['attention_mask'].squeeze(0),
            'labels': response_tokens['input_ids'].squeeze(0),
            'data_type': item['data_type'],
            'prompt': prompt,
            'response': response
        }

        # Add vision features if available
        if 'image_path' in item and self.vision_processor:
            try:
                vision_features = self.vision_processor(item['image_path'])
                result['vision_features'] = vision_features
            except Exception as e:
                logger.warning(f"Failed to process vision features: {e}")
                result['vision_features'] = None
        else:
            result['vision_features'] = None

        return result


class BitGenGRPOTrainer:
    """Enhanced GRPO trainer for BitGen with multimodal and robot reasoning integration"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize accelerator
        self.accelerator = Accelerator()

        # Setup logging
        self.setup_logging()

        # Initialize model components
        self.initialize_model()

        # Initialize datasets
        self.initialize_datasets()

        # Initialize GRPO trainer
        self.initialize_grpo_trainer()

        logger.info("✅ BitGenGRPOTrainer initialized successfully")

    def setup_logging(self):
        """Setup logging and wandb"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        logging.getLogger().setLevel(log_level)

        # Initialize wandb if configured
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'bitgen-grpo'),
                name=f"grpo-robot-reasoning-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )

    def initialize_model(self):
        """Initialize BitGen model with GRPO robot reasoning"""

        # Load base model
        model_name = self.config['model']['name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add special tokens for robot reasoning
        special_tokens = ["<reasoning>",
                          "</reasoning>", "<answer>", "</answer>"]
        self.tokenizer.add_tokens(special_tokens)

        # Initialize BitGen model
        self.model = BitGenModel(self.config['model'])

        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Initialize vision encoder if specified
        if self.config.get('vision', {}).get('enabled', False):
            self.vision_encoder = VisionEncoder(self.config['vision'])
        else:
            self.vision_encoder = None

        # Initialize GRPO robot reasoning integration
        robot_data_dir = self.config['robot_reasoning']['data_dir']
        self.robot_reasoning = GRPORobotReasoningIntegration(
            self.model, robot_data_dir, self.config['robot_reasoning']
        )

        logger.info(f"✅ Model initialized: {model_name}")
        logger.info(
            f"   • Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(
            f"   • Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def initialize_datasets(self):
        """Initialize training and validation datasets"""

        # Load robot reasoning datasets
        robot_datasets = create_robot_reasoning_datasets(
            self.config['robot_reasoning']['data_dir']
        )

        # Load multimodal datasets (if available)
        multimodal_data = []
        if 'multimodal_data_dir' in self.config.get('data', {}):
            multimodal_data = self._load_multimodal_data(
                self.config['data']['multimodal_data_dir']
            )

        # Create integrated datasets
        self.train_dataset = IntegratedGRPODataset(
            robot_data=robot_datasets['train'],
            multimodal_data=multimodal_data,
            tokenizer=self.tokenizer,
            max_length=self.config['training']['max_length'],
            vision_processor=self.vision_encoder,
            config=self.config
        )

        self.eval_dataset = IntegratedGRPODataset(
            robot_data=robot_datasets['validation'],
            multimodal_data=[],  # Only robot reasoning for eval
            tokenizer=self.tokenizer,
            max_length=self.config['training']['max_length'],
            vision_processor=self.vision_encoder,
            config=self.config
        )

        logger.info(f"✅ Datasets initialized:")
        logger.info(f"   • Training samples: {len(self.train_dataset):,}")
        logger.info(f"   • Validation samples: {len(self.eval_dataset):,}")

    def _load_multimodal_data(self, data_dir: str) -> List[Dict]:
        """Load multimodal training data"""
        # Placeholder for multimodal data loading
        # This would load your existing multimodal datasets
        return []

    def initialize_grpo_trainer(self):
        """Initialize GRPO trainer with robot reasoning rewards"""

        # GRPO configuration
        grpo_config = GRPOConfig(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['eval_batch_size'],
            learning_rate=self.config['training']['learning_rate'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            eval_steps=self.config['training']['eval_steps'],
            save_steps=self.config['training']['save_steps'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            remove_unused_columns=False,
            # GRPO specific parameters
            num_generations=self.config['grpo']['num_generations'],
            grpo_alpha=self.config['grpo']['alpha'],
            temperature=self.config['grpo']['temperature'],
            reward_model_tokenizer=self.tokenizer,
        )

        # Convert datasets to HuggingFace format
        def collate_fn(examples):
            batch = {}

            # Tokenize prompts
            prompts = [ex['prompt'] for ex in examples]
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                padding=True,
                max_length=self.config['training']['max_length'],
                return_tensors="pt"
            )

            batch['input_ids'] = tokenized['input_ids']
            batch['attention_mask'] = tokenized['attention_mask']
            batch['prompts'] = prompts
            batch['responses'] = [ex['response'] for ex in examples]
            batch['data_types'] = [ex['data_type'] for ex in examples]

            return batch

        # Create data loaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2
        )

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config['training']['eval_batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )

        # Initialize GRPO trainer
        self.grpo_trainer = GRPOTrainer(
            config=grpo_config,
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            reward_function=self.robot_reasoning.compute_grpo_rewards,
        )

        logger.info("✅ GRPO trainer initialized")

    def train(self):
        """Run GRPO training with robot reasoning focus"""

        logger.info("🚀 Starting GRPO robot reasoning training...")

        # Prepare everything with accelerator
        self.model, self.grpo_trainer, self.train_dataloader, self.eval_dataloader = \
            self.accelerator.prepare(
                self.model, self.grpo_trainer, self.train_dataloader, self.eval_dataloader
            )

        num_epochs = self.config['training']['num_epochs']

        for epoch in range(num_epochs):
            logger.info(f"\n📊 Epoch {epoch + 1}/{num_epochs}")

            # Training phase
            epoch_metrics = self._train_epoch(epoch)

            # Evaluation phase
            if (epoch + 1) % self.config['training'].get('eval_frequency', 1) == 0:
                eval_metrics = self._evaluate_epoch(epoch)
                epoch_metrics.update(eval_metrics)

            # Logging
            self._log_metrics(epoch, epoch_metrics)

            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_frequency', 5) == 0:
                self._save_checkpoint(epoch, epoch_metrics)

        logger.info("✅ GRPO training completed!")

        # Final evaluation and testing
        self._final_evaluation()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training loop for one epoch"""

        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        policy_losses = []
        value_losses = []

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {epoch + 1}",
            disable=not self.accelerator.is_local_main_process
        )

        for batch_idx, batch in enumerate(progress_bar):

            # Run GRPO step
            try:
                outputs = self.grpo_trainer.training_step(self.model, batch)

                loss = outputs.get('loss', 0.0)
                policy_loss = outputs.get('policy_loss', 0.0)
                value_loss = outputs.get('value_loss', 0.0)

                total_loss += loss
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

                # Calculate rewards for robot reasoning batches
                if any(dt == 'robot_reasoning' for dt in batch['data_types']):
                    robot_indices = [i for i, dt in enumerate(
                        batch['data_types']) if dt == 'robot_reasoning']

                    if robot_indices:
                        robot_prompts = [batch['prompts'][i]
                                         for i in robot_indices]
                        robot_responses = [batch['responses'][i]
                                           for i in robot_indices]

                        # Compute rewards
                        rewards = self.robot_reasoning.compute_grpo_rewards(
                            robot_prompts,
                            [[{"content": resp}] for resp in robot_responses],
                            robot_responses  # Using responses as pseudo ground truth for now
                        )

                        avg_reward = np.mean(rewards.get('total', [0.0]))
                        total_reward += avg_reward

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'policy_loss': f'{policy_loss:.4f}',
                    'value_loss': f'{value_loss:.4f}',
                    'avg_reward': f'{total_reward / (batch_idx + 1):.4f}'
                })

            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                continue

        # Calculate epoch metrics
        num_batches = len(self.train_dataloader)
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'train_value_loss': np.mean(value_losses) if value_losses else 0.0,
            'train_avg_reward': total_reward / num_batches,
            'epoch': epoch + 1
        }

        return metrics

    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluation loop for one epoch"""

        self.model.eval()
        total_eval_loss = 0.0
        total_eval_reward = 0.0

        with torch.no_grad():
            progress_bar = tqdm(
                self.eval_dataloader,
                desc=f"Evaluation Epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process
            )

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Evaluation step
                    outputs = self.grpo_trainer.evaluation_step(
                        self.model, batch)

                    eval_loss = outputs.get('eval_loss', 0.0)
                    total_eval_loss += eval_loss

                    # Calculate robot reasoning rewards
                    robot_indices = [i for i, dt in enumerate(
                        batch['data_types']) if dt == 'robot_reasoning']

                    if robot_indices:
                        robot_prompts = [batch['prompts'][i]
                                         for i in robot_indices]
                        robot_responses = [batch['responses'][i]
                                           for i in robot_indices]

                        rewards = self.robot_reasoning.compute_grpo_rewards(
                            robot_prompts,
                            [[{"content": resp}] for resp in robot_responses],
                            robot_responses
                        )

                        avg_reward = np.mean(rewards.get('total', [0.0]))
                        total_eval_reward += avg_reward

                    progress_bar.set_postfix({
                        'eval_loss': f'{eval_loss:.4f}',
                        'avg_reward': f'{total_eval_reward / (batch_idx + 1):.4f}'
                    })

                except Exception as e:
                    logger.error(f"Error in eval step {batch_idx}: {e}")
                    continue

        # Calculate evaluation metrics
        num_batches = len(self.eval_dataloader)
        eval_metrics = {
            'eval_loss': total_eval_loss / num_batches,
            'eval_avg_reward': total_eval_reward / num_batches
        }

        return eval_metrics

    def _log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics"""

        logger.info(f"\n📊 Epoch {epoch + 1} Results:")
        logger.info(f"   • Training Loss: {metrics.get('train_loss', 0):.4f}")
        logger.info(
            f"   • Policy Loss: {metrics.get('train_policy_loss', 0):.4f}")
        logger.info(
            f"   • Value Loss: {metrics.get('train_value_loss', 0):.4f}")
        logger.info(
            f"   • Average Reward: {metrics.get('train_avg_reward', 0):.4f}")

        if 'eval_loss' in metrics:
            logger.info(f"   • Eval Loss: {metrics['eval_loss']:.4f}")
            logger.info(
                f"   • Eval Average Reward: {metrics['eval_avg_reward']:.4f}")

        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log(metrics, step=epoch)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""

        output_dir = Path(self.config['training']['output_dir'])
        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        torch.save({
            'epoch': epoch + 1,
            'metrics': metrics,
            'config': self.config,
            'robot_reasoning_state': self.robot_reasoning.robot_selection_head.state_dict()
        }, checkpoint_dir / "training_state.pt")

        logger.info(f"💾 Checkpoint saved: {checkpoint_dir}")

    def _final_evaluation(self):
        """Comprehensive final evaluation with robot reasoning testing"""

        logger.info("\n🧪 Running final evaluation...")

        # Test robot reasoning capabilities
        test_scenarios = [
            "Explore underwater cave system for marine research",
            "Inspect and repair damaged power lines at height",
            "Navigate through dense forest to collect environmental samples",
            "Assist in hospital patient care and medication delivery",
            "Perform search and rescue in earthquake debris"
        ]

        for i, scenario in enumerate(test_scenarios):
            logger.info(f"\n🤖 Test Scenario {i + 1}: {scenario}")

            try:
                result = self.robot_reasoning.generate_robot_reasoning_with_policy(
                    task=scenario,
                    temperature=0.1,  # Low temperature for evaluation
                    top_n=None  # Let model decide optimal N
                )

                logger.info(
                    f"   • Selected Robots: {', '.join(result['selected_robots'])}")
                logger.info(
                    f"   • Estimated Complexity: {result['estimated_complexity']}")
                logger.info(
                    f"   • Reasoning Quality: {result['reasoning_quality']:.3f}")
                logger.info(f"   • Selection Probabilities:")

                for robot, prob in result['selection_probabilities'].items():
                    logger.info(f"     - {robot}: {prob:.3f}")

            except Exception as e:
                logger.error(f"Error testing scenario {i + 1}: {e}")

        logger.info("✅ Final evaluation completed!")


def load_config(config_path: str) -> Dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="BitGen GRPO Robot Reasoning Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize trainer
    trainer = BitGenGRPOTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"📂 Resuming from checkpoint: {args.resume}")
        # Add checkpoint loading logic here

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
