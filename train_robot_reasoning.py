"""
Robot Reasoning Training Script for BitGen
Implements deepseek-r1 style structured reasoning for robot selection
Combines multimodal training with robot reasoning capabilities
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
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robot_reasoning_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import components
from src.model import create_bitmar_model, count_parameters
from src.robot_reasoning_dataset import create_robot_reasoning_data_module, create_robot_reasoning_trainer_integration
from src.robot_reasoning import ReasoningFormatValidator


class RobotReasoningTrainer:
    """Specialized trainer for robot reasoning capabilities"""

    def __init__(self, config_path: str, device: Optional[str] = None):
        """Initialize robot reasoning trainer"""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"🤖 Robot Reasoning Trainer initialized on {self.device}")

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_reasoning_accuracy = 0.0

        # Setup directories
        self.setup_directories()

        # Setup logging
        self.setup_wandb()

        # Integrate robot reasoning trainer mixin
        self.reasoning_trainer_mixin = create_robot_reasoning_trainer_integration()

    def setup_directories(self):
        """Create output directories"""
        output_config = self.config['output']
        for dir_name in ['checkpoint_dir', 'log_dir', 'reasoning_dir']:
            dir_path = Path(output_config[dir_name])
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dir_name, dir_path)

    def setup_wandb(self):
        """Setup Weights & Biases logging for robot reasoning"""
        wandb_config = self.config.get('wandb', {})

        if wandb_config.get('project'):
            try:
                wandb.init(
                    project=wandb_config['project'],
                    entity=wandb_config.get('entity'),
                    config=self.config,
                    name=f"robot-reasoning-{wandb.util.generate_id()[:8]}",
                    tags=['robot-reasoning', 'deepseek-r1-style', 'bitgen']
                )
                self.use_wandb = True
                logger.info("✅ Weights & Biases initialized for robot reasoning")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False

    def setup_model_and_data(self):
        """Setup model and robot reasoning data"""
        logger.info("🤖 Setting up robot reasoning model and data...")

        # Create robot reasoning data module
        self.data_module = create_robot_reasoning_data_module(self.config['data'])
        self.data_module.setup()

        # Create model with robot reasoning enabled
        model_config = self.config['model'].copy()
        model_config['enable_robot_reasoning'] = True  # Ensure robot reasoning is enabled
        model_config['robot_data_dir'] = self.config['data'].get('robot_data_dir', "D:/BabyLM/robot_selection_data/data")

        self.model = create_bitmar_model(model_config)
        self.model.to(self.device)

        # Verify robot reasoning integration
        if self.model.robot_reasoning_integration is None:
            logger.error("❌ Robot reasoning integration failed!")
            raise RuntimeError("Robot reasoning not properly integrated")

        logger.info("✅ Robot reasoning model and data setup completed")
        logger.info(f"   • Model has robot reasoning: {self.model.robot_reasoning_integration is not None}")

        # Log model info
        param_count = count_parameters(self.model)
        logger.info(f"Model parameters: {param_count['total_parameters']:,}")

        # Setup optimizer
        self.setup_optimizer()

    def setup_optimizer(self):
        """Setup optimizer for robot reasoning training"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )

        # Scheduler with longer periods for reasoning learning
        scheduler_config = self.config['training']['scheduler_config']
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=scheduler_config['T_0'],
            T_mult=scheduler_config['T_mult'],
            eta_min=self.config['training']['learning_rate'] * scheduler_config['eta_min_ratio']
        )

        logger.info("✅ Optimizer configured for robot reasoning training")

    def evaluate_reasoning_quality(self, generated_text: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate reasoning quality using deepseek-r1 style metrics"""
        metrics = {}

        # Format validation
        metrics['format_valid'] = float(ReasoningFormatValidator.validate_format(generated_text))

        # Extract reasoning and answer
        reasoning = ReasoningFormatValidator.extract_reasoning(generated_text)
        answer = ReasoningFormatValidator.extract_answer(generated_text)

        # Reasoning quality (similar to deepseek-r1's reward functions)
        reasoning_score = 0.0
        if reasoning:
            # Check for key reasoning elements
            if any(word in reasoning.lower() for word in ['task', 'analyze', 'analysis']):
                reasoning_score += 0.25
            if any(word in reasoning.lower() for word in ['environment', 'terrain', 'condition']):
                reasoning_score += 0.25
            if any(word in reasoning.lower() for word in ['capability', 'able', 'suited', 'optimal']):
                reasoning_score += 0.25
            if any(word in reasoning.lower() for word in ['limitation', 'constraint', 'restriction']):
                reasoning_score += 0.25

        metrics['reasoning_quality'] = reasoning_score

        # Answer correctness
        if answer and ground_truth:
            # Extract robot names for comparison
            if 'Selected robot(s):' in answer:
                predicted = answer.split('Selected robot(s):')[1].strip()
            else:
                predicted = answer.strip()

            # Normalize and compare
            pred_set = set(r.strip() for r in predicted.split(','))
            truth_set = set(r.strip() for r in ground_truth.split(','))

            if pred_set == truth_set:
                metrics['answer_correctness'] = 1.0
            elif pred_set.intersection(truth_set):
                overlap = len(pred_set.intersection(truth_set))
                total = len(pred_set.union(truth_set))
                metrics['answer_correctness'] = overlap / total
            else:
                metrics['answer_correctness'] = 0.0
        else:
            metrics['answer_correctness'] = 0.0

        return metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with robot reasoning focus"""
        self.model.train()
        train_loader = self.data_module.train_dataloader()

        epoch_metrics = {
            'train_loss': 0.0,
            'robot_reasoning_accuracy': 0.0,
            'reasoning_format_accuracy': 0.0,
            'reasoning_quality_score': 0.0,
            'decoder_loss': 0.0,
            'cross_modal_loss': 0.0,
            'robot_reasoning_loss': 0.0
        }

        total_batches = 0
        reasoning_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Robot Reasoning Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(self.device)

                # Set robot labels for robot reasoning loss computation
                is_robot_batch = batch.get('is_robot_reasoning', torch.tensor([False]))
                if isinstance(is_robot_batch, torch.Tensor):
                    is_robot_batch = is_robot_batch.any().item()
                elif isinstance(is_robot_batch, (list, tuple)):
                    is_robot_batch = any(is_robot_batch)

                if is_robot_batch:
                    # Set robot labels for loss computation (following deepseek-r1's approach)
                    self.model._current_robot_labels = batch.get('robot_labels', [])
                    reasoning_batches += 1
                else:
                    self.model._current_robot_labels = None

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_features=batch['vision_features'],
                    labels=batch['labels'],
                    step=self.global_step
                )

                loss = outputs['loss']

                # Compute robot reasoning metrics if this batch contains reasoning data
                if is_robot_batch:
                    # Use the reasoning trainer mixin to compute metrics
                    reasoning_metrics = self.reasoning_trainer_mixin.compute_robot_reasoning_metrics(outputs, batch)

                    if reasoning_metrics:
                        for key, value in reasoning_metrics.items():
                            if key in epoch_metrics:
                                epoch_metrics[key] += value

                    # Log reasoning examples periodically
                    if self.global_step % 500 == 0:
                        self.reasoning_trainer_mixin.log_robot_reasoning_examples(batch, outputs, self.global_step)

                    # Compute deepseek-r1 style rewards for logging
                    if self.model.robot_reasoning_integration:
                        reward_functions = self.model.robot_reasoning_integration.get_reward_functions()
                        reward_metrics = self.reasoning_trainer_mixin.compute_tiny_r1_style_rewards(
                            batch, outputs, reward_functions
                        )

                        # Log reward metrics
                        if self.use_wandb and reward_metrics:
                            wandb_rewards = {f'rewards/{k}': v for k, v in reward_metrics.items()}
                            wandb.log(wandb_rewards, step=self.global_step)

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

                # Update metrics
                epoch_metrics['train_loss'] += loss.item()

                # Track loss components
                loss_components = outputs.get('loss_components', {})
                if 'decoder_loss' in loss_components:
                    epoch_metrics['decoder_loss'] += loss_components['decoder_loss'].item()
                if 'cross_modal_loss' in loss_components:
                    epoch_metrics['cross_modal_loss'] += loss_components['cross_modal_loss'].item()
                if 'robot_reasoning_loss' in loss_components and loss_components['robot_reasoning_loss'] is not None:
                    epoch_metrics['robot_reasoning_loss'] += loss_components['robot_reasoning_loss'].item()

                total_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'reasoning_batches': f"{reasoning_batches}/{total_batches}",
                    'robot_acc': f"{reasoning_metrics.get('robot_selection_accuracy', 0.0):.3f}" if 'reasoning_metrics' in locals() and reasoning_metrics else "N/A"
                })

                # Log to wandb
                if self.use_wandb and self.global_step % 100 == 0:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/reasoning_batch_ratio': reasoning_batches / max(total_batches, 1),
                        'train/decoder_loss': loss_components.get('decoder_loss', torch.tensor(0.0)).item(),
                        'train/cross_modal_loss': loss_components.get('cross_modal_loss', torch.tensor(0.0)).item(),
                        'step': self.global_step
                    }

                    # Add robot reasoning loss if available
                    if loss_components.get('robot_reasoning_loss') is not None:
                        log_dict['train/robot_reasoning_loss'] = loss_components['robot_reasoning_loss'].item()

                    # Add robot reasoning metrics if available
                    if 'reasoning_metrics' in locals() and reasoning_metrics:
                        for key, value in reasoning_metrics.items():
                            log_dict[f'train/{key}'] = value

                    # Add robot selection probabilities if available
                    robot_outputs = outputs.get('robot_reasoning_outputs')
                    if robot_outputs and 'robot_selections' in robot_outputs:
                        for robot_key, probs in robot_outputs['robot_selections'].items():
                            log_dict[f'robot_reasoning/{robot_key}_avg_prob'] = probs.mean().item()

                    wandb.log(log_dict, step=self.global_step)

                self.global_step += 1

                # Save checkpoint periodically
                if self.global_step % 2000 == 0:
                    self.save_checkpoint()

            except Exception as e:
                logger.error(f"Training step failed at step {self.global_step}: {e}")
                continue

        # Calculate epoch averages
        if total_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= total_batches if key == 'train_loss' else max(reasoning_batches, 1)

        return epoch_metrics

    def test_robot_reasoning(self, num_examples: int = 10):
        """Test robot reasoning capabilities with sample tasks"""
        logger.info(f"🧪 Testing robot reasoning with {num_examples} examples...")

        if self.model.robot_reasoning_integration is None:
            logger.error("❌ No robot reasoning integration found!")
            return

        # Sample test tasks
        test_tasks = [
            "Inspect a high-rise building's exterior for damage",
            "Navigate underwater caves to collect samples",
            "Deliver supplies through a crowded market",
            "Survey agricultural fields from above",
            "Climb stairs in a damaged building"
        ]

        self.model.eval()
        test_results = []

        with torch.no_grad():
            for i, task in enumerate(test_tasks[:num_examples]):
                try:
                    # Generate reasoning for this task
                    result = self.model.robot_reasoning_integration.generate_robot_reasoning(
                        task=task,
                        context=None,
                        vision_features=None
                    )

                    test_results.append({
                        'task': task,
                        'reasoning': result['reasoning'],
                        'selected_robots': result['selected_robots'],
                        'full_response': result['full_response']
                    })

                    logger.info(f"🤖 Test {i+1}: {task}")
                    logger.info(f"   Reasoning: {result['reasoning'][:100]}...")
                    logger.info(f"   Selected: {result['selected_robots']}")

                except Exception as e:
                    logger.error(f"Test {i+1} failed: {e}")

        # Save test results
        test_results_path = self.reasoning_dir / f"test_results_epoch_{self.current_epoch}.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"💾 Test results saved to: {test_results_path}")
        self.model.train()

        return test_results

    def save_checkpoint(self):
        """Save checkpoint with robot reasoning state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_reasoning_accuracy': self.best_reasoning_accuracy,
            'config': self.config
        }

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'robot_reasoning_checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save latest
        latest_path = self.checkpoint_dir / 'latest_robot_reasoning_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        logger.info(f"💾 Robot reasoning checkpoint saved: {checkpoint_path}")

    def train(self):
        """Main training loop for robot reasoning"""
        logger.info("🚀 Starting robot reasoning training...")

        # Setup model and data
        self.setup_model_and_data()

        try:
            for epoch in range(self.config['training']['max_epochs']):
                logger.info(f"Starting robot reasoning epoch {epoch + 1}")
                self.current_epoch = epoch

                # Train epoch
                epoch_metrics = self.train_epoch(epoch)

                # Test reasoning capabilities
                if epoch % 2 == 0:  # Test every 2 epochs
                    test_results = self.test_robot_reasoning(num_examples=5)

                # Save checkpoint
                self.save_checkpoint()

                # Log epoch summary
                if self.use_wandb:
                    wandb.log({
                        'epoch/train_loss': epoch_metrics['train_loss'],
                        'epoch/robot_reasoning_accuracy': epoch_metrics['robot_reasoning_accuracy'],
                        'epoch/reasoning_format_accuracy': epoch_metrics['reasoning_format_accuracy'],
                        'epoch/reasoning_quality_score': epoch_metrics['reasoning_quality_score'],
                        'epoch/number': epoch
                    }, step=self.global_step)

                logger.info(f"Epoch {epoch} completed:")
                logger.info(f"  • Loss: {epoch_metrics['train_loss']:.4f}")
                logger.info(f"  • Reasoning accuracy: {epoch_metrics['robot_reasoning_accuracy']:.4f}")
                logger.info(f"  • Format accuracy: {epoch_metrics['reasoning_format_accuracy']:.4f}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Final test
            logger.info("🧪 Final robot reasoning test...")
            final_results = self.test_robot_reasoning(num_examples=10)

            # Final checkpoint
            self.save_checkpoint()

            if self.use_wandb:
                wandb.finish()

            logger.info("✅ Robot reasoning training completed")


def main():
    """Main function for robot reasoning training"""
    parser = argparse.ArgumentParser(description="Train BitMar with Robot Reasoning")

    parser.add_argument("--config", type=str, default="configs/bitmar_robot_reasoning.yaml",
                       help="Path to robot reasoning configuration file")
    parser.add_argument("--device", type=str, help="Device to use (cuda:0, cpu)")
    parser.add_argument("--test_only", action="store_true",
                       help="Only test robot reasoning, don't train")

    args = parser.parse_args()

    try:
        trainer = RobotReasoningTrainer(args.config, device=args.device)

        if args.test_only:
            logger.info("🧪 Running robot reasoning test only...")
            trainer.setup_model_and_data()
            trainer.test_robot_reasoning(num_examples=20)
        else:
            trainer.train()

    except Exception as e:
        logger.error(f"Robot reasoning training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
