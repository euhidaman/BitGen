#!/usr/bin/env python3
"""
GRPO-based Robot Reasoning Training for BitGen
Implements Group Relative Policy Optimization (GRPO) from deepseek-r1 style training
Enables better robot selection reasoning through policy optimization instead of supervised learning
"""

from src.robot_reasoning_dataset import create_robot_reasoning_data_module
from src.robot_reasoning import RobotReasoningProcessor, ReasoningFormatValidator, RobotSelectionRewardFunctions
import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import json
import re
from tqdm import tqdm
import wandb

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# TRL imports for GRPO
try:
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    logging.error(f"TRL library not found: {e}")
    logging.error("Install with: pip install trl transformers")
    sys.exit(1)

# BitGen imports

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grpo_robot_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RobotReasoningGRPORewardFunctions:
    """Reward functions adapted for GRPO training of robot reasoning"""

    def __init__(self, robot_data_dir: str):
        self.robot_processor = RobotReasoningProcessor(robot_data_dir)
        self.reward_functions = RobotSelectionRewardFunctions(
            self.robot_processor)
        self.available_robots = ['Drone', 'Underwater Robot',
                                 'Humanoid', 'Robot with Wheels', 'Robot with Legs']

    def robot_correctness_reward_func(self, prompts, completions, ground_truth_robots, **kwargs) -> List[float]:
        """GRPO-compatible robot correctness reward function"""
        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for response, truth in zip(responses, ground_truth_robots):
            extracted_robots = ReasoningFormatValidator.extract_answer(
                response)

            # Normalize robot names
            pred_robots = set(r.strip()
                              for r in extracted_robots.split(',') if r.strip())
            true_robots = set(r.strip() for r in truth.split(',') if r.strip())

            # Perfect match gets highest reward (like deepseek-r1)
            if pred_robots == true_robots:
                rewards.append(2.0)
            elif pred_robots.intersection(true_robots):
                # Partial match with proportional reward
                overlap = len(pred_robots.intersection(true_robots))
                total = len(pred_robots.union(true_robots))
                rewards.append(1.0 * (overlap / total))
            else:
                rewards.append(0.0)

        return rewards

    def robot_validity_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward for selecting valid robots"""
        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for response in responses:
            extracted_robots = ReasoningFormatValidator.extract_answer(
                response)

            if not extracted_robots.strip():
                rewards.append(0.0)
                continue

            robot_list = [r.strip()
                          for r in extracted_robots.split(',') if r.strip()]
            valid_count = sum(
                1 for robot in robot_list if robot in self.available_robots)

            if len(robot_list) > 0:
                validity_score = valid_count / len(robot_list)
                # Max 0.5 like deepseek-r1
                rewards.append(0.5 * validity_score)
            else:
                rewards.append(0.0)

        return rewards

    def strict_format_reward_func(self, completions, **kwargs) -> List[float]:
        """Strict XML format validation reward"""
        responses = [completion[0]["content"] for completion in completions]
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\nSelected robot\(s\):.*?\n</answer>\n?$"

        rewards = []
        for response in responses:
            if re.match(pattern, response.strip(), re.DOTALL):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards

    def soft_format_reward_func(self, completions, **kwargs) -> List[float]:
        """Soft XML format validation reward"""
        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for response in responses:
            if ReasoningFormatValidator.validate_format(response):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards

    def xmlcount_reward_func(self, completions, **kwargs) -> List[float]:
        """XML structure counting reward (adapted from tiny-r1)"""
        responses = [completion[0]["content"] for completion in completions]
        return [self._count_xml_structure(response) for response in responses]

    def reasoning_quality_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward for reasoning quality"""
        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for response in responses:
            reasoning = ReasoningFormatValidator.extract_reasoning(response)
            quality_score = 0.0

            # Check for structured reasoning components
            if any(word in reasoning.lower() for word in ['task analysis', 'analysis', 'assess']):
                quality_score += 0.2
            if any(word in reasoning.lower() for word in ['environment', 'terrain', 'condition']):
                quality_score += 0.2
            if any(word in reasoning.lower() for word in ['capability', 'suited', 'optimal']):
                quality_score += 0.2
            if any(word in reasoning.lower() for word in ['rationale', 'reason', 'because']):
                quality_score += 0.2
            if any(word in reasoning.lower() for word in ['conclusion', 'therefore', 'selected']):
                quality_score += 0.2

            rewards.append(quality_score)

        return rewards

    def top_n_selection_reward_func(self, completions, ground_truth_robots, n=3, **kwargs) -> List[float]:
        """Reward for selecting top-N most suitable robots"""
        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for response, truth in zip(responses, ground_truth_robots):
            extracted_robots = ReasoningFormatValidator.extract_answer(
                response)

            # Parse selected robots
            pred_robots = [r.strip()
                           for r in extracted_robots.split(',') if r.strip()]
            true_robots = [r.strip() for r in truth.split(',') if r.strip()]

            # Reward based on top-N selection quality
            reward = 0.0

            # Bonus for correct number of selections
            if len(pred_robots) <= n and len(pred_robots) > 0:
                reward += 0.3

            # Bonus for including the most important robots
            if len(true_robots) > 0:
                overlap_ratio = len(set(pred_robots).intersection(
                    set(true_robots))) / len(set(true_robots))
                reward += 1.0 * overlap_ratio

            # Penalty for selecting too many robots (inefficiency)
            if len(pred_robots) > n:
                reward -= 0.2 * (len(pred_robots) - n)

            # Bonus for selecting diverse robot types (complementary capabilities)
            if len(set(pred_robots)) == len(pred_robots) and len(pred_robots) > 1:
                reward += 0.2

            rewards.append(max(0.0, reward))  # Ensure non-negative

        return rewards

    def _count_xml_structure(self, text: str) -> float:
        """Count XML structure quality (adapted from tiny-r1)"""
        count = 0.0

        # Reasoning tags
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125

        # Answer tags
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            # Penalty for extra content after closing tag
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001

        return max(0.0, count)  # Ensure non-negative


class RobotReasoningGRPODataset:
    """Dataset adapter for GRPO training with robot reasoning data"""

    def __init__(self, robot_data_dir: str, tokenizer, max_length: int = 512):
        self.robot_data_dir = robot_data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load robot reasoning data
        self.robot_processor = RobotReasoningProcessor(robot_data_dir)
        self.examples = self.robot_processor.create_reasoning_examples()

        logger.info(
            f"✅ Loaded {len(self.examples)} robot reasoning examples for GRPO training")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Create GRPO-compatible format
        system_prompt = """You are an expert robot selection assistant. Analyze the given task and select the most suitable robot(s) from the available options.

Respond in the following format:
<reasoning>
Analyze the task requirements, environment constraints, and match them with robot capabilities and limitations. Consider all factors before making your selection.
</reasoning>
<answer>
Selected robot(s): [Robot Name(s)]
</answer>"""

        # Create prompt for GRPO
        user_prompt = f"Available robots: Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs\n\nTask: {example['task']}"

        # Full conversation for GRPO
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        return {
            'prompt': full_prompt,
            'ground_truth_robots': example['ground_truth'],
            'task': example['task'],
            'reasoning': example['reasoning'],
            'answer': example['answer'],
            'type': example['type']
        }


class RobotReasoningGRPOTrainer:
    """GRPO trainer for robot reasoning with BitGen integration"""

    def __init__(self, config_path: str, device: Optional[str] = None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            f"🤖 Robot Reasoning GRPO Trainer initialized on {self.device}")

        # Setup directories
        self.setup_directories()

        # Setup wandb
        self.setup_wandb()

    def setup_directories(self):
        """Create output directories"""
        output_config = self.config['output']
        for dir_name in ['checkpoint_dir', 'log_dir', 'reasoning_dir']:
            dir_path = Path(output_config[dir_name])
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dir_name, dir_path)

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb_config = self.config.get('wandb', {})

        if wandb_config.get('project'):
            try:
                wandb.init(
                    project=wandb_config['project'],
                    entity=wandb_config.get('entity'),
                    config=self.config,
                    name=f"grpo-robot-reasoning-{wandb.util.generate_id()[:8]}",
                    tags=['grpo', 'robot-reasoning', 'deepseek-r1-style',
                          'bitgen', 'policy-optimization']
                )
                self.use_wandb = True
                logger.info("✅ Weights & Biases initialized for GRPO training")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        else:
            self.use_wandb = False

    def setup_model_and_data(self):
        """Setup model and data for GRPO training"""
        logger.info("🔄 Setting up model and data for GRPO training...")

        # Load tokenizer and model
        model_config = self.config['model']
        model_name = model_config.get(
            'base_model', 'microsoft/DialoGPT-medium')

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info(
            f"✅ Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # Create dataset
        robot_data_dir = self.config['data'].get(
            'robot_data_dir', "D:/BabyLM/robot_selection_data/data")
        self.dataset = RobotReasoningGRPODataset(
            robot_data_dir=robot_data_dir,
            tokenizer=self.tokenizer,
            max_length=self.config['training'].get('max_length', 512)
        )

        # Create reward functions
        self.reward_functions_handler = RobotReasoningGRPORewardFunctions(
            robot_data_dir)

        logger.info("✅ Model and data setup completed for GRPO training")

    def create_grpo_trainer(self):
        """Create GRPO trainer with robot-specific reward functions"""

        # GRPO training arguments
        training_config = self.config['training']

        training_args = GRPOConfig(
            learning_rate=training_config.get('learning_rate', 5e-6),
            adam_beta1=training_config.get('adam_beta1', 0.9),
            adam_beta2=training_config.get('adam_beta2', 0.99),
            weight_decay=training_config.get('weight_decay', 0.1),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            lr_scheduler_type=training_config.get(
                'lr_scheduler_type', 'cosine'),
            optim=training_config.get('optimizer', 'adamw_torch'),
            logging_steps=training_config.get('logging_steps', 10),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            per_device_train_batch_size=training_config.get('batch_size', 1),
            gradient_accumulation_steps=training_config.get(
                'gradient_accumulation_steps', 4),
            # Multiple generations for ranking
            num_generations=training_config.get('num_generations', 6),
            max_prompt_length=training_config.get('max_prompt_length', 256),
            max_completion_length=training_config.get(
                'max_completion_length', 300),
            max_steps=training_config.get('max_steps', 1000),
            save_steps=training_config.get('save_steps', 100),
            max_grad_norm=training_config.get('max_grad_norm', 0.1),
            report_to="wandb" if self.use_wandb else None,
            output_dir=str(self.checkpoint_dir),
            evaluation_strategy="steps" if training_config.get(
                'eval_steps') else "no",
            eval_steps=training_config.get('eval_steps', 50),
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="rewards/mean",
            greater_is_better=True,
            remove_unused_columns=False
        )

        # Define reward functions for GRPO
        reward_funcs = [
            self.reward_functions_handler.xmlcount_reward_func,
            self.reward_functions_handler.soft_format_reward_func,
            self.reward_functions_handler.strict_format_reward_func,
            self.reward_functions_handler.robot_validity_reward_func,
            self.reward_functions_handler.robot_correctness_reward_func,
            self.reward_functions_handler.reasoning_quality_reward_func,
            # New: Top-N selection reward
            self.reward_functions_handler.top_n_selection_reward_func
        ]

        # Create GRPO trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=self.dataset
        )

        logger.info(
            "✅ GRPO trainer created with robot reasoning reward functions")
        logger.info(f"   • Reward functions: {len(reward_funcs)}")
        logger.info(f"   • Training examples: {len(self.dataset)}")
        logger.info(
            f"   • Generations per sample: {training_args.num_generations}")

    def train(self):
        """Main GRPO training loop"""
        logger.info("🚀 Starting GRPO training for robot reasoning...")

        # Setup model and data
        self.setup_model_and_data()

        # Create GRPO trainer
        self.create_grpo_trainer()

        try:
            # Start training
            logger.info("🔄 Beginning GRPO policy optimization...")
            self.trainer.train()

            logger.info("✅ GRPO training completed successfully!")

            # Save final model
            final_model_dir = self.checkpoint_dir / "final_grpo_model"
            self.trainer.save_model(str(final_model_dir))
            self.tokenizer.save_pretrained(str(final_model_dir))

            logger.info(f"💾 Final model saved to: {final_model_dir}")

            # Test reasoning capabilities
            self.test_reasoning_capabilities()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            if self.use_wandb:
                wandb.finish()

    def test_reasoning_capabilities(self, num_examples: int = 10):
        """Test robot reasoning capabilities after GRPO training"""
        logger.info(
            f"🧪 Testing robot reasoning capabilities with {num_examples} examples...")

        test_tasks = [
            "Inspect underwater pipelines for corrosion and damage",
            "Deliver medical supplies through a crowded urban area",
            "Survey agricultural fields for crop health assessment",
            "Navigate rough mountain terrain to collect geological samples",
            "Perform precision assembly tasks in a manufacturing facility",
            "Monitor wildlife in a forest environment without disturbance",
            "Clean windows on a 50-story skyscraper",
            "Explore and map underground cave systems",
            "Provide assistance to elderly patients in a hospital",
            "Transport heavy equipment across a desert"
        ]

        test_results = []
        self.model.eval()

        with torch.no_grad():
            for i, task in enumerate(test_tasks[:num_examples]):
                try:
                    # Create test prompt
                    system_prompt = """You are an expert robot selection assistant. Analyze the given task and select the most suitable robot(s) from the available options.

Respond in the following format:
<reasoning>
Analyze the task requirements, environment constraints, and match them with robot capabilities and limitations. Consider all factors before making your selection.
</reasoning>
<answer>
Selected robot(s): [Robot Name(s)]
</answer>"""

                    user_prompt = f"Available robots: Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs\n\nTask: {task}"
                    full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

                    # Tokenize and generate
                    inputs = self.tokenizer(
                        full_prompt, return_tensors="pt", truncation=True, max_length=256)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    # Generate response
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=300,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.eos_token_id
                        )

                    # Decode response
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                    # Extract reasoning and answer
                    reasoning = ReasoningFormatValidator.extract_reasoning(
                        response)
                    selected_robots = ReasoningFormatValidator.extract_answer(
                        response)

                    test_result = {
                        'task': task,
                        'full_response': response,
                        'reasoning': reasoning,
                        'selected_robots': selected_robots,
                        'format_valid': ReasoningFormatValidator.validate_format(response)
                    }

                    test_results.append(test_result)

                    logger.info(f"🤖 Test {i+1}: {task}")
                    logger.info(f"   Selected: {selected_robots}")
                    logger.info(
                        f"   Format valid: {test_result['format_valid']}")
                    logger.info(
                        f"   Reasoning preview: {reasoning[:100]}..." if reasoning else "   No reasoning extracted")

                except Exception as e:
                    logger.error(f"Test {i+1} failed: {e}")

        # Save test results
        test_results_path = self.reasoning_dir / "grpo_test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"💾 Test results saved to: {test_results_path}")

        # Calculate metrics
        valid_format_count = sum(1 for r in test_results if r['format_valid'])
        logger.info(f"📊 Test Results Summary:")
        logger.info(
            f"   • Valid format: {valid_format_count}/{len(test_results)} ({valid_format_count/len(test_results)*100:.1f}%)")
        logger.info(
            f"   • Average reasoning length: {np.mean([len(r['reasoning']) for r in test_results]):.1f} chars")

        return test_results


def main():
    """Main function for GRPO robot reasoning training"""
    parser = argparse.ArgumentParser(
        description="Train BitGen Robot Reasoning with GRPO")

    parser.add_argument("--config", type=str, default="configs/bitmar_robot_reasoning.yaml",
                        help="Path to robot reasoning configuration file")
    parser.add_argument("--device", type=str,
                        help="Device to use (cuda:0, cpu)")
    parser.add_argument("--test_only", action="store_true",
                        help="Only test reasoning capabilities, don't train")

    args = parser.parse_args()

    try:
        trainer = RobotReasoningGRPOTrainer(args.config, device=args.device)

        if args.test_only:
            logger.info("🧪 Running reasoning test only...")
            trainer.setup_model_and_data()
            trainer.test_reasoning_capabilities(num_examples=15)
        else:
            trainer.train()

    except Exception as e:
        logger.error(f"GRPO training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
