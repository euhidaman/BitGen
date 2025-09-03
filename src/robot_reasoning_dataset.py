"""
Robot Reasoning Dataset Module for BitGen
Implements deepseek-r1 style structured reasoning for robot selection tasks
Combines multimodal training data with robot selection reasoning tasks
Following deepseek-r1's XML format and reward-based training approach
"""

import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RobotReasoningDataset(Dataset):
    """Dataset that implements deepseek-r1 style reasoning for robot selection"""

    def __init__(
        self,
        robot_data_dir: str,
        tokenizer,
        max_length: int = 512,
        robot_data_ratio: float = 0.3,
        reasoning_format: str = "xml"
    ):
        self.robot_data_dir = Path(robot_data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.robot_data_ratio = robot_data_ratio
        self.reasoning_format = reasoning_format

        # Load robot selection data
        self.load_robot_data()

        # Create reasoning examples following deepseek-r1's format
        self.create_reasoning_examples()

        logger.info(f"✅ Robot reasoning dataset initialized (deepseek-r1 style)")
        logger.info(f"   • Single robot examples: {len(self.single_robot_examples)}")
        logger.info(f"   • Multi robot examples: {len(self.multi_robot_examples)}")
        logger.info(f"   • Total reasoning examples: {len(self.reasoning_examples)}")

    def load_robot_data(self):
        """Load robot selection datasets"""
        # Load single robot selection data
        single_path = self.robot_data_dir / "Single-Robot-Selection" / "single_robot_selection_dataset.json"
        with open(single_path, 'r') as f:
            self.single_robot_data = json.load(f)

        # Load multi robot selection data
        multi_path = self.robot_data_dir / "Multi-Robot-Selection" / "multi_robot_selection_dataset.json"
        with open(multi_path, 'r') as f:
            self.multi_robot_data = json.load(f)

        logger.info(f"📊 Robot data loaded: {len(self.single_robot_data)} single + {len(self.multi_robot_data)} multi")

    def create_reasoning_examples(self):
        """Create reasoning-formatted examples from robot data (following deepseek-r1's XML structure)"""
        self.reasoning_examples = []
        self.single_robot_examples = []
        self.multi_robot_examples = []

        # deepseek-r1 style system prompt for robot reasoning
        system_prompt = """You are an expert robot selection assistant. Analyze the given task and select the most suitable robot(s) from the available options.

Respond in the following format:
<reasoning>
Analyze the task requirements, environment constraints, and match them with robot capabilities and limitations. Consider all factors before making your selection.
</reasoning>
<answer>
Selected robot(s): [Robot Name(s)]
</answer>"""

        # Process single robot examples (following deepseek-r1's conversation format)
        for item in self.single_robot_data:
            reasoning = self._generate_reasoning_for_task(item['input'], item['output'])

            # Create deepseek-r1 style conversation with XML response
            conversation_text = f"System: {system_prompt}\n\nUser: Available robots: Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs\n\nTask: {item['input']}\n\nAssistant: <reasoning>\n{reasoning}\n</reasoning>\n<answer>\nSelected robot(s): {item['output']}\n</answer>"

            example = {
                'type': 'single_robot',
                'conversation_text': conversation_text,
                'system_prompt': system_prompt,
                'user_prompt': f"Available robots: Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs\n\nTask: {item['input']}",
                'reasoning': reasoning,
                'answer': item['output'],
                'full_response': f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\nSelected robot(s): {item['output']}\n</answer>",
                'robot_labels': item['output'],
                'ground_truth': item['output']
            }

            self.single_robot_examples.append(example)
            self.reasoning_examples.append(example)

        # Process multi robot examples
        for item in self.multi_robot_data:
            if 'subtasks' in item:
                # Extract unique robots from subtasks
                selected_robots = list(set([subtask['assigned_robot'] for subtask in item['subtasks']]))
                robot_output = ', '.join(selected_robots)

                reasoning = self._generate_multi_robot_reasoning(item['input'], item['subtasks'])

                conversation_text = f"System: {system_prompt}\n\nUser: Available robots: Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs\n\nTask: {item['input']}\n\nAssistant: <reasoning>\n{reasoning}\n</reasoning>\n<answer>\nSelected robot(s): {robot_output}\n</answer>"

                example = {
                    'type': 'multi_robot',
                    'conversation_text': conversation_text,
                    'system_prompt': system_prompt,
                    'user_prompt': f"Available robots: Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs\n\nTask: {item['input']}",
                    'reasoning': reasoning,
                    'answer': robot_output,
                    'full_response': f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\nSelected robot(s): {robot_output}\n</answer>",
                    'robot_labels': robot_output,
                    'ground_truth': robot_output,
                    'subtasks': item['subtasks']
                }

                self.multi_robot_examples.append(example)
                self.reasoning_examples.append(example)

    def _generate_reasoning_for_task(self, task: str, selected_robot: str) -> str:
        """Generate structured reasoning for robot selection (following deepseek-r1's detailed reasoning style)"""
        task_lower = task.lower()
        reasoning_parts = []

        # Task analysis (like deepseek-r1's step-by-step breakdown)
        reasoning_parts.append(f"Task Analysis: {task}")

        # Environment analysis
        if 'underwater' in task_lower or 'marine' in task_lower or 'ocean' in task_lower or 'pipes' in task_lower or 'seabed' in task_lower:
            reasoning_parts.append("Environment Assessment: Underwater/aquatic environment requires waterproof robot with specialized underwater navigation and marine operation capabilities")
        elif 'aerial' in task_lower or 'air' in task_lower or 'above' in task_lower or 'high-rise' in task_lower or 'exterior' in task_lower or 'from above' in task_lower:
            reasoning_parts.append("Environment Assessment: Aerial/elevated environment requires flight capability, aerial maneuvering, and access to hard-to-reach elevated areas")
        elif 'rough' in task_lower or 'rocky' in task_lower or 'mountain' in task_lower or 'uneven' in task_lower or 'terrain' in task_lower or 'forest' in task_lower:
            reasoning_parts.append("Environment Assessment: Challenging terrain requires stability, balance, and advanced navigation on irregular, uneven surfaces")
        elif 'flat' in task_lower or 'desert' in task_lower or 'warehouse' in task_lower or 'road' in task_lower or 'industrial' in task_lower:
            reasoning_parts.append("Environment Assessment: Flat/structured environment allows for efficient wheeled movement, stable platforms, and high-speed transport")
        elif 'indoor' in task_lower or 'building' in task_lower or 'stairs' in task_lower or 'urban' in task_lower or 'pedestrian' in task_lower or 'crowded' in task_lower:
            reasoning_parts.append("Environment Assessment: Indoor/human environment requires careful navigation, obstacle avoidance, and potential human interaction capabilities")

        # Capability requirements analysis
        capability_requirements = []
        if 'inspect' in task_lower or 'survey' in task_lower or 'monitor' in task_lower or 'check' in task_lower or 'assess' in task_lower:
            capability_requirements.append("inspection and surveillance systems")
        if 'deliver' in task_lower or 'transport' in task_lower or 'carry' in task_lower or 'supplies' in task_lower or 'package' in task_lower:
            capability_requirements.append("payload capacity and reliable transport mechanisms")
        if 'explore' in task_lower or 'navigate' in task_lower or 'traverse' in task_lower or 'map' in task_lower:
            capability_requirements.append("autonomous navigation and exploration capabilities")
        if 'interaction' in task_lower or 'human' in task_lower or 'pedestrian' in task_lower:
            capability_requirements.append("safe human interaction and social navigation")

        if capability_requirements:
            reasoning_parts.append(f"Required Capabilities: {', '.join(capability_requirements)}")

        # Robot-specific evaluation and justification
        robot_justifications = {
            'Drone': "Drone is optimal for aerial operations with advanced surveillance capabilities, lightweight transport, and ability to access hard-to-reach elevated areas safely and efficiently",
            'Underwater Robot': "Underwater Robot is specialized for aquatic environments with waterproof design, underwater navigation systems, deep sea exploration, and marine inspection capabilities",
            'Humanoid': "Humanoid robot excels in human environments with advanced manipulation capabilities, complex task execution, human interaction skills, and versatile tool use",
            'Robot with Wheels': "Robot with Wheels provides fast and efficient movement on flat surfaces with excellent payload capacity, stable platform design, and energy-efficient transport",
            'Robot with Legs': "Robot with Legs offers superior stability and navigation on rough, uneven terrain with good load-carrying ability, balance systems, and versatile mobility"
        }

        # Handle multiple robots (for cases like "Humanoid, Robot with Legs")
        if ',' in selected_robot:
            robots = [r.strip() for r in selected_robot.split(',')]
            reasoning_parts.append("Multiple Robot Selection Analysis:")
            reasoning_parts.append("Task complexity requires multiple specialized robots working in coordination:")
            for robot in robots:
                if robot in robot_justifications:
                    reasoning_parts.append(f"• {robot}: {robot_justifications[robot]}")
            reasoning_parts.append(f"Coordination Strategy: {', '.join(robots)} working together provides complementary capabilities for optimal task completion")
        else:
            reasoning_parts.append("Robot Selection Rationale:")
            if selected_robot in robot_justifications:
                reasoning_parts.append(robot_justifications[selected_robot])

        # Final conclusion (like deepseek-r1's conclusive reasoning)
        reasoning_parts.append(f"Conclusion: {selected_robot} is the optimal choice for this specific task based on environmental requirements and capability matching")

        return "\n".join(reasoning_parts)

    def _generate_multi_robot_reasoning(self, task: str, subtasks: List[Dict]) -> str:
        """Generate reasoning for multi-robot coordination (deepseek-r1 style complex reasoning)"""
        reasoning_parts = []
        reasoning_parts.append(f"Complex Multi-Robot Task Analysis: {task}")
        reasoning_parts.append("Task Decomposition Analysis:")
        reasoning_parts.append("This complex task requires coordination between multiple specialized robots with different capabilities")

        # Group subtasks by execution order
        execution_phases = {}
        for subtask in subtasks:
            order = subtask['execution_order']
            if order not in execution_phases:
                execution_phases[order] = []
            execution_phases[order].append(subtask)

        # Analyze each execution phase (like deepseek-r1's sequential reasoning)
        reasoning_parts.append("\nExecution Phase Analysis:")
        for phase in sorted(execution_phases.keys()):
            phase_tasks = execution_phases[phase]
            reasoning_parts.append(f"Phase {phase}:")

            for task_info in phase_tasks:
                robot = task_info['assigned_robot']
                subtask_desc = task_info['subtask']
                reasoning_parts.append(f"  • Subtask: {subtask_desc}")
                reasoning_parts.append(f"  • Assigned Robot: {robot}")
                reasoning_parts.append(f"  • Justification: {self._justify_robot_for_subtask(subtask_desc, robot)}")

        # Coordination strategy analysis
        if len(execution_phases) > 1:
            reasoning_parts.append("\nCoordination Strategy Analysis:")
            reasoning_parts.append("Sequential execution required with proper task handoffs and communication between specialized robot teams")
            reasoning_parts.append("Each phase builds upon previous phase results, requiring coordinated timing and data sharing")
        else:
            reasoning_parts.append("\nCoordination Strategy Analysis:")
            reasoning_parts.append("Parallel execution enables simultaneous task completion by multiple specialized robots")
            reasoning_parts.append("Robots can work independently in parallel, improving overall efficiency and task completion time")

        # Final selection summary
        selected_robots = list(set([subtask['assigned_robot'] for subtask in subtasks]))
        reasoning_parts.append(f"\nFinal Robot Selection: {', '.join(selected_robots)}")
        reasoning_parts.append("These robots provide complementary capabilities necessary for comprehensive task completion")

        return "\n".join(reasoning_parts)

    def _justify_robot_for_subtask(self, subtask: str, robot: str) -> str:
        """Justify robot selection for specific subtask (detailed like deepseek-r1)"""
        subtask_lower = subtask.lower()

        # Detailed justifications for each robot type
        if robot == 'Drone':
            if any(word in subtask_lower for word in ['aerial', 'overview', 'above', 'facade', 'exterior']):
                return "Optimal for aerial operations with surveillance capabilities and elevated access without ground-based limitations"
            return "Provides aerial perspective and rapid deployment capabilities"

        elif robot == 'Underwater Robot':
            if any(word in subtask_lower for word in ['underwater', 'marine', 'seabed', 'pipes']):
                return "Specialized for underwater operations with waterproof design and aquatic navigation systems"
            return "Designed specifically for aquatic environments and underwater tasks"

        elif robot == 'Humanoid':
            if any(word in subtask_lower for word in ['documentation', 'coordination', 'complex', 'manipulation']):
                return "Excels at complex tasks requiring advanced manipulation, tool use, and coordination capabilities"
            return "Provides human-like dexterity and complex task execution abilities"

        elif robot == 'Robot with Wheels':
            if any(word in subtask_lower for word in ['transport', 'equipment', 'fast', 'efficient']):
                return "Provides fast and efficient movement with high payload capacity on flat surfaces"
            return "Optimal for rapid transport and efficient movement on structured terrain"

        elif robot == 'Robot with Legs':
            if any(word in subtask_lower for word in ['ground', 'foundation', 'terrain', 'stability']):
                return "Offers superior stability and navigation capabilities on varied and challenging terrain"
            return "Provides stability and versatile mobility on uneven surfaces"

        return "Selected based on task-specific capability requirements and environmental suitability"

    def __len__(self):
        return len(self.reasoning_examples)

    def __getitem__(self, idx):
        """Get reasoning example with proper tokenization (following deepseek-r1's data format)"""
        example = self.reasoning_examples[idx]

        # Use the conversation text directly (like deepseek-r1's prompt format)
        formatted_text = example['conversation_text']

        # Tokenize (following deepseek-r1's tokenization approach)
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Create dummy vision features (robot selection is primarily text-based, but BitGen is multimodal)
        vision_features = torch.randn(768)  # Standard vision feature size

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0).bool(),  # Convert to boolean
            'labels': encoding['input_ids'].squeeze(0).clone(),  # For language modeling loss
            'vision_features': vision_features,
            'robot_labels': example['robot_labels'],
            'reasoning_text': example['reasoning'],
            'answer_text': example['answer'],
            'ground_truth': example['ground_truth'],
            'full_response': example['full_response'],
            'task_type': example['type'],
            'has_vision': torch.tensor(False),  # Robot reasoning is primarily text-based
            'vision_index': torch.tensor(idx),
            'index': torch.tensor(idx),
            'is_robot_reasoning': torch.tensor(True)  # Flag for training logic
        }


class HybridReasoningDataset(Dataset):
    """Hybrid dataset combining multimodal data with robot reasoning data (deepseek-r1 style integration)"""

    def __init__(
        self,
        multimodal_dataset,
        robot_reasoning_dataset,
        robot_data_ratio: float = 0.3
    ):
        self.multimodal_dataset = multimodal_dataset
        self.robot_reasoning_dataset = robot_reasoning_dataset
        self.robot_data_ratio = robot_data_ratio

        # Calculate dataset sizes
        self.multimodal_size = len(multimodal_dataset)
        self.robot_size = len(robot_reasoning_dataset)

        # Calculate effective sizes based on ratio (like deepseek-r1's data mixing)
        total_robot_samples = int(self.multimodal_size * robot_data_ratio / (1 - robot_data_ratio))
        self.effective_robot_size = min(total_robot_samples, self.robot_size)

        # Total dataset size
        self.total_size = self.multimodal_size + self.effective_robot_size

        logger.info(f"📊 Hybrid dataset created (deepseek-r1 style):")
        logger.info(f"   • Multimodal samples: {self.multimodal_size}")
        logger.info(f"   • Robot reasoning samples: {self.effective_robot_size}")
        logger.info(f"   • Total samples: {self.total_size}")
        logger.info(f"   • Robot data ratio: {self.robot_data_ratio:.1%}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if idx < self.multimodal_size:
            # Return multimodal sample
            sample = self.multimodal_dataset[idx]
            if not isinstance(sample, dict):
                sample = {'data': sample}
            sample['is_robot_reasoning'] = torch.tensor(False)

            # Ensure consistent keys for robot reasoning integration
            if 'robot_labels' not in sample:
                sample['robot_labels'] = ""
            if 'reasoning_text' not in sample:
                sample['reasoning_text'] = ""
            if 'answer_text' not in sample:
                sample['answer_text'] = ""
            if 'ground_truth' not in sample:
                sample['ground_truth'] = ""
            if 'full_response' not in sample:
                sample['full_response'] = ""
            if 'task_type' not in sample:
                sample['task_type'] = "multimodal"

            return sample
        else:
            # Return robot reasoning sample
            robot_idx = (idx - self.multimodal_size) % self.robot_size
            sample = self.robot_reasoning_dataset[robot_idx]
            sample['is_robot_reasoning'] = torch.tensor(True)
            return sample


def create_robot_reasoning_data_module(config: Dict):
    """Create data module with robot reasoning capabilities (following deepseek-r1's data integration)"""

    class RobotReasoningDataModule:
        def __init__(self, config):
            self.config = config
            self.tokenizer = None
            self.robot_dataset = None
            self.hybrid_dataset = None

        def setup(self, stage=None, rebuild_cache=False):
            """Setup datasets with robot reasoning integration"""
            from transformers import AutoTokenizer

            # Initialize tokenizer (like deepseek-r1's tokenizer setup)
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Create robot reasoning dataset (following deepseek-r1's dataset creation)
            robot_data_dir = self.config.get('robot_data_dir', "D:/BabyLM/robot_selection_data/data")
            self.robot_dataset = RobotReasoningDataset(
                robot_data_dir=robot_data_dir,
                tokenizer=self.tokenizer,
                max_length=self.config.get('max_seq_length', 512),
                robot_data_ratio=self.config.get('robot_data_ratio', 0.3)
            )

            # Create hybrid dataset if multimodal data is included
            if self.config.get('include_multimodal_data', True):
                try:
                    # Try to load existing multimodal dataset
                    from .dataset import create_data_module
                    base_data_module = create_data_module(self.config)
                    base_data_module.setup()

                    multimodal_dataset = None
                    if hasattr(base_data_module, 'train_dataset'):
                        multimodal_dataset = base_data_module.train_dataset
                    elif hasattr(base_data_module, 'dataset'):
                        multimodal_dataset = base_data_module.dataset

                    if multimodal_dataset is not None:
                        self.hybrid_dataset = HybridReasoningDataset(
                            multimodal_dataset=multimodal_dataset,
                            robot_reasoning_dataset=self.robot_dataset,
                            robot_data_ratio=self.config.get('robot_data_ratio', 0.3)
                        )
                        logger.info("✅ Hybrid dataset created with multimodal + robot reasoning data")
                    else:
                        self.hybrid_dataset = self.robot_dataset
                        logger.info("🤖 Using robot reasoning dataset only")

                except Exception as e:
                    logger.warning(f"Failed to load multimodal data: {e}")
                    self.hybrid_dataset = self.robot_dataset
                    logger.info("🤖 Using robot reasoning dataset only")
            else:
                self.hybrid_dataset = self.robot_dataset
                logger.info("🤖 Using robot reasoning dataset only (multimodal disabled)")

        def train_dataloader(self):
            """Create training dataloader with proper collation for robot reasoning"""
            return DataLoader(
                self.hybrid_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=True,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=self.config.get('pin_memory', True),
                persistent_workers=self.config.get('persistent_workers', True) and self.config.get('num_workers', 4) > 0,
                drop_last=True,
                collate_fn=self._robot_reasoning_collate_fn
            )

        def _robot_reasoning_collate_fn(self, batch):
            """Custom collate function for robot reasoning data (following deepseek-r1's batch processing)"""
            # Initialize batch dictionary with all possible keys
            collated_batch = {
                'input_ids': [],
                'attention_mask': [],
                'labels': [],
                'vision_features': [],
                'robot_labels': [],
                'reasoning_text': [],
                'answer_text': [],
                'ground_truth': [],
                'full_response': [],
                'task_type': [],
                'has_vision': [],
                'vision_index': [],
                'index': [],
                'is_robot_reasoning': []
            }

            # Process each sample in batch
            for item in batch:
                for key in collated_batch.keys():
                    if key in item:
                        collated_batch[key].append(item[key])
                    else:
                        # Provide sensible defaults for missing keys
                        if key == 'robot_labels':
                            collated_batch[key].append("")
                        elif key == 'is_robot_reasoning':
                            collated_batch[key].append(torch.tensor(False))
                        elif key == 'task_type':
                            collated_batch[key].append("multimodal")
                        elif key in ['reasoning_text', 'answer_text', 'ground_truth', 'full_response']:
                            collated_batch[key].append("")
                        elif key in ['has_vision', 'vision_index', 'index']:
                            collated_batch[key].append(torch.tensor(0))
                        elif key == 'vision_features':
                            collated_batch[key].append(torch.randn(768))  # Dummy vision features
                        else:
                            collated_batch[key].append(None)

            # Stack tensor data
            tensor_keys = ['input_ids', 'attention_mask', 'labels', 'vision_features', 'has_vision', 'vision_index', 'index', 'is_robot_reasoning']
            for key in tensor_keys:
                if key in collated_batch and all(item is not None for item in collated_batch[key]):
                    try:
                        if all(torch.is_tensor(item) for item in collated_batch[key]):
                            collated_batch[key] = torch.stack(collated_batch[key])
                    except Exception as e:
                        logger.warning(f"Failed to stack {key}: {e}")

            return collated_batch

        def get_dataset_info(self):
            """Get dataset information"""
            return {
                'total_samples': len(self.hybrid_dataset),
                'robot_reasoning_samples': len(self.robot_dataset),
                'single_robot_examples': len(self.robot_dataset.single_robot_examples),
                'multi_robot_examples': len(self.robot_dataset.multi_robot_examples),
                'robot_data_ratio': self.config.get('robot_data_ratio', 0.3)
            }

    return RobotReasoningDataModule(config)


def create_robot_reasoning_trainer_integration():
    """Create trainer integration for robot reasoning (following deepseek-r1's training approach)"""

    class RobotReasoningTrainerMixin:
        """Mixin for integrating robot reasoning into training (deepseek-r1 style)"""

        def compute_robot_reasoning_metrics(self, outputs, batch):
            """Compute robot reasoning specific metrics (following deepseek-r1's reward computation)"""
            metrics = {}

            # Only process robot reasoning batches
            is_robot_batch = batch.get('is_robot_reasoning', torch.tensor([False]))
            if isinstance(is_robot_batch, torch.Tensor):
                is_robot_batch = is_robot_batch.any().item()
            elif isinstance(is_robot_batch, (list, tuple)):
                is_robot_batch = any(is_robot_batch)

            if not is_robot_batch:
                return metrics

            # Extract robot reasoning outputs
            robot_outputs = outputs.get('robot_reasoning_outputs')
            if robot_outputs is None:
                return metrics

            # Compute robot selection accuracy (like deepseek-r1's correctness evaluation)
            robot_labels = batch.get('robot_labels', [])
            full_responses = batch.get('full_response', [])

            if robot_labels and robot_outputs.get('robot_selections'):
                robot_types = ['Drone', 'Underwater Robot', 'Humanoid', 'Robot with Wheels', 'Robot with Legs']

                batch_accuracy = []
                format_accuracy = []
                reasoning_quality_scores = []

                for i, label in enumerate(robot_labels):
                    if isinstance(label, str) and label.strip():
                        # Get predicted robots above threshold
                        predicted_robots = []
                        for j, robot_type in enumerate(robot_types):
                            robot_key = robot_type.replace(' ', '_')
                            if robot_key in robot_outputs['robot_selections']:
                                prob = robot_outputs['robot_selections'][robot_key][i].item()
                                if prob > 0.5:  # Threshold for selection
                                    predicted_robots.append(robot_type)

                        # Compare with ground truth (like deepseek-r1's correctness reward)
                        true_robots = set(r.strip() for r in label.split(',') if r.strip())
                        pred_robots = set(predicted_robots)

                        if true_robots == pred_robots:
                            batch_accuracy.append(1.0)
                        elif true_robots.intersection(pred_robots):
                            # Partial credit (like deepseek-r1's proportional rewards)
                            overlap = len(true_robots.intersection(pred_robots))
                            total = len(true_robots.union(pred_robots))
                            batch_accuracy.append(overlap / total)
                        else:
                            batch_accuracy.append(0.0)

                        # Format validation (like deepseek-r1's format rewards)
                        if i < len(full_responses) and full_responses[i]:
                            from .robot_reasoning import ReasoningFormatValidator
                            format_valid = ReasoningFormatValidator.validate_format(full_responses[i])
                            format_accuracy.append(1.0 if format_valid else 0.0)

                            # Reasoning quality (like deepseek-r1's quality assessment)
                            xml_score = ReasoningFormatValidator.count_xml_structure(full_responses[i])
                            reasoning_quality_scores.append(xml_score)
                        else:
                            format_accuracy.append(0.0)
                            reasoning_quality_scores.append(0.0)

                # Calculate metrics
                if batch_accuracy:
                    metrics['robot_selection_accuracy'] = sum(batch_accuracy) / len(batch_accuracy)
                if format_accuracy:
                    metrics['reasoning_format_accuracy'] = sum(format_accuracy) / len(format_accuracy)
                if reasoning_quality_scores:
                    metrics['reasoning_quality_score'] = sum(reasoning_quality_scores) / len(reasoning_quality_scores)

            return metrics

        def log_robot_reasoning_examples(self, batch, outputs, step):
            """Log robot reasoning examples for inspection (like deepseek-r1's example logging)"""
            is_robot_batch = batch.get('is_robot_reasoning', torch.tensor([False]))
            if isinstance(is_robot_batch, torch.Tensor):
                is_robot_batch = is_robot_batch.any().item()
            elif isinstance(is_robot_batch, (list, tuple)):
                is_robot_batch = any(is_robot_batch)

            if not is_robot_batch:
                return

            # Log first example in batch (like deepseek-r1's example inspection)
            reasoning_texts = batch.get('reasoning_text', [])
            answer_texts = batch.get('answer_text', [])
            robot_labels = batch.get('robot_labels', [])

            if len(reasoning_texts) > 0:
                example_reasoning = reasoning_texts[0] if reasoning_texts[0] else "No reasoning generated"
                example_answer = answer_texts[0] if len(answer_texts) > 0 and answer_texts[0] else "No answer generated"
                example_labels = robot_labels[0] if len(robot_labels) > 0 and robot_labels[0] else "No ground truth"

                logger.info(f"🤖 Robot Reasoning Example (Step {step}) - deepseek-r1 Style:")
                logger.info(f"   Reasoning: {example_reasoning[:200]}...")
                logger.info(f"   Answer: {example_answer}")
                logger.info(f"   Ground Truth: {example_labels}")

                # Log format validation
                if len(batch.get('full_response', [])) > 0 and batch['full_response'][0]:
                    from .robot_reasoning import ReasoningFormatValidator
                    format_valid = ReasoningFormatValidator.validate_format(batch['full_response'][0])
                    xml_score = ReasoningFormatValidator.count_xml_structure(batch['full_response'][0])
                    logger.info(f"   Format Valid: {format_valid}, XML Score: {xml_score:.3f}")

        def compute_tiny_r1_style_rewards(self, batch, outputs, reward_functions):
            """Compute deepseek-r1 style multiple rewards for robot reasoning"""
            rewards = {}

            # Extract completions and ground truth
            full_responses = batch.get('full_response', [])
            ground_truth = batch.get('ground_truth', [])

            if not full_responses or not ground_truth:
                return rewards

            # Apply all reward functions (like deepseek-r1's reward_funcs)
            try:
                reward_results = {}
                for i, reward_func in enumerate(reward_functions):
                    if i == 0:  # Correctness reward
                        reward_results['correctness'] = reward_func(full_responses, ground_truth)
                    elif i == 1:  # Validity reward
                        reward_results['validity'] = reward_func(full_responses)
                    elif i == 2:  # Strict format reward
                        reward_results['strict_format'] = reward_func(full_responses)
                    elif i == 3:  # Soft format reward
                        reward_results['soft_format'] = reward_func(full_responses)
                    elif i == 4:  # XML count reward
                        reward_results['xml_count'] = reward_func(full_responses)
                    elif i == 5:  # Reasoning quality reward
                        reward_results['reasoning_quality'] = reward_func(full_responses)

                # Average rewards for logging
                for reward_name, reward_values in reward_results.items():
                    if reward_values:
                        rewards[f'reward_{reward_name}'] = sum(reward_values) / len(reward_values)

            except Exception as e:
                logger.warning(f"Failed to compute deepseek-r1 style rewards: {e}")

            return rewards

    return RobotReasoningTrainerMixin
