"""
Robot Selection Reasoning Module for BitGen
Implements structured reasoning similar to deepseek-r1 but for robot selection tasks
Uses XML-structured reasoning format with episodic memory integration
Following deepseek-r1's multi-reward training approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


# deepseek-r1 style system prompt for robot selection
ROBOT_REASONING_SYSTEM_PROMPT = """You are an expert robot selection assistant. Analyze the given task and select the most suitable robot(s) from the available options.

Respond in the following format:
<reasoning>
Analyze the task requirements, environment constraints, and match them with robot capabilities and limitations. Consider all factors before making your selection.
</reasoning>
<answer>
Selected robot(s): [Robot Name(s)]
</answer>"""


class RobotReasoningProcessor:
    """Processes robot selection data and creates reasoning-formatted training examples following deepseek-r1 approach"""

    def __init__(self, robot_data_dir: str):
        self.robot_data_dir = Path(robot_data_dir)
        self.single_robot_data = None
        self.multi_robot_data = None
        self.robot_capabilities = {}
        self.robot_types = ['Drone', 'Underwater Robot', 'Humanoid', 'Robot with Wheels', 'Robot with Legs']
        self.load_robot_data()

    def load_robot_data(self):
        """Load robot selection datasets"""
        try:
            # Load single robot selection data
            single_path = self.robot_data_dir / "Single-Robot-Selection" / "single_robot_selection_dataset.json"
            with open(single_path, 'r') as f:
                self.single_robot_data = json.load(f)

            # Load multi robot selection data
            multi_path = self.robot_data_dir / "Multi-Robot-Selection" / "multi_robot_selection_dataset.json"
            with open(multi_path, 'r') as f:
                self.multi_robot_data = json.load(f)

            # Extract robot capabilities from the instruction format
            self.extract_robot_capabilities()

            logger.info(f"✅ Robot data loaded: {len(self.single_robot_data)} single-robot, {len(self.multi_robot_data)} multi-robot examples")

        except Exception as e:
            logger.error(f"Failed to load robot data: {e}")
            raise

    def extract_robot_capabilities(self):
        """Extract robot capabilities from the instruction format"""
        if not self.single_robot_data:
            return

        # Parse robot capabilities from instruction text
        instruction = self.single_robot_data[0]['instruction']

        # Initialize robot capabilities
        for robot in self.robot_types:
            self.robot_capabilities[robot] = {
                'capabilities': [],
                'limitations': [],
                'environments': []
            }

        # Parse capabilities, limitations, and environments for each robot
        lines = instruction.split('\n')
        current_robot = None

        for line in lines:
            line = line.strip()
            if line.endswith(':') and any(robot in line for robot in self.robot_types):
                for robot in self.robot_types:
                    if robot in line:
                        current_robot = robot
                        break
            elif current_robot and line.startswith('capabilities:'):
                caps = line.replace('capabilities:', '').strip().rstrip(',')
                self.robot_capabilities[current_robot]['capabilities'] = [c.strip() for c in caps.split(',') if c.strip()]
            elif current_robot and line.startswith('limitations:'):
                lims = line.replace('limitations:', '').strip().rstrip(',')
                self.robot_capabilities[current_robot]['limitations'] = [l.strip() for l in lims.split(',') if l.strip()]
            elif current_robot and line.startswith('environments:'):
                envs = line.replace('environments:', '').strip().rstrip(',')
                self.robot_capabilities[current_robot]['environments'] = [e.strip() for e in envs.split(',') if e.strip()]

    def create_reasoning_examples(self) -> List[Dict]:
        """Create reasoning-formatted examples following deepseek-r1's XML structure"""
        examples = []

        # Process single robot examples (following deepseek-r1's format)
        for item in self.single_robot_data:
            reasoning = self._generate_structured_reasoning(item['input'], item['output'])

            # Create deepseek-r1 style conversation format
            conversation = [
                {"role": "system", "content": ROBOT_REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Available robots: {', '.join(self.robot_types)}\n\nTask: {item['input']}"},
                {"role": "assistant", "content": f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\nSelected robot(s): {item['output']}\n</answer>"}
            ]

            examples.append({
                'type': 'single_robot',
                'conversation': conversation,
                'reasoning': reasoning,
                'answer': item['output'],
                'task': item['input'],
                'ground_truth': item['output']
            })

        # Process multi robot examples
        for item in self.multi_robot_data:
            if 'subtasks' in item:
                # Extract unique robots from subtasks
                selected_robots = list(set([subtask['assigned_robot'] for subtask in item['subtasks']]))
                robot_output = ', '.join(selected_robots)

                reasoning = self._generate_multi_robot_reasoning(item['input'], item['subtasks'])

                conversation = [
                    {"role": "system", "content": ROBOT_REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Available robots: {', '.join(self.robot_types)}\n\nTask: {item['input']}"},
                    {"role": "assistant", "content": f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\nSelected robot(s): {robot_output}\n</answer>"}
                ]

                examples.append({
                    'type': 'multi_robot',
                    'conversation': conversation,
                    'reasoning': reasoning,
                    'answer': robot_output,
                    'task': item['input'],
                    'ground_truth': robot_output,
                    'subtasks': item['subtasks']
                })

        logger.info(f"✅ Created {len(examples)} reasoning examples following deepseek-r1 format")
        return examples

    def _generate_structured_reasoning(self, task: str, selected_robot: str) -> str:
        """Generate structured reasoning following deepseek-r1's detailed analysis approach"""
        task_lower = task.lower()
        reasoning_parts = []

        # Task Analysis (following deepseek-r1's step-by-step approach)
        reasoning_parts.append(f"Task Analysis: {task}")

        # Environment Analysis
        environment_analysis = self._analyze_environment(task_lower)
        if environment_analysis:
            reasoning_parts.append(f"Environment Assessment: {environment_analysis}")

        # Capability Requirements
        capability_analysis = self._analyze_required_capabilities(task_lower)
        if capability_analysis:
            reasoning_parts.append(f"Required Capabilities: {capability_analysis}")

        # Robot Evaluation (similar to deepseek-r1's detailed reasoning)
        robot_evaluation = self._evaluate_robot_suitability(task_lower, selected_robot)
        reasoning_parts.extend(robot_evaluation)

        # Final Selection Justification
        reasoning_parts.append(f"Conclusion: {selected_robot} is the optimal choice for this task")

        return "\n".join(reasoning_parts)

    def _analyze_environment(self, task_lower: str) -> str:
        """Analyze task environment requirements"""
        if any(word in task_lower for word in ['underwater', 'marine', 'ocean', 'pipes', 'seabed']):
            return "Underwater/aquatic environment requires waterproof design and underwater navigation capabilities"
        elif any(word in task_lower for word in ['aerial', 'air', 'above', 'high-rise', 'exterior', 'from above']):
            return "Aerial/elevated environment requires flight capability and aerial maneuvering systems"
        elif any(word in task_lower for word in ['rough', 'rocky', 'mountain', 'uneven', 'terrain', 'forest']):
            return "Challenging terrain requires stability, balance, and navigation on irregular surfaces"
        elif any(word in task_lower for word in ['flat', 'desert', 'warehouse', 'road', 'industrial']):
            return "Flat/structured environment allows for efficient wheeled movement and stable platforms"
        elif any(word in task_lower for word in ['indoor', 'building', 'stairs', 'urban', 'pedestrian']):
            return "Indoor/human environment requires careful navigation and potential human interaction"
        return "Standard operational environment"

    def _analyze_required_capabilities(self, task_lower: str) -> str:
        """Analyze required capabilities for the task"""
        capabilities = []

        if any(word in task_lower for word in ['inspect', 'survey', 'monitor', 'check', 'assess']):
            capabilities.append("inspection and surveillance sensors")
        if any(word in task_lower for word in ['deliver', 'transport', 'carry', 'supplies', 'package']):
            capabilities.append("payload capacity and reliable transport")
        if any(word in task_lower for word in ['explore', 'navigate', 'traverse', 'map']):
            capabilities.append("autonomous navigation and exploration")
        if any(word in task_lower for word in ['interaction', 'human', 'pedestrian', 'crowded']):
            capabilities.append("safe human interaction and social navigation")
        if any(word in task_lower for word in ['manipulation', 'tool', 'complex']):
            capabilities.append("advanced manipulation and tool use")

        return ", ".join(capabilities) if capabilities else "basic operational capabilities"

    def _evaluate_robot_suitability(self, task_lower: str, selected_robot: str) -> List[str]:
        """Evaluate why specific robot was selected (deepseek-r1 style detailed analysis)"""
        evaluation = []

        if selected_robot not in self.robot_capabilities:
            return [f"Robot {selected_robot} evaluation: capabilities unknown"]

        robot_caps = self.robot_capabilities[selected_robot]

        # Positive capability matching
        matching_caps = []
        if any(cap in task_lower for cap in ['aerial', 'above', 'high-rise', 'survey']):
            if any('aerial' in cap or 'surveillance' in cap for cap in robot_caps['capabilities']):
                matching_caps.append("aerial navigation and surveillance")

        if any(cap in task_lower for cap in ['underwater', 'marine', 'seabed']):
            if any('underwater' in cap or 'marine' in cap for cap in robot_caps['capabilities']):
                matching_caps.append("underwater navigation and marine operations")

        if any(cap in task_lower for cap in ['rough', 'rocky', 'mountain', 'terrain']):
            if any('rough terrain' in cap or 'stability' in cap for cap in robot_caps['capabilities']):
                matching_caps.append("rough terrain navigation and stability")

        if any(cap in task_lower for cap in ['fast', 'rapid', 'desert', 'warehouse']):
            if any('fast' in cap or 'efficient' in cap for cap in robot_caps['capabilities']):
                matching_caps.append("fast movement and efficiency")

        if any(cap in task_lower for cap in ['human', 'pedestrian', 'interaction', 'delivery']):
            if any('interaction' in cap or 'manipulation' in cap for cap in robot_caps['capabilities']):
                matching_caps.append("human interaction and manipulation")

        if matching_caps:
            evaluation.append(f"Robot Selection Rationale: {selected_robot} provides {', '.join(matching_caps)}")

        # Capability justification
        key_capabilities = robot_caps['capabilities'][:3]  # Top 3 capabilities
        evaluation.append(f"Key Capabilities: {', '.join(key_capabilities)}")

        # Environment suitability
        suitable_envs = robot_caps['environments']
        evaluation.append(f"Suitable Environments: {', '.join(suitable_envs)}")

        return evaluation

    def _generate_multi_robot_reasoning(self, task: str, subtasks: List[Dict]) -> str:
        """Generate reasoning for multi-robot coordination (deepseek-r1 style structured analysis)"""
        reasoning_parts = []
        reasoning_parts.append(f"Complex Multi-Robot Task Analysis: {task}")
        reasoning_parts.append("Task decomposition requires specialized robot coordination:")

        # Analyze subtask sequence
        execution_phases = {}
        for subtask in subtasks:
            order = subtask['execution_order']
            if order not in execution_phases:
                execution_phases[order] = []
            execution_phases[order].append(subtask)

        # Sequential analysis (like deepseek-r1's step-by-step reasoning)
        for phase in sorted(execution_phases.keys()):
            phase_tasks = execution_phases[phase]
            reasoning_parts.append(f"Phase {phase}:")

            for task_info in phase_tasks:
                robot = task_info['assigned_robot']
                subtask_desc = task_info['subtask']

                # Justify robot selection for this subtask
                justification = self._justify_subtask_robot_selection(subtask_desc, robot)
                reasoning_parts.append(f"  • {subtask_desc} → {robot}")
                reasoning_parts.append(f"    Rationale: {justification}")

        # Coordination strategy
        if len(execution_phases) > 1:
            reasoning_parts.append("Coordination Strategy: Sequential execution with proper task handoffs between specialized robots")
        else:
            reasoning_parts.append("Coordination Strategy: Parallel execution for simultaneous task completion")

        # Extract final robot selection
        selected_robots = list(set([subtask['assigned_robot'] for subtask in subtasks]))
        reasoning_parts.append(f"Final Selection: {', '.join(selected_robots)} working in coordination")

        return "\n".join(reasoning_parts)

    def _justify_subtask_robot_selection(self, subtask: str, robot: str) -> str:
        """Justify robot selection for specific subtask"""
        subtask_lower = subtask.lower()

        justifications = {
            'Drone': {
                'keywords': ['aerial', 'overview', 'above', 'facade', 'exterior'],
                'reason': "optimal for aerial operations and elevated access"
            },
            'Underwater Robot': {
                'keywords': ['underwater', 'marine', 'seabed', 'pipes', 'aquatic'],
                'reason': "specialized for underwater operations and aquatic environments"
            },
            'Humanoid': {
                'keywords': ['documentation', 'coordination', 'complex', 'manipulation', 'interaction'],
                'reason': "excels at complex tasks requiring manipulation and coordination"
            },
            'Robot with Wheels': {
                'keywords': ['transport', 'equipment', 'warehouse', 'fast', 'efficient'],
                'reason': "provides fast and efficient movement on flat surfaces"
            },
            'Robot with Legs': {
                'keywords': ['ground', 'foundation', 'terrain', 'stability', 'inspection'],
                'reason': "offers stability and navigation on varied terrain"
            }
        }

        if robot in justifications:
            robot_info = justifications[robot]
            if any(keyword in subtask_lower for keyword in robot_info['keywords']):
                return robot_info['reason']

        return f"selected for task-specific requirements"


class ReasoningFormatValidator:
    """Validates reasoning format following deepseek-r1's validation approach"""

    @staticmethod
    def extract_reasoning(text: str) -> str:
        """Extract reasoning content from XML tags (like deepseek-r1's extract_xml_answer)"""
        try:
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
            return reasoning_match.group(1).strip() if reasoning_match else ""
        except:
            return ""

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract answer content from XML tags"""
        try:
            answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            answer_content = answer_match.group(1).strip() if answer_match else ""

            # Extract robot names from "Selected robot(s): ..." format
            if 'Selected robot(s):' in answer_content:
                robots = answer_content.split('Selected robot(s):')[1].strip()
                return robots
            return answer_content
        except:
            return ""

    @staticmethod
    def validate_format(text: str) -> bool:
        """Validate XML format structure (following deepseek-r1's strict_format_reward_func)"""
        has_reasoning = '<reasoning>' in text and '</reasoning>' in text
        has_answer = '<answer>' in text and '</answer>' in text

        # Check for proper structure (similar to deepseek-r1's pattern matching)
        reasoning_pattern = r'<reasoning>.*?</reasoning>'
        answer_pattern = r'<answer>.*?</answer>'

        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        answer_match = re.search(answer_pattern, text, re.DOTALL)

        return has_reasoning and has_answer and reasoning_match is not None and answer_match is not None

    @staticmethod
    def count_xml_structure(text: str) -> float:
        """Count XML structure quality (following deepseek-r1's xmlcount_reward_func)"""
        score = 0.0

        # Reasoning tags
        if text.count('<reasoning>') == 1:
            score += 0.125
        if text.count('</reasoning>') == 1:
            score += 0.125

        # Answer tags
        if text.count('<answer>') == 1:
            score += 0.125
        if text.count('</answer>') == 1:
            score += 0.125

        # Penalty for extra content after closing tags (like deepseek-r1)
        if '</answer>' in text:
            remaining_content = text.split('</answer>')[-1].strip()
            if remaining_content:
                score -= len(remaining_content) * 0.001

        return min(score, 0.5)  # Cap at 0.5 like deepseek-r1


class RobotSelectionRewardFunctions:
    """Reward functions for robot selection reasoning training (following deepseek-r1's multi-reward approach)"""

    def __init__(self, robot_processor: RobotReasoningProcessor):
        self.robot_processor = robot_processor
        self.available_robots = robot_processor.robot_types

    def robot_correctness_reward_func(self, completions: List[str], ground_truth: List[str]) -> List[float]:
        """Reward for correct robot selection (following deepseek-r1's correctness_reward_func)"""
        rewards = []
        for completion, truth in zip(completions, ground_truth):
            extracted_robots = ReasoningFormatValidator.extract_answer(completion)

            # Normalize robot names for comparison
            pred_robots = set(r.strip() for r in extracted_robots.split(',') if r.strip())
            true_robots = set(r.strip() for r in truth.split(',') if r.strip())

            # Perfect match gets full reward (like deepseek-r1's 2.0 reward)
            if pred_robots == true_robots:
                rewards.append(2.0)
            elif pred_robots.intersection(true_robots):
                # Partial match (proportional reward)
                overlap = len(pred_robots.intersection(true_robots))
                total = len(pred_robots.union(true_robots))
                rewards.append(overlap / total)
            else:
                rewards.append(0.0)

        return rewards

    def robot_validity_reward_func(self, completions: List[str]) -> List[float]:
        """Reward for selecting valid robots (similar to deepseek-r1's int_reward_func)"""
        rewards = []
        for completion in completions:
            extracted_robots = ReasoningFormatValidator.extract_answer(completion)

            if not extracted_robots.strip():
                rewards.append(0.0)
                continue

            robot_list = [r.strip() for r in extracted_robots.split(',') if r.strip()]
            valid_count = sum(1 for robot in robot_list if robot in self.available_robots)

            if len(robot_list) > 0:
                validity_score = valid_count / len(robot_list)
                rewards.append(0.5 * validity_score)  # 0.5 max like deepseek-r1's int_reward
            else:
                rewards.append(0.0)

        return rewards

    def strict_format_reward_func(self, completions: List[str]) -> List[float]:
        """Strict format validation reward (following deepseek-r1's strict_format_reward_func)"""
        rewards = []
        # Adapted pattern for robot selection
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\nSelected robot\(s\):.*?\n</answer>$"

        for completion in completions:
            if re.match(pattern, completion.strip(), re.DOTALL):
                rewards.append(0.5)
            else:
                rewards.append(0.0)

        return rewards

    def soft_format_reward_func(self, completions: List[str]) -> List[float]:
        """Soft format validation reward (following deepseek-r1's soft_format_reward_func)"""
        rewards = []
        for completion in completions:
            if ReasoningFormatValidator.validate_format(completion):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards

    def xmlcount_reward_func(self, completions: List[str]) -> List[float]:
        """XML structure counting reward (following deepseek-r1's xmlcount_reward_func)"""
        return [ReasoningFormatValidator.count_xml_structure(completion) for completion in completions]

    def reasoning_quality_reward_func(self, completions: List[str]) -> List[float]:
        """Reward for reasoning quality (custom for robot selection)"""
        rewards = []
        for completion in completions:
            reasoning = ReasoningFormatValidator.extract_reasoning(completion)

            quality_score = 0.0

            # Check for structured analysis components
            if any(word in reasoning.lower() for word in ['task analysis', 'analysis:', 'assess']):
                quality_score += 0.2
            if any(word in reasoning.lower() for word in ['environment', 'terrain', 'condition']):
                quality_score += 0.2
            if any(word in reasoning.lower() for word in ['capability', 'capabilities', 'suited', 'optimal']):
                quality_score += 0.2
            if any(word in reasoning.lower() for word in ['rationale', 'reason', 'because', 'justification']):
                quality_score += 0.2
            if any(word in reasoning.lower() for word in ['conclusion', 'therefore', 'final', 'selected']):
                quality_score += 0.2

            rewards.append(quality_score)

        return rewards


class RobotSelectionHead(nn.Module):
    """Specialized head for robot selection with reasoning (enhanced for deepseek-r1 style training)"""

    def __init__(self, input_dim: int, num_robots: int = 5, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_robots = num_robots

        # Robot types (matching the dataset)
        self.robot_types = ['Drone', 'Underwater Robot', 'Humanoid', 'Robot with Wheels', 'Robot with Legs']

        # Enhanced reasoning encoder (deeper for better structured thinking)
        self.reasoning_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

        # Robot selection heads (one per robot type)
        self.robot_selection_heads = nn.ModuleDict({
            robot.replace(' ', '_'): nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()  # Probability of selection
            )
            for robot in self.robot_types
        })

        # Task analysis components
        self.task_analyzer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 64)  # Task embedding
        )

        # Environment classifier
        self.environment_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 8)  # Environment types
        )

        # Reasoning quality predictor (for self-evaluation)
        self.reasoning_quality_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for robot selection reasoning"""
        # Handle different input shapes
        if fused_features.dim() == 3:
            # Pool sequence features (like deepseek-r1's processing)
            pooled_features = fused_features.mean(dim=1)  # [batch_size, input_dim]
        else:
            pooled_features = fused_features

        # Apply reasoning encoder
        reasoning_features = self.reasoning_encoder(pooled_features)

        # Generate robot selection probabilities
        robot_selections = {}
        for robot_key, head in self.robot_selection_heads.items():
            robot_selections[robot_key] = head(reasoning_features).squeeze(-1)

        # Additional analysis outputs
        task_embedding = self.task_analyzer(reasoning_features)
        environment_logits = self.environment_classifier(reasoning_features)
        reasoning_quality = self.reasoning_quality_head(reasoning_features).squeeze(-1)

        return {
            'robot_selections': robot_selections,
            'task_embedding': task_embedding,
            'environment_logits': environment_logits,
            'reasoning_features': reasoning_features,
            'reasoning_quality': reasoning_quality
        }


class RobotReasoningIntegration:
    """Integration class for adding deepseek-r1 style reasoning to BitGen model"""

    def __init__(self, model, robot_data_dir: str):
        self.model = model
        self.robot_processor = RobotReasoningProcessor(robot_data_dir)
        self.reward_functions = RobotSelectionRewardFunctions(self.robot_processor)

        # Add robot selection head to the model
        if hasattr(model, 'config'):
            hidden_dim = model.config.get('fusion_hidden_size', 128)
        else:
            hidden_dim = 128  # Default

        self.robot_selection_head = RobotSelectionHead(hidden_dim)

        # Add to model
        if not hasattr(model, 'robot_reasoning'):
            model.robot_reasoning = self.robot_selection_head
            logger.info("✅ Robot reasoning head integrated into BitGen model")

    def prepare_robot_reasoning_training_data(self) -> Dict:
        """Prepare robot reasoning data for training (following deepseek-r1's data preparation)"""
        examples = self.robot_processor.create_reasoning_examples()

        # Convert to deepseek-r1 style training format
        training_data = {
            'conversations': [],
            'ground_truth_answers': [],
            'reasoning_texts': [],
            'robot_labels': [],
            'task_types': []
        }

        for example in examples:
            training_data['conversations'].append(example['conversation'])
            training_data['ground_truth_answers'].append(example['answer'])
            training_data['reasoning_texts'].append(example['reasoning'])
            training_data['robot_labels'].append(example['ground_truth'])
            training_data['task_types'].append(example['type'])

        logger.info(f"✅ Prepared {len(examples)} robot reasoning examples for training")
        return training_data

    def compute_robot_reasoning_rewards(self, completions: List[str], ground_truth: List[str]) -> Dict[str, List[float]]:
        """Compute all reward functions (following deepseek-r1's multi-reward approach)"""
        return {
            'robot_correctness': self.reward_functions.robot_correctness_reward_func(completions, ground_truth),
            'robot_validity': self.reward_functions.robot_validity_reward_func(completions),
            'strict_format': self.reward_functions.strict_format_reward_func(completions),
            'soft_format': self.reward_functions.soft_format_reward_func(completions),
            'xml_structure': self.reward_functions.xmlcount_reward_func(completions),
            'reasoning_quality': self.reward_functions.reasoning_quality_reward_func(completions)
        }

    def compute_reasoning_loss(self, outputs: Dict, robot_labels: List[str], completions: List[str] = None) -> torch.Tensor:
        """Compute loss for robot selection reasoning (following deepseek-r1's multi-objective approach)"""
        if not hasattr(self.model, 'robot_reasoning'):
            return torch.tensor(0.0, device=outputs['fused_features'].device)

        # Get robot selection predictions
        robot_outputs = self.model.robot_reasoning(outputs['fused_features'])

        # Convert labels to multi-hot encoding for robot selection
        batch_size = len(robot_labels)
        device = outputs['fused_features'].device

        targets = torch.zeros(batch_size, len(self.robot_processor.robot_types), device=device)

        for i, label in enumerate(robot_labels):
            if isinstance(label, str) and label.strip():
                selected_robots = [r.strip() for r in label.split(',') if r.strip()]
                for robot in selected_robots:
                    if robot in self.robot_processor.robot_types:
                        robot_idx = self.robot_processor.robot_types.index(robot)
                        targets[i, robot_idx] = 1.0

        # Compute binary cross-entropy loss for each robot
        robot_selection_loss = 0.0
        for i, robot in enumerate(self.robot_processor.robot_types):
            robot_key = robot.replace(' ', '_')
            if robot_key in robot_outputs['robot_selections']:
                pred = robot_outputs['robot_selections'][robot_key]
                target = targets[:, i]
                loss = F.binary_cross_entropy(pred, target)
                robot_selection_loss += loss

        robot_selection_loss = robot_selection_loss / len(self.robot_processor.robot_types)

        # Add reasoning quality loss (encourage higher quality reasoning)
        reasoning_quality_loss = 0.0
        if 'reasoning_quality' in robot_outputs:
            # Encourage high-quality reasoning (target = 1.0)
            quality_targets = torch.ones_like(robot_outputs['reasoning_quality'])
            reasoning_quality_loss = F.binary_cross_entropy(robot_outputs['reasoning_quality'], quality_targets)

        # Combined robot reasoning loss
        total_robot_loss = robot_selection_loss + 0.1 * reasoning_quality_loss

        return total_robot_loss

    def generate_robot_reasoning(self, task: str, context: str = None, vision_features: torch.Tensor = None) -> Dict:
        """Generate structured reasoning for robot selection (following deepseek-r1's generation approach)"""
        # Create prompt in deepseek-r1 style
        user_prompt = f"Available robots: {', '.join(self.robot_processor.robot_types)}\n\nTask: {task}"
        if context:
            user_prompt += f"\n\nAdditional context: {context}"

        # Tokenize the conversation
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Format as conversation (like deepseek-r1)
        conversation = [
            {"role": "system", "content": ROBOT_REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Convert to text format
        prompt_text = ""
        for turn in conversation:
            if turn["role"] == "system":
                prompt_text += f"System: {turn['content']}\n\n"
            elif turn["role"] == "user":
                prompt_text += f"User: {turn['content']}\n\nAssistant: "

        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)

        # Use dummy vision features if not provided
        if vision_features is None:
            vision_features = torch.randn(input_ids.size(0), 768, device=self.model.device)

        # Generate with the model (following deepseek-r1's generation parameters)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                vision_features=vision_features,
                max_length=256,
                temperature=0.8,  # Like deepseek-r1's temperature
                do_sample=True,
                top_p=0.9,
                attention_mask=attention_mask
            )

        # Decode response
        response = tokenizer.decode(generated[0][input_ids.size(1):], skip_special_tokens=True)

        # Extract reasoning and answer (like deepseek-r1's extraction)
        reasoning = ReasoningFormatValidator.extract_reasoning(response)
        answer = ReasoningFormatValidator.extract_answer(response)

        # Get model's internal robot selection predictions
        try:
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_features=vision_features
            )

            robot_predictions = {}
            if hasattr(self.model, 'robot_reasoning'):
                robot_outputs = self.model.robot_reasoning(model_outputs['fused_features'])
                robot_predictions = robot_outputs['robot_selections']
        except Exception as e:
            logger.warning(f"Failed to get robot predictions: {e}")
            robot_predictions = {}

        return {
            'task': task,
            'context': context,
            'reasoning': reasoning,
            'selected_robots': answer,
            'full_response': response,
            'robot_probabilities': robot_predictions,
            'format_valid': ReasoningFormatValidator.validate_format(response),
            'xml_structure_score': ReasoningFormatValidator.count_xml_structure(response)
        }

    def get_reward_functions(self):
        """Get all reward functions for training (like deepseek-r1's reward_funcs list)"""
        return [
            self.reward_functions.robot_correctness_reward_func,
            self.reward_functions.robot_validity_reward_func,
            self.reward_functions.strict_format_reward_func,
            self.reward_functions.soft_format_reward_func,
            self.reward_functions.xmlcount_reward_func,
            self.reward_functions.reasoning_quality_reward_func
        ]


def create_robot_reasoning_integration(model, robot_data_dir: str = "D:/BabyLM/robot_selection_data/data") -> RobotReasoningIntegration:
    """Create and integrate robot reasoning capabilities into BitGen model (following deepseek-r1's integration approach)"""
    logger.info("🤖 Integrating deepseek-r1 style robot selection reasoning into BitGen model...")

    integration = RobotReasoningIntegration(model, robot_data_dir)

    logger.info("✅ Robot reasoning integration completed")
    logger.info(f"   Available robots: {', '.join(integration.robot_processor.robot_types)}")
    logger.info(f"   Training examples: {len(integration.robot_processor.single_robot_data)} single + {len(integration.robot_processor.multi_robot_data)} multi")
    logger.info(f"   Reward functions: {len(integration.get_reward_functions())} (following deepseek-r1 approach)")

    return integration
