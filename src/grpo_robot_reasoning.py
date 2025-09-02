"""
Enhanced Robot Reasoning Integration for GRPO Training
Extends the existing robot reasoning system with policy optimization capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from pathlib import Path

from src.robot_reasoning import (
    RobotReasoningProcessor, 
    ReasoningFormatValidator, 
    RobotSelectionRewardFunctions,
    RobotReasoningIntegration
)

logger = logging.getLogger(__name__)


class PolicyOptimizedRobotHead(nn.Module):
    """Enhanced robot selection head optimized for policy-based training"""
    
    def __init__(self, input_dim: int, num_robots: int = 5, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_robots = num_robots
        
        # Robot types
        self.robot_types = ['Drone', 'Underwater_Robot', 'Humanoid', 'Robot_with_Wheels', 'Robot_with_Legs']
        
        # Enhanced reasoning encoder with residual connections
        self.reasoning_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.LayerNorm(input_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim)
            )
        ])
        
        # Policy-based robot selection heads (output probability distributions)
        self.robot_policy_heads = nn.ModuleDict({
            robot: nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, 2),  # [not_selected, selected] probabilities
                nn.Softmax(dim=-1)
            )
            for robot in self.robot_types
        })
        
        # Value head for policy optimization (estimates expected reward)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1)  # Scalar value estimate
        )
        
        # Top-N selection controller
        self.top_n_controller = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 5),  # Probability distribution over N (1-5 robots)
            nn.Softmax(dim=-1)
        )
        
        # Task complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 3),  # [simple, moderate, complex]
            nn.Softmax(dim=-1)
        )
        
        # Reasoning quality predictor
        self.reasoning_quality_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass for policy-based robot selection"""
        
        # Handle different input shapes
        if fused_features.dim() == 3:
            # Pool sequence features
            pooled_features = fused_features.mean(dim=1)
        else:
            pooled_features = fused_features
            
        # Apply reasoning encoder with residual connections
        x = pooled_features
        residual = x
        
        for i, layer in enumerate(self.reasoning_encoder):
            x = layer(x)
            if i == 1:  # Add residual connection
                x = x + residual
                
        reasoning_features = x
        
        # Generate policy distributions for each robot
        robot_policies = {}
        robot_selections = {}
        
        for robot_name, policy_head in self.robot_policy_heads.items():
            policy_dist = policy_head(reasoning_features)  # [batch_size, 2]
            robot_policies[robot_name] = policy_dist
            robot_selections[robot_name] = policy_dist[:, 1]  # Probability of selection
            
        # Estimate state value
        state_value = self.value_head(reasoning_features)
        
        # Predict optimal number of robots
        top_n_dist = self.top_n_controller(reasoning_features)
        
        # Estimate task complexity
        complexity_dist = self.complexity_estimator(reasoning_features)
        
        # Predict reasoning quality
        reasoning_quality = self.reasoning_quality_head(reasoning_features)
        
        return {
            'robot_policies': robot_policies,          # Full policy distributions
            'robot_selections': robot_selections,      # Selection probabilities
            'state_value': state_value,                # Value estimate for RL
            'top_n_distribution': top_n_dist,          # Optimal N distribution  
            'complexity_distribution': complexity_dist, # Task complexity
            'reasoning_features': reasoning_features,   # Encoded features
            'reasoning_quality': reasoning_quality      # Quality prediction
        }
    
    def sample_robot_selection(self, outputs: Dict[str, torch.Tensor], temperature: float = 1.0) -> Dict[str, bool]:
        """Sample robot selection from policy distributions"""
        robot_policies = outputs['robot_policies']
        selected_robots = {}
        
        for robot_name, policy_dist in robot_policies.items():
            # Apply temperature scaling
            scaled_logits = torch.log(policy_dist + 1e-8) / temperature
            scaled_probs = F.softmax(scaled_logits, dim=-1)
            
            # Sample from distribution
            samples = torch.multinomial(scaled_probs, 1)
            selected_robots[robot_name] = samples[:, 0].bool()  # True if selected
            
        return selected_robots
    
    def get_top_n_robots(self, outputs: Dict[str, torch.Tensor], n: Optional[int] = None) -> List[str]:
        """Get top-N robots based on selection probabilities"""
        robot_selections = outputs['robot_selections']
        
        # If n not specified, sample from top_n_distribution
        if n is None:
            top_n_dist = outputs['top_n_distribution']
            n = torch.multinomial(top_n_dist, 1).item() + 1  # 1-5 robots
            
        # Get top N robots by probability
        all_probs = torch.stack([robot_selections[robot] for robot in self.robot_types])
        top_indices = torch.topk(all_probs, n, dim=0).indices
        
        return [self.robot_types[i] for i in top_indices.cpu().numpy()]


class GRPORobotReasoningIntegration(RobotReasoningIntegration):
    """Enhanced robot reasoning integration with GRPO support"""
    
    def __init__(self, model, robot_data_dir: str, config: Dict = None):
        super().__init__(model, robot_data_dir)
        
        self.config = config or {}
        self.grpo_enabled = True
        
        # Replace standard head with policy-optimized head
        hidden_dim = getattr(model, 'config', {}).get('fusion_hidden_size', 128)
        if hasattr(model.config, 'hidden_size'):
            hidden_dim = model.config.hidden_size
            
        self.robot_selection_head = PolicyOptimizedRobotHead(hidden_dim)
        model.robot_reasoning = self.robot_selection_head
        
        # Enhanced reward functions for GRPO
        self.grpo_reward_functions = GRPORobotRewardFunctions(self.robot_processor, config)
        
        logger.info("✅ GRPO Robot Reasoning Integration initialized")
        logger.info(f"   • Policy-based robot head: {sum(p.numel() for p in self.robot_selection_head.parameters()):,} parameters")

    def generate_robot_reasoning_with_policy(
        self,
        task: str,
        context: Optional[str] = None,
        vision_features: Optional[torch.Tensor] = None,
        temperature: float = 0.7,
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate robot reasoning using policy-based selection"""
        
        # Create input features (simplified for demo)
        if vision_features is not None:
            input_features = vision_features
        else:
            # Create dummy features for text-only task
            input_features = torch.randn(1, 128)  # [batch_size, hidden_dim]
            
        # Get policy outputs
        with torch.no_grad():
            outputs = self.robot_selection_head(input_features)
            
            # Sample robot selection
            selected_robots_dict = self.robot_selection_head.sample_robot_selection(
                outputs, temperature=temperature
            )
            
            # Get top-N robots
            top_n_robots = self.robot_selection_head.get_top_n_robots(outputs, n=top_n)
            
            # Estimate task complexity
            complexity_probs = outputs['complexity_distribution'][0].cpu().numpy()
            complexity_levels = ['simple', 'moderate', 'complex']
            estimated_complexity = complexity_levels[np.argmax(complexity_probs)]
            
            # Get reasoning quality prediction
            reasoning_quality = outputs['reasoning_quality'][0].item()
            
        # Generate structured reasoning
        reasoning = self._generate_policy_based_reasoning(
            task=task,
            selected_robots=top_n_robots,
            complexity=estimated_complexity,
            context=context
        )
        
        # Format final response
        robot_names = [name.replace('_', ' ') for name in top_n_robots]
        answer = f"Selected robot(s): {', '.join(robot_names)}"
        
        full_response = f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"
        
        return {
            'reasoning': reasoning,
            'selected_robots': robot_names,
            'top_n_robots': top_n_robots,
            'full_response': full_response,
            'policy_outputs': outputs,
            'estimated_complexity': estimated_complexity,
            'reasoning_quality': reasoning_quality,
            'selection_probabilities': {k.replace('_', ' '): v[0].item() for k, v in outputs['robot_selections'].items()}
        }

    def _generate_policy_based_reasoning(
        self,
        task: str,
        selected_robots: List[str],
        complexity: str,
        context: Optional[str] = None
    ) -> str:
        """Generate structured reasoning based on policy decisions"""
        
        reasoning_parts = []
        
        # Task analysis with complexity consideration
        reasoning_parts.append(f"Task Analysis: {task}")
        reasoning_parts.append(f"Estimated Task Complexity: {complexity.title()}")
        
        if complexity == "complex":
            reasoning_parts.append("This complex task requires careful coordination between multiple specialized robots.")
        elif complexity == "moderate":
            reasoning_parts.append("This moderately complex task can be handled by one or two suitable robots.")
        else:
            reasoning_parts.append("This is a straightforward task that can be efficiently completed by a single robot.")
            
        # Environment and requirement analysis
        task_lower = task.lower()
        
        if any(word in task_lower for word in ['underwater', 'marine', 'seabed', 'ocean']):
            reasoning_parts.append("Environment Analysis: Underwater/aquatic environment requiring specialized waterproof systems and marine navigation capabilities.")
        elif any(word in task_lower for word in ['aerial', 'sky', 'above', 'high-rise', 'roof']):
            reasoning_parts.append("Environment Analysis: Aerial/elevated environment requiring flight capabilities and elevated access systems.")
        elif any(word in task_lower for word in ['rough', 'mountain', 'rocky', 'uneven', 'terrain']):
            reasoning_parts.append("Environment Analysis: Challenging terrain requiring advanced stability and navigation systems.")
        elif any(word in task_lower for word in ['indoor', 'building', 'facility', 'hospital', 'office']):
            reasoning_parts.append("Environment Analysis: Indoor/structured environment requiring precise navigation and human interaction capabilities.")
        else:
            reasoning_parts.append("Environment Analysis: Standard operational environment with typical mobility requirements.")
            
        # Robot selection rationale
        reasoning_parts.append("Robot Selection Analysis:")
        
        robot_justifications = {
            'Drone': "Optimal for aerial operations with advanced surveillance capabilities and rapid deployment",
            'Underwater_Robot': "Specialized for aquatic environments with waterproof design and marine operation systems", 
            'Humanoid': "Excels in human environments with advanced manipulation and complex task execution capabilities",
            'Robot_with_Wheels': "Provides efficient high-speed movement on flat surfaces with excellent payload capacity",
            'Robot_with_Legs': "Offers superior stability and navigation on rough terrain with adaptive balance systems"
        }
        
        for robot in selected_robots:
            robot_key = robot if robot in robot_justifications else robot.replace(' ', '_')
            if robot_key in robot_justifications:
                reasoning_parts.append(f"• {robot.replace('_', ' ')}: {robot_justifications[robot_key]}")
                
        # Coordination strategy for multiple robots
        if len(selected_robots) > 1:
            reasoning_parts.append("Coordination Strategy:")
            reasoning_parts.append("Multiple robots selected to provide complementary capabilities and ensure comprehensive task completion.")
            reasoning_parts.append("Coordinated operation will maximize efficiency while providing redundancy for critical operations.")
        else:
            reasoning_parts.append("Single Robot Strategy:")
            reasoning_parts.append("Task requirements can be efficiently met by this specialized robot, optimizing resource utilization.")
            
        # Final conclusion
        robot_names = [r.replace('_', ' ') for r in selected_robots]
        reasoning_parts.append(f"Conclusion: {', '.join(robot_names)} selected based on optimal capability matching and task complexity analysis.")
        
        return '\n'.join(reasoning_parts)

    def compute_grpo_rewards(
        self,
        prompts: List[str],
        completions: List[List[Dict]],
        ground_truth: List[str],
        **kwargs
    ) -> Dict[str, List[float]]:
        """Compute GRPO-compatible rewards for robot reasoning"""
        
        return self.grpo_reward_functions.compute_all_rewards(
            prompts, completions, ground_truth, **kwargs
        )


class GRPORobotRewardFunctions:
    """Enhanced reward functions optimized for GRPO training"""
    
    def __init__(self, robot_processor: RobotReasoningProcessor, config: Dict = None):
        self.robot_processor = robot_processor
        self.config = config or {}
        self.base_rewards = RobotSelectionRewardFunctions(robot_processor)
        
        # Reward weights from config
        self.reward_weights = self.config.get('reward_weights', {
            'correctness': 0.30,
            'validity': 0.20,
            'format': 0.20,
            'reasoning_quality': 0.15,
            'top_n_efficiency': 0.15
        })
        
    def compute_all_rewards(
        self,
        prompts: List[str],
        completions: List[List[Dict]],
        ground_truth: List[str],
        **kwargs
    ) -> Dict[str, List[float]]:
        """Compute all GRPO rewards"""
        
        # Extract responses from completions
        responses = [completion[0]["content"] for completion in completions]
        
        # Compute individual reward components
        rewards = {
            'correctness': self._correctness_reward(responses, ground_truth),
            'validity': self._validity_reward(responses),
            'strict_format': self._strict_format_reward(responses),
            'soft_format': self._soft_format_reward(responses),
            'xml_count': self._xml_count_reward(responses),
            'reasoning_quality': self._reasoning_quality_reward(responses),
            'top_n_efficiency': self._top_n_efficiency_reward(responses, ground_truth),
            'diversity_bonus': self._diversity_bonus_reward(responses)
        }
        
        # Compute weighted total reward
        total_rewards = []
        for i in range(len(responses)):
            total_reward = 0.0
            total_reward += rewards['correctness'][i] * self.reward_weights.get('correctness', 0.30)
            total_reward += rewards['validity'][i] * self.reward_weights.get('validity', 0.20)
            total_reward += (rewards['strict_format'][i] + rewards['soft_format'][i]) * self.reward_weights.get('format', 0.20)
            total_reward += rewards['reasoning_quality'][i] * self.reward_weights.get('reasoning_quality', 0.15)
            total_reward += rewards['top_n_efficiency'][i] * self.reward_weights.get('top_n_efficiency', 0.15)
            total_reward += rewards['diversity_bonus'][i] * 0.05  # Small bonus
            
            total_rewards.append(total_reward)
            
        rewards['total'] = total_rewards
        
        return rewards
    
    def _correctness_reward(self, responses: List[str], ground_truth: List[str]) -> List[float]:
        """Robot selection correctness reward"""
        return self.base_rewards.robot_correctness_reward_func(responses, ground_truth)
    
    def _validity_reward(self, responses: List[str]) -> List[float]:
        """Robot validity reward"""  
        return self.base_rewards.robot_validity_reward_func(responses)
    
    def _strict_format_reward(self, responses: List[str]) -> List[float]:
        """Strict format reward"""
        return self.base_rewards.strict_format_reward_func(responses)
    
    def _soft_format_reward(self, responses: List[str]) -> List[float]:
        """Soft format reward"""
        return self.base_rewards.soft_format_reward_func(responses)
    
    def _xml_count_reward(self, responses: List[str]) -> List[float]:
        """XML structure counting reward"""
        return self.base_rewards.xmlcount_reward_func(responses)
    
    def _reasoning_quality_reward(self, responses: List[str]) -> List[float]:
        """Reasoning quality reward"""
        return self.base_rewards.reasoning_quality_reward_func(responses)
    
    def _top_n_efficiency_reward(self, responses: List[str], ground_truth: List[str]) -> List[float]:
        """Reward for efficient top-N robot selection"""
        rewards = []
        max_robots = self.config.get('max_robots_per_task', 3)
        
        for response, truth in zip(responses, ground_truth):
            extracted_robots = ReasoningFormatValidator.extract_answer(response)
            
            pred_robots = [r.strip() for r in extracted_robots.split(',') if r.strip()]
            true_robots = [r.strip() for r in truth.split(',') if r.strip()]
            
            reward = 0.0
            
            # Bonus for correct number selection  
            if 0 < len(pred_robots) <= max_robots:
                reward += 0.3
                
            # Overlap bonus
            if len(true_robots) > 0:
                overlap = len(set(pred_robots).intersection(set(true_robots)))
                reward += 0.5 * (overlap / len(set(true_robots)))
                
            # Efficiency penalty for too many robots
            if len(pred_robots) > max_robots:
                reward -= 0.2 * (len(pred_robots) - max_robots)
                
            rewards.append(max(0.0, reward))
            
        return rewards
    
    def _diversity_bonus_reward(self, responses: List[str]) -> List[float]:
        """Bonus for selecting diverse robot types"""
        rewards = []
        
        for response in responses:
            extracted_robots = ReasoningFormatValidator.extract_answer(response)
            pred_robots = [r.strip() for r in extracted_robots.split(',') if r.strip()]
            
            # Check diversity (different robot types)
            unique_robots = set(pred_robots)
            if len(unique_robots) == len(pred_robots) and len(pred_robots) > 1:
                rewards.append(0.1)  # Small diversity bonus
            else:
                rewards.append(0.0)
                
        return rewards


def create_grpo_robot_reasoning_integration(model, robot_data_dir: str, config: Dict = None) -> GRPORobotReasoningIntegration:
    """Factory function to create GRPO robot reasoning integration"""
    return GRPORobotReasoningIntegration(model, robot_data_dir, config)
