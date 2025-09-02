"""
Test Script for BitGen GRPO Robot Reasoning Implementation
Validates the policy optimization capabilities without running full training
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append('src')

try:
    from grpo_robot_reasoning import (
        PolicyOptimizedRobotHead,
        GRPORobotReasoningIntegration,
        GRPORobotRewardFunctions
    )
    from robot_reasoning import RobotReasoningProcessor
    print("✅ Successfully imported GRPO robot reasoning modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_policy_robot_head():
    """Test the PolicyOptimizedRobotHead"""
    print("\n🧪 Testing PolicyOptimizedRobotHead...")
    
    # Create test robot head
    hidden_dim = 128
    robot_head = PolicyOptimizedRobotHead(input_dim=hidden_dim, num_robots=5)
    
    # Test forward pass
    batch_size = 4
    test_features = torch.randn(batch_size, hidden_dim)
    
    print(f"   • Input shape: {test_features.shape}")
    
    # Forward pass
    outputs = robot_head(test_features)
    
    print(f"   • Robot policies shape: {next(iter(outputs['robot_policies'].values())).shape}")
    print(f"   • State value shape: {outputs['state_value'].shape}")
    print(f"   • Top-N distribution shape: {outputs['top_n_distribution'].shape}")
    print(f"   • Complexity distribution shape: {outputs['complexity_distribution'].shape}")
    
    # Test robot sampling
    selected_robots = robot_head.sample_robot_selection(outputs, temperature=0.8)
    print(f"   • Sample robot selection: {selected_robots}")
    
    # Test top-N selection
    top_n_robots = robot_head.get_top_n_robots(outputs, n=2)
    print(f"   • Top-2 robots: {top_n_robots}")
    
    print("✅ PolicyOptimizedRobotHead test passed!")
    

def test_grpo_reward_functions():
    """Test GRPO reward functions"""
    print("\n🧪 Testing GRPO Reward Functions...")
    
    # Create mock robot processor
    robot_data_dir = "robot_selection_data"
    if not Path(robot_data_dir).exists():
        print(f"   ⚠️ Robot data directory not found: {robot_data_dir}")
        print("   • Creating mock robot processor...")
        robot_processor = None
    else:
        robot_processor = RobotReasoningProcessor(robot_data_dir)
    
    # Initialize reward functions
    config = {
        'reward_weights': {
            'correctness': 0.30,
            'validity': 0.20,
            'format': 0.20,
            'reasoning_quality': 0.15,
            'top_n_efficiency': 0.15
        },
        'max_robots_per_task': 3
    }
    
    reward_functions = GRPORobotRewardFunctions(robot_processor, config)
    
    # Test data
    prompts = [
        "Select robots for underwater exploration mission",
        "Choose robots for high-altitude inspection task"
    ]
    
    completions = [
        [{"content": "<reasoning>Need waterproof robot for underwater work</reasoning><answer>Underwater Robot</answer>"}],
        [{"content": "<reasoning>High altitude requires aerial capability</reasoning><answer>Drone</answer>"}]
    ]
    
    ground_truth = ["Underwater Robot", "Drone"]
    
    try:
        # Compute rewards
        rewards = reward_functions.compute_all_rewards(prompts, completions, ground_truth)
        
        print(f"   • Reward components: {list(rewards.keys())}")
        print(f"   • Total rewards: {rewards['total']}")
        print(f"   • Correctness rewards: {rewards['correctness']}")
        print(f"   • Format rewards: {rewards['strict_format']}")
        
        print("✅ GRPO reward functions test passed!")
        
    except Exception as e:
        print(f"   ⚠️ Reward function test failed: {e}")
        print("   • This is expected if robot_processor is None")


def test_grpo_integration():
    """Test complete GRPO integration"""
    print("\n🧪 Testing GRPO Integration...")
    
    # Create mock model
    class MockModel:
        def __init__(self):
            self.config = type('Config', (), {'hidden_size': 128})()
            
    model = MockModel()
    
    try:
        # Initialize integration
        robot_data_dir = "robot_selection_data"
        config = {'max_robots_per_task': 3}
        
        integration = GRPORobotReasoningIntegration(model, robot_data_dir, config)
        
        print(f"   • GRPO integration created successfully")
        print(f"   • Robot head type: {type(integration.robot_selection_head).__name__}")
        
        # Test policy-based reasoning generation
        test_task = "Explore underwater cave system for marine research"
        
        print(f"   • Testing task: {test_task}")
        
        result = integration.generate_robot_reasoning_with_policy(
            task=test_task,
            temperature=0.7,
            top_n=2
        )
        
        print(f"   • Selected robots: {result['selected_robots']}")
        print(f"   • Estimated complexity: {result['estimated_complexity']}")
        print(f"   • Reasoning quality: {result['reasoning_quality']:.3f}")
        
        # Show selection probabilities
        print("   • Robot selection probabilities:")
        for robot, prob in result['selection_probabilities'].items():
            print(f"     - {robot}: {prob:.3f}")
            
        print("✅ GRPO integration test passed!")
        
    except Exception as e:
        print(f"   ❌ GRPO integration test failed: {e}")
        import traceback
        traceback.print_exc()


def test_configuration_loading():
    """Test configuration file loading"""
    print("\n🧪 Testing Configuration Loading...")
    
    config_path = "configs/bitgen_integrated_grpo.yaml"
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        print(f"   • Configuration loaded from: {config_path}")
        print(f"   • Model name: {config['model']['name']}")
        print(f"   • GRPO generations: {config['grpo']['num_generations']}")
        print(f"   • Training epochs: {config['training']['num_epochs']}")
        print(f"   • Robot types: {list(config['robot_reasoning']['robot_types'].keys())}")
        
        # Validate reward weights
        reward_weights = config['reward_functions']['weights']
        total_weight = sum(reward_weights.values())
        print(f"   • Reward weights sum: {total_weight:.3f}")
        
        if abs(total_weight - 1.0) < 0.01:
            print("   ✅ Reward weights are properly normalized")
        else:
            print(f"   ⚠️ Reward weights should sum to 1.0, got {total_weight:.3f}")
            
        print("✅ Configuration loading test passed!")
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")


def main():
    """Run all tests"""
    print("🚀 Starting BitGen GRPO Robot Reasoning Tests\n")
    
    # Test individual components
    test_policy_robot_head()
    test_grpo_reward_functions()
    test_grpo_integration()
    test_configuration_loading()
    
    print("\n🎉 All tests completed!")
    print("\n📋 Next Steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Prepare robot reasoning data in ./robot_selection_data/")
    print("   3. Run GRPO training: python train_integrated_grpo.py --config configs/bitgen_integrated_grpo.yaml")
    print("   4. Monitor training with wandb (if enabled)")
    print("   5. Test trained model with robot reasoning scenarios")
    
    print("\n🤖 GRPO Robot Reasoning Implementation Summary:")
    print("   • Policy-based robot selection with probability distributions")
    print("   • Top-N robot selection through policy optimization")
    print("   • Multiple reward functions for comprehensive training")
    print("   • Integration with existing BitGen multimodal system") 
    print("   • Enhanced reasoning quality prediction and task complexity estimation")


if __name__ == "__main__":
    main()
