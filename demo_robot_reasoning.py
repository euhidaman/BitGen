"""
Demonstration Script for BitGen Robot Reasoning Capabilities
Shows deepseek-r1 style structured reasoning for robot selection tasks
"""

import sys
import torch
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_robot_reasoning():
    """Demonstrate robot reasoning capabilities"""
    print("🤖 BitGen Robot Reasoning Demo (deepseek-r1 Style)")
    print("=" * 60)

    try:
        # Import robot reasoning components
        from src.robot_reasoning import RobotReasoningProcessor, ReasoningFormatValidator, RobotSelectionRewardFunctions
        from src.robot_reasoning_dataset import RobotReasoningDataset
        from transformers import AutoTokenizer

        print("✅ Successfully imported robot reasoning modules")

        # Load robot data
        print("\n📊 Loading Robot Selection Data...")
        robot_data_dir = "D:/BabyLM/robot_selection_data/data"
        processor = RobotReasoningProcessor(robot_data_dir)

        print(f"✅ Loaded {len(processor.single_robot_data)} single-robot examples")
        print(f"✅ Loaded {len(processor.multi_robot_data)} multi-robot examples")
        print(f"✅ Available robots: {', '.join(processor.robot_types)}")

        # Show robot capabilities
        print("\n🤖 Robot Capabilities Summary:")
        for robot, caps in processor.robot_capabilities.items():
            print(f"  • {robot}:")
            print(f"    Capabilities: {', '.join(caps['capabilities'][:3])}...")
            print(f"    Environments: {', '.join(caps['environments'])}")

        # Demonstrate reasoning generation
        print("\n🧠 Generating Structured Reasoning Examples (deepseek-r1 Style):")
        print("-" * 50)

        # Test case 1: Underwater inspection
        task1 = "Inspect underwater pipes for leaks"
        expected_robot1 = "Underwater Robot"
        reasoning1 = processor._generate_structured_reasoning(task1, expected_robot1)

        print(f"\n🔍 Task 1: {task1}")
        print(f"Expected Robot: {expected_robot1}")
        print(f"Generated Reasoning:")
        print(reasoning1)

        # Format as XML (like deepseek-r1)
        xml_response1 = f"<reasoning>\n{reasoning1}\n</reasoning>\n<answer>\nSelected robot(s): {expected_robot1}\n</answer>"
        print(f"\n📝 XML-Formatted Response (deepseek-r1 Style):")
        print(xml_response1)

        # Validate format
        format_valid = ReasoningFormatValidator.validate_format(xml_response1)
        xml_score = ReasoningFormatValidator.count_xml_structure(xml_response1)
        extracted_reasoning = ReasoningFormatValidator.extract_reasoning(xml_response1)
        extracted_answer = ReasoningFormatValidator.extract_answer(xml_response1)

        print(f"\n✅ Format Validation:")
        print(f"  • Format Valid: {format_valid}")
        print(f"  • XML Structure Score: {xml_score:.3f}")
        print(f"  • Extracted Answer: {extracted_answer}")

        # Test case 2: Multi-robot coordination
        print("\n" + "="*60)

        if processor.multi_robot_data:
            multi_example = processor.multi_robot_data[0]
            task2 = multi_example['input']
            subtasks = multi_example.get('subtasks', [])

            if subtasks:
                selected_robots = list(set([subtask['assigned_robot'] for subtask in subtasks]))
                robot_output = ', '.join(selected_robots)

                print(f"\n🔍 Multi-Robot Task: {task2}")
                print(f"Expected Robots: {robot_output}")

                reasoning2 = processor._generate_multi_robot_reasoning(task2, subtasks)
                print(f"Generated Multi-Robot Reasoning:")
                print(reasoning2)

                xml_response2 = f"<reasoning>\n{reasoning2}\n</reasoning>\n<answer>\nSelected robot(s): {robot_output}\n</answer>"

                format_valid2 = ReasoningFormatValidator.validate_format(xml_response2)
                xml_score2 = ReasoningFormatValidator.count_xml_structure(xml_response2)

                print(f"\n✅ Multi-Robot Format Validation:")
                print(f"  • Format Valid: {format_valid2}")
                print(f"  • XML Structure Score: {xml_score2:.3f}")

        # Test reward functions (deepseek-r1 style)
        print("\n" + "="*60)
        print("\n🏆 Testing deepseek-r1 Style Reward Functions:")

        reward_functions = RobotSelectionRewardFunctions(processor)

        # Test completions
        test_completions = [xml_response1, xml_response2] if 'xml_response2' in locals() else [xml_response1]
        test_ground_truth = [expected_robot1, robot_output] if 'robot_output' in locals() else [expected_robot1]

        # Compute rewards
        correctness_rewards = reward_functions.robot_correctness_reward_func(test_completions, test_ground_truth)
        validity_rewards = reward_functions.robot_validity_reward_func(test_completions)
        format_rewards = reward_functions.soft_format_reward_func(test_completions)
        xml_rewards = reward_functions.xmlcount_reward_func(test_completions)
        quality_rewards = reward_functions.reasoning_quality_reward_func(test_completions)

        print(f"✅ Reward Computation Results:")
        print(f"  • Correctness Rewards: {correctness_rewards}")
        print(f"  • Validity Rewards: {validity_rewards}")
        print(f"  • Format Rewards: {format_rewards}")
        print(f"  • XML Structure Rewards: {xml_rewards}")
        print(f"  • Quality Rewards: {quality_rewards}")

        # Test dataset creation
        print("\n" + "="*60)
        print("\n📚 Testing Dataset Creation:")

        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = RobotReasoningDataset(
            robot_data_dir=robot_data_dir,
            tokenizer=tokenizer,
            max_length=512
        )

        print(f"✅ Dataset created with {len(dataset)} examples")

        # Test data loading
        sample = dataset[0]
        print(f"✅ Sample data shape:")
        print(f"  • Input IDs: {sample['input_ids'].shape}")
        print(f"  • Vision Features: {sample['vision_features'].shape}")
        print(f"  • Robot Labels: {sample['robot_labels']}")
        print(f"  • Task Type: {sample['task_type']}")

        print("\n🎉 Robot Reasoning Integration Test Completed Successfully!")
        print("✅ All components working correctly with deepseek-r1 style structured reasoning")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_robot_reasoning()
