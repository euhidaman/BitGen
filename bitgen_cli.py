#!/usr/bin/env python3
"""
Complete Training and Deployment Script for BitGen
One-command solution for training and deploying BitGen models
"""

import argparse
import json
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="BitGen: Advanced Tiny Language Model for Embedded Systems")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download data command
    download_parser = subparsers.add_parser('download', help='Download COCO dataset')
    download_parser.add_argument('--output_dir', type=str, default='data/coco',
                                help='Output directory for dataset')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train BitGen model')
    train_parser.add_argument('--coco_data', type=str, required=True,
                             help='Path to COCO dataset')
    train_parser.add_argument('--robot_data', type=str,
                             help='Path to robot selection dataset')
    train_parser.add_argument('--model_size', type=str, default='tiny',
                             choices=['nano', 'tiny', 'small'])
    train_parser.add_argument('--target_device', type=str, default='cortex-m4',
                             choices=['cortex-m0', 'cortex-m4', 'cortex-m7'])
    train_parser.add_argument('--batch_size', type=int, default=4)
    train_parser.add_argument('--learning_rate', type=float, default=1e-4)
    train_parser.add_argument('--num_epochs', type=int, default=10)
    train_parser.add_argument('--output_dir', type=str, default='checkpoints')
    train_parser.add_argument('--use_wandb', action='store_true')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate BitGen model')
    eval_parser.add_argument('--model_path', type=str, required=True)
    eval_parser.add_argument('--coco_test', type=str, required=True)
    eval_parser.add_argument('--robot_test', type=str, required=True)
    eval_parser.add_argument('--output_dir', type=str, default='evaluation_results')

    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model for embedded')
    deploy_parser.add_argument('--model_path', type=str, required=True)
    deploy_parser.add_argument('--output_dir', type=str, required=True)
    deploy_parser.add_argument('--target_device', type=str, default='cortex-m4',
                              choices=['cortex-m0', 'cortex-m4', 'cortex-m7'])

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
    interactive_parser.add_argument('--model_path', type=str, required=True)

    args = parser.parse_args()

    if args.command == 'download':
        download_coco_data(args.output_dir)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'deploy':
        deploy_model(args)
    elif args.command == 'interactive':
        interactive_mode(args.model_path)
    else:
        parser.print_help()

def download_coco_data(output_dir: str):
    """Download COCO dataset"""
    print("Downloading COCO dataset...")

    # Import and run the download script
    try:
        import sys
        sys.path.append('.')
        from download_coco_dataset import COCODownloader

        downloader = COCODownloader(output_dir)
        success = downloader.download_from_kaggle()

        if success:
            downloader.process_dataset()
            print("✅ COCO dataset downloaded and processed successfully!")
        else:
            print("❌ Failed to download COCO dataset")

    except Exception as e:
        print(f"Error downloading dataset: {e}")

def train_model(args):
    """Train BitGen model"""
    print(f"Training BitGen {args.model_size} model for {args.target_device}...")

    try:
        from src import BitGen

        # Create BitGen instance
        bitgen = BitGen(model_size=args.model_size, target_device=args.target_device)

        # Train the model
        bitgen.train(
            coco_data_path=args.coco_data,
            robot_data_path=args.robot_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            use_wandb=args.use_wandb
        )

        print("✅ Training completed successfully!")

    except Exception as e:
        print(f"❌ Training failed: {e}")

def evaluate_model(args):
    """Evaluate BitGen model"""
    print("Evaluating BitGen model...")

    try:
        from src import quick_evaluate

        results = quick_evaluate(
            model_path=args.model_path,
            coco_test_path=args.coco_test,
            robot_test_path=args.robot_test,
            output_dir=args.output_dir
        )

        print(f"✅ Evaluation completed!")
        print(f"Overall Score: {results.get('overall_score', 0):.3f}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def deploy_model(args):
    """Deploy model for embedded"""
    print(f"Deploying model for {args.target_device}...")

    try:
        from src import create_embedded_deployment

        create_embedded_deployment(
            model_path=args.model_path,
            output_dir=args.output_dir,
            target_device=args.target_device
        )

        print("✅ Embedded deployment package created!")

    except Exception as e:
        print(f"❌ Deployment failed: {e}")

def interactive_mode(model_path: str):
    """Interactive mode for testing model"""
    print("Starting BitGen Interactive Mode...")
    print("Type 'quit' to exit, 'help' for commands")

    try:
        from src import BitGen

        # Load model
        bitgen = BitGen()
        bitgen.load_checkpoint(model_path)

        while True:
            try:
                user_input = input("\nBitGen> ").strip()

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    print_help()
                elif user_input.startswith('generate:'):
                    prompt = user_input[9:].strip()
                    response = bitgen.generate_text(prompt)
                    print(f"Generated: {response}")
                elif user_input.startswith('robot:'):
                    task = user_input[6:].strip()
                    result = bitgen.select_robot_for_task(task)
                    print(f"Selected Robot: {result['selected_robot']} (confidence: {result['confidence']:.3f})")
                elif user_input.startswith('image:'):
                    parts = user_input[6:].strip().split('|')
                    if len(parts) == 2:
                        image_path, text = parts[0].strip(), parts[1].strip()
                        try:
                            result = bitgen.process_image_and_text(image_path, text)
                            print("✅ Image processed successfully")
                        except Exception as e:
                            print(f"❌ Error processing image: {e}")
                    else:
                        print("Usage: image: <path>|<text>")
                else:
                    print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Goodbye!")

    except Exception as e:
        print(f"❌ Failed to start interactive mode: {e}")

def print_help():
    """Print help for interactive mode"""
    help_text = """
Available commands:
  generate: <text>           - Generate text from prompt
  robot: <task_description>  - Select robot for task
  image: <path>|<text>       - Process image with text
  help                       - Show this help
  quit                       - Exit interactive mode

Examples:
  generate: The robot should move to
  robot: Pick up a heavy box from the floor
  image: /path/to/image.jpg|Describe this scene
"""
    print(help_text)

if __name__ == "__main__":
    main()
