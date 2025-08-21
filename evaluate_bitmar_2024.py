"""
BitMar Evaluation Script for 2024 Pipeline
Evaluates BitMar models on multimodal tasks only using the 2024 evaluation pipeline
"""

import os
import sys
import argparse
import logging
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_checkpoint(checkpoint_path: str, device: str = 'cuda:0'):
    """Load BitMar model from checkpoint"""
    try:
        # Add src to path for model loading
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.model import create_bitmar_model

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract config and model state
        config = checkpoint.get('config', {})
        model_state = checkpoint['model_state_dict']

        # Create model with config
        model = create_bitmar_model(config['model'])
        model.load_state_dict(model_state)

        # Move to device
        device = torch.device(device)
        model = model.to(device)
        model.eval()

        logger.info(f"✅ Model loaded successfully on {device}")
        return model, config

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def setup_evaluation_environment(pipeline_path: Path, eval_data_path: Path):
    """Setup evaluation environment for 2024 pipeline"""
    try:
        # Change to pipeline directory
        original_cwd = os.getcwd()
        os.chdir(pipeline_path)

        # Check if evaluation_data exists, if not create symlink
        eval_data_link = pipeline_path / "evaluation_data"

        if not eval_data_link.exists():
            logger.info(f"Creating evaluation data link: {eval_data_link} -> {eval_data_path}")
            try:
                eval_data_link.symlink_to(eval_data_path.resolve(), target_is_directory=True)
            except OSError:
                # Fallback to copying on Windows
                shutil.copytree(eval_data_path, eval_data_link)
                logger.info(f"Copied evaluation data to: {eval_data_link}")

        return original_cwd

    except Exception as e:
        logger.error(f"Failed to setup evaluation environment: {e}")
        raise


def run_multimodal_evaluations(model_path: str, eval_type: str = "fast", output_dir: str = "results_2024"):
    """Run multimodal evaluations (VQA, Winoground, etc.) for 2024 pipeline"""
    logger.info(f"🖼️ Running multimodal evaluations ({eval_type}) - 2024 Pipeline...")

    results = {}

    try:
        # Convert BitMar checkpoint to HuggingFace format first
        hf_model_dir = f"hf_model_temp_2024_{eval_type}"
        try:
            sys.path.append(str(Path(__file__).parent))
            from bitmar_hf_adapter import save_hf_compatible_model
            hf_model_path = save_hf_compatible_model(model_path, hf_model_dir)
            logger.info(f"✅ Created HuggingFace compatible model for 2024 evaluation at: {hf_model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create HF adapter for 2024 evaluation: {e}")
            results['multimodal'] = {"error": f"HF adapter failed: {e}"}
            return results

        # Check for multimodal evaluation script in 2024 pipeline
        multimodal_scripts = [
            "eval_multimodal.sh",
            "run_multimodal_eval.py",
            "scripts/eval_multimodal.py"
        ]

        eval_script = None
        for script in multimodal_scripts:
            if Path(script).exists():
                eval_script = script
                break

        if not eval_script:
            logger.warning("⚠️ No multimodal evaluation script found in 2024 pipeline")
            # Try to run individual multimodal tasks
            individual_results = run_individual_multimodal_tasks(hf_model_path, eval_type, output_dir)
            # Cleanup temp HF model
            if Path(hf_model_dir).exists():
                shutil.rmtree(hf_model_dir)
            return individual_results

        # Run multimodal evaluation with HuggingFace model
        if eval_script.endswith('.py'):
            cmd = [
                sys.executable, eval_script,
                "--model_path", hf_model_path,  # Use HF model path
                "--eval_type", eval_type,
                "--output_dir", f"{output_dir}/multimodal_results"
            ]
            # Use shell=False for Python scripts with proper argument list
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        else:
            cmd = [
                "bash", eval_script,
                hf_model_path,  # Use HF model path for bash scripts
                eval_type,
                f"{output_dir}/multimodal_results"
            ]
            # Use normal subprocess for bash scripts
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        logger.info(f"Running command: {' '.join(cmd)}")

        if result.returncode == 0:
            logger.info("✅ Multimodal evaluation completed successfully")

            # Try to parse results
            results_file = Path(f"{output_dir}/multimodal_results/results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results['multimodal'] = json.load(f)
        else:
            logger.warning(f"⚠️ Multimodal evaluation failed: {result.stderr}")
            results['multimodal'] = {"error": result.stderr}

        # Cleanup temp HF model
        if Path(hf_model_dir).exists():
            shutil.rmtree(hf_model_dir)

    except Exception as e:
        logger.error(f"❌ Multimodal evaluation error: {e}")
        results['multimodal'] = {"error": str(e)}
        # Cleanup temp HF model
        if 'hf_model_dir' in locals() and Path(hf_model_dir).exists():
            shutil.rmtree(hf_model_dir)

    return results


def run_individual_multimodal_tasks(hf_model_path: str, eval_type: str, output_dir: str):
    """Run individual multimodal tasks if main script not found"""
    logger.info("🔧 Running individual multimodal tasks...")

    results = {}

    # Define multimodal tasks for 2024 pipeline
    multimodal_tasks = [
        {
            "name": "vqa",
            "script": "eval_vqa.py",
            "data_dir": "evaluation_data/full_eval/vqa_filtered" if eval_type == "full" else "evaluation_data/fast_eval/vqa_fast"
        },
        {
            "name": "winoground",
            "script": "eval_winoground.py",
            "data_dir": "evaluation_data/full_eval/winoground_filtered" if eval_type == "full" else "evaluation_data/fast_eval/winoground_fast"
        }
    ]

    for task in multimodal_tasks:
        try:
            # Check if data directory exists
            data_path = Path(task["data_dir"])
            if not data_path.exists():
                logger.warning(f"⚠️ Data not found for {task['name']}: {data_path}")
                continue

            # Check if script exists
            script_paths = [
                task["script"],
                f"scripts/{task['script']}",
                f"evaluation/{task['script']}"
            ]

            script_path = None
            for path in script_paths:
                if Path(path).exists():
                    script_path = path
                    break

            if not script_path:
                logger.warning(f"⚠️ Script not found for {task['name']}")
                continue

            # Run task with HuggingFace model path
            cmd = [
                sys.executable, script_path,
                "--model_path", hf_model_path,  # Use HF model path
                "--data_dir", str(data_path),
                "--output_dir", f"{output_dir}/{task['name']}_results"
            ]

            logger.info(f"Running {task['name']}: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            if result.returncode == 0:
                logger.info(f"✅ {task['name']} evaluation completed")

                # Try to parse results
                task_results_file = Path(f"{output_dir}/{task['name']}_results/results.json")
                if task_results_file.exists():
                    with open(task_results_file, 'r') as f:
                        results[task['name']] = json.load(f)
                else:
                    results[task['name']] = {"status": "completed", "details": "No results file found"}
            else:
                logger.warning(f"⚠️ {task['name']} evaluation failed: {result.stderr}")
                results[task['name']] = {"error": result.stderr}

        except Exception as e:
            logger.error(f"❌ {task['name']} evaluation error: {e}")
            results[task['name']] = {"error": str(e)}

    return results


def run_ewok_evaluation(model_path: str, eval_type: str, output_dir: str):
    """Run EWOK evaluation if available"""
    logger.info("🦉 Running EWOK evaluation...")

    results = {}

    try:
        # Check for EWOK script
        ewok_script = "eval_ewok.sh"

        if not Path(ewok_script).exists():
            logger.warning("⚠️ EWOK evaluation script not found")
            return results

        # Determine EWOK data directory
        if eval_type == "fast":
            ewok_data = "evaluation_data/fast_eval/ewok_fast"
        else:
            ewok_data = "evaluation_data/full_eval/ewok"

        if not Path(ewok_data).exists():
            logger.warning(f"⚠️ EWOK data not found: {ewok_data}")
            return results

        # Run EWOK evaluation
        cmd = [
            "bash", ewok_script,
            "--model_path", model_path,
            "--data_dir", ewok_data,
            "--output_dir", f"{output_dir}/ewok_results"
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            logger.info("✅ EWOK evaluation completed successfully")

            # Try to parse results
            results_file = Path(f"{output_dir}/ewok_results/results.json")
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results['ewok'] = json.load(f)
        else:
            logger.warning(f"⚠️ EWOK evaluation failed: {result.stderr}")
            results['ewok'] = {"error": result.stderr}

    except Exception as e:
        logger.error(f"❌ EWOK evaluation error: {e}")
        results['ewok'] = {"error": str(e)}

    return results


def evaluate_bitmar_2024(
    model_path: str,
    eval_type: str = "fast",
    evaluation_pipeline_path: str = "../evaluation-pipeline-2024",
    output_dir: str = "results_2024"
):
    """
    Evaluate BitMar model using 2024 pipeline (multimodal tasks only)

    Args:
        model_path: Path to model checkpoint
        eval_type: "fast" or "full" evaluation
        evaluation_pipeline_path: Path to 2024 evaluation pipeline
        output_dir: Output directory for results
    """
    logger.info("🚀 Starting BitMar 2024 Pipeline Evaluation (Multimodal Only)")
    logger.info(f"📁 Model: {model_path}")
    logger.info(f"🎯 Evaluation type: {eval_type}")
    logger.info(f"📂 Pipeline: {evaluation_pipeline_path}")
    logger.info(f"💾 Output: {output_dir}")

    # Convert paths to absolute paths BEFORE changing directory
    pipeline_path = Path(evaluation_pipeline_path).resolve()
    model_path = Path(model_path).resolve()

    # Convert output path to absolute path before any directory changes
    if Path(output_dir).is_absolute():
        output_path = Path(output_dir)
    else:
        # Make relative path absolute based on current working directory
        output_path = Path.cwd() / output_dir
    output_path = output_path.resolve()

    logger.info(f"📂 Resolved output path: {output_path}")

    # Validate paths
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline path not found: {pipeline_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Create output directory with absolute path
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Created output directory: {output_path}")

    # Find evaluation data
    eval_data_paths = [
        pipeline_path / "evaluation_data",
        pipeline_path.parent / "evaluation_data"
    ]

    eval_data_path = None
    for path in eval_data_paths:
        if path.exists():
            eval_data_path = path
            break

    if not eval_data_path:
        raise FileNotFoundError("Evaluation data not found. Run download_evaluation_data.py first.")

    logger.info(f"📊 Using evaluation data: {eval_data_path}")

    # Setup evaluation environment
    original_cwd = setup_evaluation_environment(pipeline_path, eval_data_path)

    try:
        all_results = {}

        # Run multimodal evaluations (main focus for 2024 pipeline)
        multimodal_results = run_multimodal_evaluations(str(model_path), eval_type, str(output_path))
        all_results.update(multimodal_results)

        # Run EWOK if available
        ewok_results = run_ewok_evaluation(str(model_path), eval_type, str(output_path))
        all_results.update(ewok_results)

        # Save combined results
        results_file = output_path / "combined_results_2024.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"📊 Results saved to: {results_file}")

        # Log summary
        logger.info("📈 Evaluation Summary (2024 Pipeline - Multimodal Only):")
        for task_type, results in all_results.items():
            if isinstance(results, dict) and "error" not in results:
                logger.info(f"  ✅ {task_type}: Success")
            else:
                logger.info(f"  ❌ {task_type}: Failed")

        return all_results

    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate BitMar using 2024 pipeline (multimodal only)")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--eval_type", type=str, choices=["fast", "full"], default="fast",
                       help="Type of evaluation to run")
    parser.add_argument("--evaluation_pipeline_path", type=str, default="../evaluation-pipeline-2024",
                       help="Path to 2024 evaluation pipeline")
    parser.add_argument("--output_dir", type=str, default="results_2024",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for evaluation")

    args = parser.parse_args()

    try:
        results = evaluate_bitmar_2024(
            model_path=args.model_path,
            eval_type=args.eval_type,
            evaluation_pipeline_path=args.evaluation_pipeline_path,
            output_dir=args.output_dir
        )

        logger.info("🎉 2024 Pipeline evaluation (multimodal only) completed successfully!")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
