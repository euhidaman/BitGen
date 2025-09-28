"""
HuggingFace Integration for BitGen
Automatic model pushing to HuggingFace Hub with rolling checkpoint management (keeps last 10 iterations only)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import torch
from huggingface_hub import HfApi, Repository, login, create_repo, upload_file, delete_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
import tempfile
import shutil
import re

class HuggingFaceIntegration:
    """Handle HuggingFace Hub integration for BitGen models with rolling checkpoint management"""

    def __init__(self,
                 repo_name: str,
                 organization: str = None,
                 token: Optional[str] = None,
                 private: bool = False,
                 max_checkpoints: int = 10):
        """
        Initialize HuggingFace integration with rolling checkpoint management

        Args:
            repo_name: Name of the repository (e.g., "bitgen-tiny-v1")
            organization: HF organization/username (if None, uses authenticated user)
            token: HF token (if None, looks for HF_TOKEN env var)
            private: Whether to create private repository
            max_checkpoints: Maximum number of checkpoints to keep (default: 10)
        """
        self.repo_name = repo_name
        self.organization = organization
        self.token = token or os.getenv("HF_TOKEN")
        self.private = private
        self.max_checkpoints = max_checkpoints

        # Initialize HF API
        self.api = HfApi()

        # Login if token provided
        if self.token:
            login(token=self.token)

        # Repository info
        if self.organization:
            self.repo_id = f"{self.organization}/{self.repo_name}"
        else:
            self.repo_id = self.repo_name

        self.logger = logging.getLogger(__name__)

        # Create repository if it doesn't exist
        self._create_repository_if_needed()

    def _create_repository_if_needed(self):
        """Create HuggingFace repository if it doesn't exist"""
        try:
            # Check if repo exists
            self.api.repo_info(self.repo_id)
            self.logger.info(f"Repository {self.repo_id} already exists")
        except:
            # Create repository
            try:
                create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    private=self.private,
                    repo_type="model"
                )
                self.logger.info(f"Created repository {self.repo_id}")
            except Exception as e:
                self.logger.error(f"Failed to create repository {self.repo_id}: {e}")
                raise

    def push_model_checkpoint(self,
                            model,
                            config,
                            tokenizer=None,
                            epoch: int = 0,
                            step: int = 0,
                            metrics: Dict = None,
                            commit_message: Optional[str] = None):
        """
        Push model checkpoint to HuggingFace Hub with rolling checkpoint management

        Args:
            model: BitGen model instance
            config: Model configuration
            tokenizer: Tokenizer (optional)
            epoch: Current epoch number
            step: Current training step
            metrics: Training metrics to include
            commit_message: Custom commit message
        """

        if commit_message is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"Checkpoint - Epoch {epoch}, Step {step} - {timestamp}"

        # Create temporary directory for model files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Create versioned checkpoint filename
                checkpoint_name = f"pytorch_model_epoch-{epoch:03d}_step-{step:06d}.bin"
                model_path = temp_path / checkpoint_name
                torch.save(model.state_dict(), model_path)

                # Create and save HuggingFace compatible config with version info
                hf_config = self._create_hf_config(config, epoch, step, metrics)
                config_path = temp_path / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(hf_config, f, indent=2)

                # Create model card with checkpoint info
                model_card_path = temp_path / "README.md"
                self._create_model_card(model_card_path, config, epoch, step, metrics)

                # Save tokenizer if provided
                if tokenizer:
                    tokenizer_config = {
                        "vocab_size": getattr(config, 'vocab_size', 8192),
                        "model_max_length": getattr(config, 'max_seq_len', 256),
                        "tokenizer_class": "BitGenTokenizer",
                        "checkpoint_epoch": epoch,
                        "checkpoint_step": step
                    }

                    tokenizer_config_path = temp_path / "tokenizer_config.json"
                    with open(tokenizer_config_path, 'w') as f:
                        json.dump(tokenizer_config, f, indent=2)

                # Create training info file with checkpoint details
                training_info_path = temp_path / f"training_info_epoch-{epoch:03d}_step-{step:06d}.json"
                training_info = {
                    "epoch": epoch,
                    "step": step,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics or {},
                    "model_architecture": "BitGen",
                    "framework": "PyTorch",
                    "checkpoint_id": f"epoch-{epoch:03d}_step-{step:06d}",
                    "is_latest": True  # Will be updated when newer checkpoints are added
                }

                with open(training_info_path, 'w') as f:
                    json.dump(training_info, f, indent=2)

                # Upload all files to repository
                self.logger.info(f"🔄 Pushing checkpoint {epoch:03d}-{step:06d} to {self.repo_id}")

                # Upload versioned checkpoint
                upload_file(
                    path_or_fileobj=str(model_path),
                    path_in_repo=checkpoint_name,
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=f"{commit_message} - Model weights"
                )

                # Upload main model file (latest)
                upload_file(
                    path_or_fileobj=str(model_path),
                    path_in_repo="pytorch_model.bin",
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=f"{commit_message} - Latest model"
                )

                # Upload other files
                for file_path in temp_path.glob("*"):
                    if file_path.is_file() and file_path.name != checkpoint_name:
                        upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=file_path.name,
                            repo_id=self.repo_id,
                            token=self.token,
                            commit_message=f"{commit_message} - {file_path.name}"
                        )

                self.logger.info(f"✅ Successfully pushed checkpoint {epoch:03d}-{step:06d} to HuggingFace Hub!")

                # Manage rolling checkpoints - keep only last N versioned checkpoints
                self._manage_rolling_checkpoints(epoch, step)

                # Return repository URL
                return f"https://huggingface.co/{self.repo_id}"

            except Exception as e:
                self.logger.error(f"Failed to push checkpoint to HuggingFace Hub: {e}")
                raise

    def _manage_rolling_checkpoints(self, current_epoch: int, current_step: int):
        """Manage rolling checkpoints - keep only latest N versioned checkpoints"""

        try:
            # List all files in the repository
            files = self.api.list_repo_files(self.repo_id, token=self.token)

            # Filter checkpoint files (versioned model files)
            checkpoint_pattern = re.compile(r"^pytorch_model_epoch-(\d+)_step-(\d+)\.bin$")
            checkpoint_files = []

            for f in files:
                match = checkpoint_pattern.match(f)
                if match:
                    epoch = int(match.group(1))
                    step = int(match.group(2))
                    checkpoint_files.append((epoch, step, f))

            # Sort by epoch then step (latest last)
            checkpoint_files.sort(key=lambda x: (x[0], x[1]))

            # Keep only the latest N checkpoints
            if len(checkpoint_files) > self.max_checkpoints:
                files_to_delete = checkpoint_files[:-self.max_checkpoints]

                self.logger.info(f"🗑️ Cleaning up old checkpoints. Keeping latest {self.max_checkpoints} out of {len(checkpoint_files)}")

                # Delete old checkpoints and their associated training info files
                for epoch, step, file_name in files_to_delete:
                    try:
                        # Delete checkpoint file
                        self.logger.info(f"   Deleting checkpoint: {file_name}")
                        delete_file(
                            path_in_repo=file_name,
                            repo_id=self.repo_id,
                            token=self.token,
                            commit_message=f"Cleanup: Remove old checkpoint epoch-{epoch:03d}_step-{step:06d}"
                        )

                        # Delete associated training info file
                        training_info_file = f"training_info_epoch-{epoch:03d}_step-{step:06d}.json"
                        if training_info_file in files:
                            self.logger.info(f"   Deleting training info: {training_info_file}")
                            delete_file(
                                path_in_repo=training_info_file,
                                repo_id=self.repo_id,
                                token=self.token,
                                commit_message=f"Cleanup: Remove old training info epoch-{epoch:03d}_step-{step:06d}"
                            )

                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file_name}: {e}")

                self.logger.info(f"✅ Rolling checkpoint cleanup complete. Kept latest {self.max_checkpoints} checkpoints.")
            else:
                self.logger.info(f"📁 Checkpoint count ({len(checkpoint_files)}) within limit ({self.max_checkpoints}). No cleanup needed.")

        except Exception as e:
            self.logger.error(f"Failed to manage rolling checkpoints: {e}")
            # Don't raise - checkpoint cleanup failure shouldn't stop training

    def _create_hf_config(self, config, epoch: int, step: int, metrics: Dict) -> Dict:
        """Create HuggingFace compatible configuration"""

        # Extract config attributes
        config_dict = config.__dict__ if hasattr(config, '__dict__') else config

        hf_config = {
            "model_type": "bitgen",
            "architectures": ["BitGenModel"],

            # Model architecture
            "vocab_size": config_dict.get('vocab_size', 8192),
            "hidden_size": config_dict.get('embed_dim', 128),
            "num_hidden_layers": config_dict.get('num_layers', 4),
            "num_attention_heads": config_dict.get('num_heads', 8),
            "intermediate_size": config_dict.get('ffn_dim', 256),
            "max_position_embeddings": config_dict.get('max_seq_len', 256),

            # BitGen specific
            "memory_size": config_dict.get('memory_size', 64),
            "memory_dim": config_dict.get('memory_dim', 128),
            "attention_sinks": config_dict.get('attention_sinks', 4),
            "window_size": config_dict.get('window_size', 128),
            "reasoning_dim": config_dict.get('reasoning_dim', 64),
            "max_reasoning_steps": config_dict.get('max_reasoning_steps', 8),
            "num_robots": config_dict.get('num_robots', 16),
            "quantization_bits": config_dict.get('quantization_bits', 1.58),

            # Training info
            "training_epoch": epoch,
            "training_step": step,
            "training_metrics": metrics or {},

            # Framework info
            "torch_dtype": "float32",
            "transformers_version": "4.35.0",
            "auto_map": {
                "AutoModel": "modeling_bitgen.BitGenModel",
                "AutoConfig": "configuration_bitgen.BitGenConfig"
            }
        }

        return hf_config

    def _create_model_card(self, card_path: Path, config, epoch: int, step: int, metrics: Dict):
        """Create model card with training information"""

        config_dict = config.__dict__ if hasattr(config, '__dict__') else config
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        model_card_content = f"""---
language: en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- bitgen
- multimodal
- robotics
- episodic-memory
- quantized
- embedded
---

# BitGen: Advanced Tiny Language Model

This is a BitGen model trained for embedded deployment with comprehensive multimodal capabilities.

## Model Description

BitGen is an advanced tiny language model specifically designed for resource-constrained environments, integrating:

- **Larimar Episodic Memory**: Core memory architecture for storing and retrieving experiences
- **BitNet 1.58-bit Quantization**: Ultra-efficient quantization for deployment optimization  
- **FIBER Cross-Modal Fusion**: Vision-language understanding with image-text association
- **Attention Sinks**: Memory-efficient attention mechanism for long sequences
- **Tiny-R1 Reasoning**: DeepSeek-R1 inspired reasoning capabilities
- **Robot Selection**: Intelligent robot selection based on task requirements

## Architecture

- **Model Size**: {config_dict.get('embed_dim', 128)}D x {config_dict.get('num_layers', 4)}L
- **Parameters**: ~{self._estimate_parameters(config_dict):.1f}M parameters
- **Vocabulary Size**: {config_dict.get('vocab_size', 8192):,} tokens
- **Max Sequence Length**: {config_dict.get('max_seq_len', 256)} tokens
- **Memory Slots**: {config_dict.get('memory_size', 64)} episodic memory slots
- **Quantization**: {config_dict.get('quantization_bits', 1.58)}-bit weights

## Training Information

- **Current Epoch**: {epoch}
- **Current Step**: {step}
- **Training Date**: {timestamp}
- **Framework**: PyTorch with custom BitGen architecture

### Training Metrics
"""

        if metrics:
            model_card_content += "\n"
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    model_card_content += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
                else:
                    model_card_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"

        model_card_content += f"""
## Usage

```python
from transformers import AutoModel, AutoConfig
import torch

# Load model
config = AutoConfig.from_pretrained("{self.repo_id}")
model = AutoModel.from_pretrained("{self.repo_id}")

# Example usage
input_ids = torch.randint(0, config.vocab_size, (1, 32))
outputs = model(input_ids)
```

## Capabilities

1. **Multimodal Understanding**: Processes both text and visual inputs
2. **Episodic Memory**: Stores and retrieves relevant experiences  
3. **Reasoning**: Multi-step logical reasoning capabilities
4. **Robot Selection**: Task-appropriate robot selection
5. **Embedded Optimization**: Designed for resource-constrained deployment

## Deployment

This model is optimized for:
- Raspberry Pi and embedded systems
- Edge computing devices  
- Mobile and IoT applications
- Real-time inference requirements

## Citation

```bibtex
@misc{{bitgen2025,
  title={{BitGen: Advanced Tiny Language Model for Embedded Multimodal Applications}},
  year={{2025}},
  url={{https://huggingface.co/{self.repo_id}}}
}}
```

## Model Card Authors

Generated automatically during training at epoch {epoch}.
"""

        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(model_card_content)

    def _estimate_parameters(self, config_dict: Dict) -> float:
        """Estimate model parameters in millions"""
        embed_dim = config_dict.get('embed_dim', 128)
        num_layers = config_dict.get('num_layers', 4)
        vocab_size = config_dict.get('vocab_size', 8192)
        ffn_dim = config_dict.get('ffn_dim', 256)

        # Rough parameter estimation
        embedding_params = vocab_size * embed_dim
        layer_params = num_layers * (
            4 * embed_dim * embed_dim +  # Attention projections
            2 * embed_dim * ffn_dim +     # FFN
            embed_dim * 2                 # Layer norms
        )
        output_params = embed_dim * vocab_size

        total_params = embedding_params + layer_params + output_params
        return total_params / 1e6

    def create_epoch_tag(self, epoch: int) -> str:
        """Create tag for specific epoch"""
        return f"epoch-{epoch}"

    def push_final_model(self, model, config, tokenizer=None, metrics: Dict = None):
        """Push final trained model with 'latest' tag"""
        return self.push_model_checkpoint(
            model=model,
            config=config,
            tokenizer=tokenizer,
            epoch=-1,  # Indicates final model
            metrics=metrics,
            commit_message="Final trained model - Latest version"
        )

def setup_huggingface_integration(model_name: str,
                                organization: str = None,
                                token: str = None,
                                private: bool = False) -> HuggingFaceIntegration:
    """
    Setup HuggingFace integration for BitGen model

    Args:
        model_name: Name for the model repository
        organization: HF organization (optional)
        token: HF authentication token
        private: Whether to create private repository

    Returns:
        HuggingFaceIntegration instance
    """

    if not token:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN environment variable or pass token parameter."
            )

    return HuggingFaceIntegration(
        repo_name=model_name,
        organization=organization,
        token=token,
        private=private
    )
