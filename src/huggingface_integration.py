"""
HuggingFace Hub Integration for BitGen
Automatically push model checkpoints to HuggingFace Hub
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class HuggingFaceIntegration:
    """Handle HuggingFace Hub operations for BitGen models"""

    def __init__(self,
                 repo_name: str = "BitGen-Reasoning",
                 organization: Optional[str] = None,
                 private: bool = False,
                 stage: str = "stage1"):
        """
        Initialize HuggingFace integration for 2-stage training

        Args:
            repo_name: Repository name on HuggingFace Hub
            organization: Organization/user name (uses authenticated user if None)
            private: Whether to create private repository
            stage: Training stage ('stage1' or 'stage2')
        """
        # Add stage suffix to repo name
        self.repo_name = f"{repo_name}-{stage}"
        self.stage = stage
        self.private = private

        # Check if huggingface_hub is available
        try:
            from huggingface_hub import HfApi, create_repo, login, whoami
            self.hf_api = HfApi()
            self.has_hf = True
            
            # Auto-detect username if organization not provided
            if organization is None:
                try:
                    user_info = whoami()
                    self.organization = user_info['name']
                    logger.info(f"✓ Auto-detected HuggingFace user: {self.organization}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not auto-detect user, using repo_name only: {e}")
                    self.organization = None
            else:
                self.organization = organization
            
            # Build full repo_id with username
            self.repo_id = f"{self.organization}/{self.repo_name}" if self.organization else self.repo_name
            logger.info(f"✓ HuggingFace Hub integration available ({stage}) - Repo: {self.repo_id}")
        except ImportError:
            self.has_hf = False
            logger.warning("⚠️ huggingface_hub not installed. Install with: pip install huggingface_hub")

    def login(self, token: Optional[str] = None):
        """Login to HuggingFace Hub"""
        if not self.has_hf:
            logger.error("HuggingFace Hub not available")
            return False

        try:
            from huggingface_hub import login
            login(token=token)
            logger.info("✓ Logged in to HuggingFace Hub")
            return True
        except Exception as e:
            logger.error(f"Failed to login to HuggingFace Hub: {e}")
            return False

    def create_repo(self):
        """Create repository on HuggingFace Hub if it doesn't exist"""
        if not self.has_hf:
            return False

        try:
            from huggingface_hub import create_repo
            create_repo(
                repo_id=self.repo_id,
                private=self.private,
                exist_ok=True
            )
            logger.info(f"✓ Repository ready: {self.repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            return False

    def push_model_checkpoint(self, model, config, epoch: int, metrics: Dict):
        """
        Push model checkpoint to HuggingFace Hub with metadata
        
        Args:
            model: The PyTorch model to save
            config: Model configuration (BitGenConfig object or dict)
            epoch: Current epoch number
            metrics: Training metrics dict
        """
        if not self.has_hf:
            logger.warning("HuggingFace Hub not available")
            return False
        
        # CRITICAL FIX: Ensure repository exists before pushing
        try:
            from huggingface_hub import create_repo, repo_exists
            import time
            
            # Check if repo exists
            repo_created_now = False
            if not repo_exists(self.repo_id, repo_type="model"):
                logger.info(f"🆕 Repository does not exist. Creating: {self.repo_id}")
                create_repo(
                    repo_id=self.repo_id,
                    private=self.private,
                    exist_ok=True,
                    repo_type="model"
                )
                logger.info(f"✅ Created repository: https://huggingface.co/{self.repo_id}")
                repo_created_now = True
                
                # CRITICAL: Wait for repo to propagate on HuggingFace servers
                # HuggingFace API has eventual consistency - repo creation takes time to propagate
                logger.info("⏳ Waiting 10 seconds for repository to propagate...")
                time.sleep(10)
                
                # Verify repo is now accessible
                max_retries = 3
                for retry in range(max_retries):
                    if repo_exists(self.repo_id, repo_type="model"):
                        logger.info(f"✓ Repository verified accessible after {retry + 1} checks")
                        break
                    else:
                        if retry < max_retries - 1:
                            logger.warning(f"⚠️ Repository not yet accessible, waiting 5 more seconds... (retry {retry + 1}/{max_retries})")
                            time.sleep(5)
                        else:
                            logger.error("❌ Repository still not accessible after retries")
                            return False
            else:
                logger.info(f"✓ Repository exists: {self.repo_id}")
                
        except Exception as e:
            logger.error(f"Failed to verify/create repository: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        try:
            import tempfile
            import torch
            from huggingface_hub import upload_file
            import json
            
            # Create temporary directory for checkpoint
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Save model state dict (standard HuggingFace naming)
                model_path = tmpdir_path / "pytorch_model.bin"
                torch.save(model.state_dict(), model_path)
                
                # Convert config to dict if it's a BitGenConfig object
                if hasattr(config, '__dict__'):
                    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
                else:
                    config_dict = config
                
                # Save config
                config_path = tmpdir_path / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                # Save training metadata (single file, overwritten each epoch)
                metadata_path = tmpdir_path / "training_metadata.json"
                # Convert tensor values to floats for JSON serialization
                json_metrics = {
                    'epoch': epoch,
                    'stage': self.stage
                }
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        json_metrics[k] = v.item()
                    elif isinstance(v, (int, float)):
                        json_metrics[k] = v
                    else:
                        json_metrics[k] = str(v)
                        
                with open(metadata_path, 'w') as f:
                    json.dump(json_metrics, f, indent=2)
                
                # Upload files with retry logic
                # Each upload OVERWRITES the previous file (no subdirectories)
                commit_msg = f"Checkpoint at epoch {epoch} - Loss: {metrics.get('total_loss', 0.0):.4f}"
                
                files_to_upload = [
                    (str(model_path), "pytorch_model.bin"),  # ROOT - overwrites previous
                    (str(config_path), "config.json"),       # ROOT - overwrites previous
                    (str(metadata_path), "training_metadata.json")  # ROOT - overwrites previous
                ]
                
                # Upload each file with retry logic
                max_upload_retries = 3
                for file_path, repo_path in files_to_upload:
                    upload_success = False
                    for retry in range(max_upload_retries):
                        try:
                            upload_file(
                                path_or_fileobj=file_path,
                                path_in_repo=repo_path,
                                repo_id=self.repo_id,
                                commit_message=commit_msg
                            )
                            upload_success = True
                            break
                        except Exception as upload_error:
                            if retry < max_upload_retries - 1:
                                logger.warning(f"⚠️ Upload failed for {repo_path} (retry {retry + 1}/{max_upload_retries}): {upload_error}")
                                time.sleep(5)  # Wait before retry
                            else:
                                logger.error(f"❌ Failed to upload {repo_path} after {max_upload_retries} retries")
                                raise upload_error
                    
                    if not upload_success:
                        logger.error(f"❌ Could not upload {repo_path}")
                        return False
                
                logger.info(f"✓ Pushed checkpoint for epoch {epoch} to {self.repo_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to push checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False

    def push_model(self, checkpoint_path: str, commit_message: str = "Update model"):
        """
        Push model checkpoint to HuggingFace Hub

        Args:
            checkpoint_path: Path to model checkpoint
            commit_message: Commit message for the push
        """
        if not self.has_hf:
            logger.error("HuggingFace Hub not available")
            return False

        try:
            from huggingface_hub import upload_file

            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False

            # Upload checkpoint
            upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=checkpoint_path.name,
                repo_id=self.repo_id,
                commit_message=commit_message
            )

            logger.info(f"✓ Pushed {checkpoint_path.name} to {self.repo_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to push model: {e}")
            return False

    def push_config(self, config_dict: Dict, filename: str = "config.json"):
        """Push model configuration to HuggingFace Hub"""
        if not self.has_hf:
            return False

        try:
            import json
            from huggingface_hub import upload_file
            import tempfile

            # Save config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_dict, f, indent=2)
                temp_path = f.name

            # Upload config
            upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=filename,
                repo_id=self.repo_id,
                commit_message="Update model config"
            )

            # Clean up temp file
            os.unlink(temp_path)

            logger.info(f"✓ Pushed config to {self.repo_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to push config: {e}")
            return False

    def push_folder(self, folder_path: str, commit_message: str = "Update model files"):
        """Push entire folder to HuggingFace Hub"""
        if not self.has_hf:
            return False

        try:
            from huggingface_hub import upload_folder

            folder_path = Path(folder_path)
            if not folder_path.exists():
                logger.error(f"Folder not found: {folder_path}")
                return False

            upload_folder(
                folder_path=str(folder_path),
                repo_id=self.repo_id,
                commit_message=commit_message
            )

            logger.info(f"✓ Pushed folder {folder_path} to {self.repo_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to push folder: {e}")
            return False
    
    def push_checkpoint(self, checkpoint_path: str, checkpoint_name: str, metrics: Dict):
        """
        Push checkpoint to HuggingFace Hub at specific iteration (BitMar-style)
        
        Args:
            checkpoint_path: Local path to checkpoint file
            checkpoint_name: Name of checkpoint (e.g., "checkpoint-1000")
            metrics: Dictionary with step, epoch, and other metrics
        """
        if not self.has_hf:
            logger.warning("HuggingFace Hub not available, skipping push")
            return False
        
        try:
            from huggingface_hub import upload_file
            import json
            import tempfile
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            # Upload checkpoint to subfolder
            repo_path = f"{checkpoint_name}/{checkpoint_name}.pt"
            upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                commit_message=f"Add {checkpoint_name} at step {metrics.get('step', 0)}"
            )
            
            # Also upload metrics
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(metrics, f, indent=2)
                temp_metrics_path = f.name
            
            metrics_repo_path = f"{checkpoint_name}/metrics.json"
            upload_file(
                path_or_fileobj=temp_metrics_path,
                path_in_repo=metrics_repo_path,
                repo_id=self.repo_id,
                commit_message=f"Add metrics for {checkpoint_name}"
            )
            
            os.unlink(temp_metrics_path)
            
            logger.info(f"✓ Pushed {checkpoint_name} to {self.repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push checkpoint {checkpoint_name}: {e}")
            return False
    
    def create_model_card(self, config_dict: Dict, training_args: Dict):
        """
        Create and push comprehensive model card to HuggingFace Hub
        
        Args:
            config_dict: Model configuration dictionary
            training_args: Training arguments and hyperparameters
        """
        if not self.has_hf:
            return False
        
        try:
            import tempfile
            from huggingface_hub import upload_file
            from datetime import datetime
            
            # Generate model card content
            model_card = f"""---
language: en
license: mit
tags:
- vision-language
- image-captioning
- contrastive-learning
- bitnet
- efficient-ai
- edge-ai
datasets:
- coco
metrics:
- accuracy
---

# BitGen Stage 1: Vision-Language Pre-training

## Model Description

**BitGen** is a tiny, efficient vision-language model designed for edge devices and resource-constrained environments. This is the **Stage 1 checkpoint** focusing on vision-language pre-training using the COCO dataset.

### Architecture

BitGen combines three powerful components:

1. **BitMar Encoder-Decoder**: 1.58-bit quantized transformer (BitNet b1.58) for extreme efficiency
2. **FIBER Cross-Modal Fusion**: Queue-based contrastive learning for vision-language alignment
3. **Larimar GPM**: Generative Parametric Memory for episodic memory and reasoning

### Model Size (Tiny Configuration)

- **Embedding Dimension**: {config_dict.get('embed_dim', 128)}
- **Encoder Layers**: {config_dict.get('num_layers', 4)}
- **Decoder Layers**: 2
- **Attention Heads**: {config_dict.get('num_heads', 4)}
- **FFN Dimension**: {config_dict.get('ffn_dim', 256)}
- **Vocabulary Size**: {config_dict.get('vocab_size', 50257)} (GPT-2 tokenizer)
- **Memory Slots**: {config_dict.get('memory_size', 32)}
- **Max Sequence Length**: {config_dict.get('max_seq_len', 256)}
- **Total Parameters**: ~5-10M (tiny enough for edge devices!)

### Training Data

- **Dataset**: MS-COCO Captions (validated subset)
- **Image-Caption Pairs**: ~118k training samples
- **Tokenizer**: GPT-2 BPE tokenizer

### Training Objectives

1. **Image-Text Contrastive (ITC) Loss** [Weight: {training_args.get('contrastive_weight', 1.0)} - PRIMARY]
   - FIBER-style queue-based contrastive learning
   - Aligns vision and language representations
   - Hard negative mining from queue

2. **Image-Text Matching (ITM) Loss** [Weight: {training_args.get('itm_weight', 0.5)}]
   - Binary classification with hard negatives
   - Learns fine-grained image-caption associations

3. **Text Reconstruction Loss** [Weight: {training_args.get('text_loss_weight', 0.5)} - AUXILIARY]
   - Decoder reconstructs captions from fused features
   - Maintains language understanding
   - Label smoothing (0.1) to prevent mode collapse

4. **Memory KL Divergence** [Weight: {training_args.get('memory_kl_weight', 0.1)}]
   - Larimar GPM episodic memory regularization
   - Bayesian inference over memory parameters

### Key Features

✅ **Tiny Model**: Suitable for edge devices (Raspberry Pi, mobile phones)  
✅ **1.58-bit Quantization**: Extreme efficiency via BitNet b1.58  
✅ **Vision-Language Alignment**: FIBER-style contrastive learning  
✅ **Episodic Memory**: Larimar GPM for memory-augmented reasoning  
✅ **Hard Negative Mining**: ITM loss for robust alignment  
✅ **DINOv2 Vision Encoder**: State-of-the-art vision features (trainable)  

## Usage

> **Note**: This repository contains only the **latest checkpoint**. Each training epoch overwrites the previous model file (`pytorch_model.bin`) to save storage. Git history preserves all versions.

### Loading the Model

```python
from transformers import AutoModel
import torch

# Load model from HuggingFace Hub (always the latest checkpoint)
model = AutoModel.from_pretrained("babylm-ntust/BitGen-PreReasoning-stage1")
model.eval()
```

### Inference Example

```python
from transformers import GPT2Tokenizer
from PIL import Image
import torchvision.transforms as transforms

# Setup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load image and caption
image = Image.open("path/to/image.jpg").convert('RGB')
caption = "A cat sitting on a couch"

# Prepare inputs
image_tensor = transform(image).unsqueeze(0)
tokens = tokenizer(caption, return_tensors='pt', padding=True, truncation=True, max_length=256)
input_ids = tokens['input_ids']

# Forward pass
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        return_contrastive_features=True
    )
    
    # Get similarity
    text_feat = outputs['contrastive_features']['text_features']
    image_feat = outputs['contrastive_features']['image_features']
    similarity = (text_feat @ image_feat.T).item()
    print(f"Similarity: {{similarity:.4f}}")
```

## Training Details

### Hyperparameters

- **Batch Size**: {training_args.get('batch_size', 128)} (effective: {training_args.get('batch_size', 128) * training_args.get('grad_accum_steps', 2)})
- **Learning Rate**: {training_args.get('learning_rate', 2e-4)}
- **Optimizer**: AdamW (weight_decay={training_args.get('weight_decay', 0.02)})
- **Gradient Accumulation**: {training_args.get('grad_accum_steps', 2)} steps
- **Max Gradient Norm**: {training_args.get('max_grad_norm', 1.0)}
- **Mixed Precision**: AMP
- **Temperature**: {training_args.get('temperature', 0.07)}
- **Queue Size**: {training_args.get('queue_size', 4096)}

### Training Schedule

- **Warmup Steps**: {training_args.get('warmup_steps', 1000)}
- **Scheduler**: Cosine decay with min LR = 0.1 × initial LR
- **Early Stopping**: Patience = {training_args.get('early_stopping_patience', 5)} epochs

## Limitations and Biases

### Limitations

1. **Tiny Model**: Designed for efficiency, not SOTA performance
2. **English Only**: Trained on English captions
3. **Stage 1 Only**: Pre-training phase; reasoning module in Stage 2
4. **Limited Context**: Max sequence length of 256 tokens
5. **COCO-Centric**: Training data from MS-COCO

### Biases

- Dataset bias from MS-COCO (Western-centric, object-focused)
- Vision bias from DINOv2 training data
- Language bias from GPT-2 tokenizer

## Citation

```bibtex
@software{{bitgen2025,
  title={{BitGen: Tiny Vision-Language Model for Edge Devices}},
  author={{BitGen Team}},
  year={{2025}},
  url={{https://huggingface.co/babylm-ntust/BitGen-PreReasoning-stage1}}
}}
```

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/euhidaman/BitGen).

---

**License**: MIT  
**Model Version**: Stage 1 (Vision-Language Pre-training)  
**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}
"""
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(model_card)
                temp_path = f.name
            
            # Upload as README.md
            upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="README.md",
                repo_id=self.repo_id,
                commit_message="Update model card"
            )
            
            os.unlink(temp_path)
            
            logger.info(f"✓ Model card pushed to {self.repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model card: {e}")
            import traceback
            traceback.print_exc()
            return False

