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
                 private: bool = False):
        """
        Initialize HuggingFace integration

        Args:
            repo_name: Repository name on HuggingFace Hub
            organization: Organization/user name (uses authenticated user if None)
            private: Whether to create private repository
        """
        self.repo_name = repo_name
        self.organization = organization
        self.private = private
        self.repo_id = f"{organization}/{repo_name}" if organization else repo_name

        # Check if huggingface_hub is available
        try:
            from huggingface_hub import HfApi, create_repo, login
            self.hf_api = HfApi()
            self.has_hf = True
            logger.info("✓ HuggingFace Hub integration available")
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
                
                # Save model state dict
                model_path = tmpdir_path / f"model_epoch_{epoch}.pt"
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
                
                # Save metrics
                metrics_path = tmpdir_path / f"metrics_epoch_{epoch}.json"
                # Convert tensor values to floats for JSON serialization
                json_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        json_metrics[k] = v.item()
                    elif isinstance(v, (int, float)):
                        json_metrics[k] = v
                    else:
                        json_metrics[k] = str(v)
                        
                with open(metrics_path, 'w') as f:
                    json.dump(json_metrics, f, indent=2)
                
                # Upload files with retry logic
                commit_msg = f"Checkpoint at epoch {epoch} - Loss: {metrics.get('total_loss', 0.0):.4f}"
                
                files_to_upload = [
                    (str(model_path), f"checkpoints/model_epoch_{epoch}.pt"),
                    (str(config_path), "config.json"),
                    (str(metrics_path), f"metrics/metrics_epoch_{epoch}.json")
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

