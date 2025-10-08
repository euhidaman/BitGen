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

