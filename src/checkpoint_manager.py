"""
Local Checkpoint Manager for BitGen
Manages local checkpoint storage with rolling cleanup to save disk space
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import torch
import shutil

class LocalCheckpointManager:
    """Manage local checkpoints with rolling cleanup"""

    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 10):
        """
        Initialize local checkpoint manager

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep locally (default: 10)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = logging.getLogger(__name__)

        # Create metadata file to track checkpoints
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint metadata: {e}")

        return {"checkpoints": [], "latest_checkpoint": None}

    def _save_metadata(self):
        """Save checkpoint metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint metadata: {e}")

    def save_checkpoint(self,
                       model,
                       config,
                       epoch: int,
                       step: int,
                       metrics: Dict = None,
                       tokenizer=None) -> str:
        """
        Save model checkpoint locally with rolling management

        Args:
            model: Model to save
            config: Model configuration
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            tokenizer: Tokenizer (optional)

        Returns:
            Path to saved checkpoint
        """

        # Create checkpoint identifier
        checkpoint_id = f"epoch-{epoch:03d}_step-{step:06d}"
        checkpoint_dir = self.checkpoint_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save model state dict
            model_path = checkpoint_dir / "pytorch_model.bin"
            torch.save(model.state_dict(), model_path)

            # Save config
            config_dict = config.__dict__ if hasattr(config, '__dict__') else config
            config_path = checkpoint_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            # Save training info
            training_info = {
                "epoch": epoch,
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics or {},
                "model_architecture": "BitGen",
                "checkpoint_id": checkpoint_id
            }

            training_info_path = checkpoint_dir / "training_info.json"
            with open(training_info_path, 'w') as f:
                json.dump(training_info, f, indent=2)

            # Save tokenizer if provided
            if tokenizer:
                tokenizer_path = checkpoint_dir / "tokenizer_config.json"
                tokenizer_config = {
                    "vocab_size": getattr(config, 'vocab_size', 8192),
                    "model_max_length": getattr(config, 'max_seq_len', 256),
                    "checkpoint_epoch": epoch,
                    "checkpoint_step": step
                }
                with open(tokenizer_path, 'w') as f:
                    json.dump(tokenizer_config, f, indent=2)

            # Update metadata
            checkpoint_entry = {
                "checkpoint_id": checkpoint_id,
                "epoch": epoch,
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "path": str(checkpoint_dir),
                "metrics": metrics or {},
                "size_mb": self._calculate_checkpoint_size(checkpoint_dir)
            }

            # Add to metadata
            self.metadata["checkpoints"].append(checkpoint_entry)
            self.metadata["latest_checkpoint"] = checkpoint_id

            # Sort checkpoints by epoch and step
            self.metadata["checkpoints"].sort(key=lambda x: (x["epoch"], x["step"]))

            # Manage rolling checkpoints
            self._manage_rolling_checkpoints()

            # Save metadata
            self._save_metadata()

            self.logger.info(f"✅ Saved checkpoint {checkpoint_id} locally: {checkpoint_dir}")
            return str(checkpoint_dir)

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            # Clean up failed checkpoint directory
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            raise

    def _manage_rolling_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""

        if len(self.metadata["checkpoints"]) > self.max_checkpoints:
            # Calculate how many to delete
            num_to_delete = len(self.metadata["checkpoints"]) - self.max_checkpoints
            checkpoints_to_delete = self.metadata["checkpoints"][:num_to_delete]

            self.logger.info(f"🗑️ Cleaning up {num_to_delete} old local checkpoints to maintain limit of {self.max_checkpoints}")

            # Delete old checkpoints
            for checkpoint in checkpoints_to_delete:
                checkpoint_path = Path(checkpoint["path"])
                if checkpoint_path.exists():
                    try:
                        shutil.rmtree(checkpoint_path)
                        self.logger.info(f"   Deleted local checkpoint: {checkpoint['checkpoint_id']}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete checkpoint {checkpoint['checkpoint_id']}: {e}")

            # Update metadata
            self.metadata["checkpoints"] = self.metadata["checkpoints"][num_to_delete:]

    def _calculate_checkpoint_size(self, checkpoint_dir: Path) -> float:
        """Calculate checkpoint size in MB"""
        total_size = 0
        for file_path in checkpoint_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        if self.metadata["latest_checkpoint"]:
            for checkpoint in self.metadata["checkpoints"]:
                if checkpoint["checkpoint_id"] == self.metadata["latest_checkpoint"]:
                    return checkpoint["path"]
        return None

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints"""
        return self.metadata["checkpoints"].copy()

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Load specific checkpoint by ID"""
        for checkpoint in self.metadata["checkpoints"]:
            if checkpoint["checkpoint_id"] == checkpoint_id:
                checkpoint_path = Path(checkpoint["path"])
                if checkpoint_path.exists():
                    return {
                        "model_path": checkpoint_path / "pytorch_model.bin",
                        "config_path": checkpoint_path / "config.json",
                        "training_info_path": checkpoint_path / "training_info.json",
                        "checkpoint_info": checkpoint
                    }
        return None

    def cleanup_all_checkpoints(self):
        """Remove all checkpoints (use with caution)"""
        self.logger.warning("🗑️ Cleaning up ALL local checkpoints")

        for checkpoint in self.metadata["checkpoints"]:
            checkpoint_path = Path(checkpoint["path"])
            if checkpoint_path.exists():
                try:
                    shutil.rmtree(checkpoint_path)
                    self.logger.info(f"   Deleted checkpoint: {checkpoint['checkpoint_id']}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete checkpoint {checkpoint['checkpoint_id']}: {e}")

        # Reset metadata
        self.metadata = {"checkpoints": [], "latest_checkpoint": None}
        self._save_metadata()

    def get_storage_info(self) -> Dict:
        """Get storage information about checkpoints"""
        total_size_mb = sum(checkpoint.get("size_mb", 0) for checkpoint in self.metadata["checkpoints"])

        return {
            "total_checkpoints": len(self.metadata["checkpoints"]),
            "max_checkpoints": self.max_checkpoints,
            "total_size_mb": total_size_mb,
            "total_size_gb": total_size_mb / 1024,
            "average_size_mb": total_size_mb / len(self.metadata["checkpoints"]) if self.metadata["checkpoints"] else 0,
            "storage_directory": str(self.checkpoint_dir),
            "latest_checkpoint": self.metadata["latest_checkpoint"]
        }
