"""
Clean Dataset processing for BitMar - Unlimited Multimodal
No BabyLM constraints or train_50M dependencies
Enhanced with Facebook DINOv2-large support
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import logging
import random
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """Clean multimodal dataset without BabyLM constraints"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 512,
        vision_encoder_dim: int = 1024,  # DINOv2-large dimension
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length
        self.vision_encoder_dim = vision_encoder_dim
        self.split = split

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load multimodal data
        self.samples = []
        self.load_multimodal_data(max_samples)

        logger.info(f"✅ Clean multimodal dataset initialized with {len(self.samples):,} samples")

    def load_multimodal_data(self, max_samples: Optional[int] = None):
        """Load multimodal data without BabyLM constraints"""
        try:
            # Load Conceptual Captions
            cc_captions_path = self.dataset_dir / "cc_3M_captions.json"
            if cc_captions_path.exists():
                with open(cc_captions_path, 'r') as f:
                    captions = json.load(f)

                # Load DINOv2 features
                features_files = [
                    self.dataset_dir / "cc_3M_dino_v2_states_1of2.npy",
                    self.dataset_dir / "cc_3M_dino_v2_states_2of2.npy"
                ]

                vision_features = []
                for feat_file in features_files:
                    if feat_file.exists():
                        features = np.load(feat_file, mmap_mode='r')
                        vision_features.append(features)

                if vision_features:
                    all_features = np.concatenate(vision_features, axis=0)
                    logger.info(f"📊 Loaded vision features: {all_features.shape}")

                    # Create samples (unlimited)
                    limit = min(len(captions), len(all_features))
                    if max_samples:
                        limit = min(limit, max_samples)

                    for i in range(limit):
                        self.samples.append({
                            'text': captions[i],
                            'vision_features': all_features[i],
                            'source': 'conceptual_captions',
                            'index': i
                        })

                    logger.info(f"✅ Loaded {len(self.samples):,} multimodal samples")
                else:
                    logger.warning("⚠️  No vision features found")
            else:
                logger.warning("⚠️  No captions file found")

        except Exception as e:
            logger.error(f"❌ Failed to load multimodal data: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a multimodal sample"""
        sample = self.samples[idx]

        # Tokenize text
        text = sample['text']
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get vision features
        vision_features = sample['vision_features']
        if isinstance(vision_features, np.ndarray):
            vision_features = torch.from_numpy(vision_features).float()

        # Ensure correct dimension for DINOv2-large
        if vision_features.size(-1) != self.vision_encoder_dim:
            if vision_features.size(-1) < self.vision_encoder_dim:
                # Pad to match DINOv2-large dimension
                pad_size = self.vision_encoder_dim - vision_features.size(-1)
                vision_features = torch.cat([
                    vision_features,
                    torch.zeros(pad_size)
                ], dim=-1)
            else:
                # Truncate to match DINOv2-large dimension
                vision_features = vision_features[:self.vision_encoder_dim]

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0).clone(),
            'vision_features': vision_features,
            'has_vision': torch.tensor(True),
            'vision_index': torch.tensor(idx),
            'source': sample['source']
        }


class MultimodalDataModule:
    """Clean data module without BabyLM constraints"""

    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """Setup clean multimodal datasets"""
        logger.info("🚀 Setting up clean multimodal datasets (no BabyLM constraints)...")

        # Initialize tokenizer
        tokenizer_name = self.config.get('text_encoder_name', 'gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create dataset
        dataset_dir = self.config.get('dataset_dir', '../babylm_dataset')
        vision_encoder_dim = self.config.get('vision_encoder_dim', 1024)  # DINOv2-large

        full_dataset = MultimodalDataset(
            dataset_dir=dataset_dir,
            tokenizer_name=tokenizer_name,
            max_seq_length=self.config.get('max_seq_length', 512),
            vision_encoder_dim=vision_encoder_dim,
            max_samples=None  # No sample limits
        )

        # Split for validation if needed
        if self.config.get('use_validation', False):
            from torch.utils.data import random_split
            val_split = self.config.get('validation_split', 0.1)
            total_size = len(full_dataset)
            val_size = int(total_size * val_split)
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

            logger.info(f"📊 Dataset split: Train={train_size:,}, Val={val_size:,}")
        else:
            self.train_dataset = full_dataset
            self.val_dataset = None
            logger.info(f"📊 Single dataset: {len(full_dataset):,} samples")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            drop_last=True,
            persistent_workers=self.config.get('persistent_workers', True) and self.config.get('num_workers', 4) > 0
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader"""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            drop_last=False,
            persistent_workers=self.config.get('persistent_workers', True) and self.config.get('num_workers', 4) > 0
        )


def create_data_module(config: Dict) -> MultimodalDataModule:
    """Create clean multimodal data module"""
    logger.info("🚀 Creating clean multimodal data module...")
    logger.info("✅ No BabyLM constraints")
    logger.info("✅ No train_50M dependencies")
    logger.info("🔥 Enhanced with DINOv2-large support")

    return MultimodalDataModule(config)

