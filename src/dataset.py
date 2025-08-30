"""
Dataset processing for BitMar
Generic dataset handler - ready for any dataset
No dataset-specific constraints or dependencies
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


class GenericMultimodalDataset(Dataset):
    """Generic multimodal dataset - works with any image-caption data"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 256
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data - flexible loading for any dataset structure
        self._load_data()

        # Create indices for dataset
        self.indices = list(range(len(self.all_captions)))

        logger.info(f"Loaded {len(self.indices)} samples from {dataset_dir}")

    def _load_data(self):
        """Load multimodal data - flexible for different dataset formats"""
        logger.info("Loading dataset...")

        # Try different common dataset formats
        self.all_captions = []
        self.all_features = None

        # Check for JSON caption files
        json_files = list(self.dataset_dir.glob("*.json"))
        npy_files = list(self.dataset_dir.glob("*.npy"))

        if json_files and npy_files:
            # Load from JSON + NPY format
            caption_file = json_files[0]  # Use first JSON file
            feature_file = npy_files[0]   # Use first NPY file

            with open(caption_file, 'r', encoding='utf-8') as f:
                self.all_captions = json.load(f)

            self.all_features = np.load(feature_file, mmap_mode='r')

            logger.info(f"Loaded from JSON+NPY: {len(self.all_captions)} captions, {len(self.all_features)} features")

        elif json_files:
            # JSON only - assume captions with optional image paths
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                self.all_captions = [item['caption'] if isinstance(item, dict) else str(item) for item in data]
            else:
                self.all_captions = list(data.values()) if isinstance(data, dict) else [str(data)]

            # Create dummy vision features if no NPY files
            self.all_features = np.random.randn(len(self.all_captions), 768).astype(np.float32)
            logger.info(f"Loaded from JSON only: {len(self.all_captions)} captions (dummy vision features created)")

        else:
            # Try text files
            txt_files = list(self.dataset_dir.glob("*.txt"))
            if txt_files:
                for txt_file in txt_files:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    self.all_captions.extend([line.strip() for line in lines if line.strip()])

                # Create dummy vision features
                self.all_features = np.random.randn(len(self.all_captions), 768).astype(np.float32)
                logger.info(f"Loaded from TXT files: {len(self.all_captions)} captions (dummy vision features created)")
            else:
                # Empty dataset - ready for new data
                self.all_captions = ["Sample caption"]
                self.all_features = np.random.randn(1, 768).astype(np.float32)
                logger.warning(f"No data found in {self.dataset_dir} - using placeholder data")

        # Verify alignment
        if len(self.all_captions) != len(self.all_features):
            logger.warning(f"Misaligned data: {len(self.all_captions)} captions vs {len(self.all_features)} features")
            # Trim to match
            min_len = min(len(self.all_captions), len(self.all_features))
            self.all_captions = self.all_captions[:min_len]
            self.all_features = self.all_features[:min_len]

        logger.info(f"Final dataset: {len(self.all_captions)} aligned caption-image pairs")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample"""
        real_idx = self.indices[idx]
        caption = self.all_captions[real_idx]
        vision_feature = self.all_features[real_idx]

        # Tokenize caption
        encoded = self.tokenizer(
            caption,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Create labels for text generation
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Handle vision features
        if len(vision_feature.shape) == 1 and vision_feature.shape[0] == 768:
            vision_tensor = torch.tensor(vision_feature.copy(), dtype=torch.float32).unsqueeze(0)
        else:
            vision_tensor = torch.tensor(vision_feature.copy(), dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'vision_features': vision_tensor,
            'has_vision': True,
            'vision_index': real_idx,
            'sample_type': 'caption',
            'sample_index': idx,
            'text': caption
        }


def create_data_module(config: Dict) -> 'GenericDataModule':
    """Create generic data module"""
    return GenericDataModule(config)


class GenericDataModule:
    """Generic data module for any multimodal dataset"""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset = None
        self.train_loader = None

    def setup(self, rebuild_cache: bool = False):
        """Setup the dataset"""
        logger.info("Setting up generic data module...")

        self.dataset = GenericMultimodalDataset(
            dataset_dir=self.config['dataset_dir'],
            tokenizer_name=self.config['text_encoder_name'],
            max_seq_length=self.config['max_seq_length']
        )

        logger.info(f"Dataset ready with {len(self.dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        if self.train_loader is None:
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=self.config.get('pin_memory', True),
                persistent_workers=self.config.get('persistent_workers', True),
                drop_last=True
            )
        return self.train_loader

    def val_dataloader(self) -> List[DataLoader]:
        """Return empty list since we use entire dataset for training"""
        return []


def test_dataset(config: Dict, max_samples: int = 10):
    """Test dataset loading and processing"""
    logger.info("Testing generic dataset...")

    # Create data module
    data_module = create_data_module(config)
    data_module.setup()

    # Test sample
    sample = data_module.dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
    logger.info(f"Vision features shape: {sample['vision_features'].shape}")
    logger.info(f"Caption: {sample['text'][:100]}...")

    logger.info("Dataset test completed successfully!")
    return data_module


if __name__ == "__main__":
    # Test configuration
    test_config = {
        'dataset_dir': "./data",  # Generic path
        'text_encoder_name': "gpt2",
        'max_seq_length': 256,
        'batch_size': 4,
        'num_workers': 0,
        'pin_memory': False
    }

    # Test dataset
    test_dataset(test_config)
