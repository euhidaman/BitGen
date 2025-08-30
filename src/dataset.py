"""
Dataset processing for BitMar
Handles Localized Narratives and COCO datasets
Loads image-caption pairs for multimodal training
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


class LocalizedNarrativesCOCODataset(Dataset):
    """Dataset for Localized Narratives and COCO image-caption pairs"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 256,
        use_dummy_vision: bool = True
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length
        self.use_dummy_vision = use_dummy_vision

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load all datasets
        self._load_datasets()

        # Create indices for dataset
        self.indices = list(range(len(self.all_captions)))

        logger.info(f"✅ Loaded {len(self.indices):,} image-caption pairs from Localized Narratives + COCO")

    def _load_datasets(self):
        """Load Localized Narratives and COCO datasets"""
        logger.info("📥 Loading Localized Narratives and COCO datasets...")

        self.all_captions = []
        self.all_image_ids = []
        self.dataset_sources = []

        # First, try to load unified captions if available
        unified_file = self.dataset_dir / "all_captions.json"
        if unified_file.exists():
            logger.info("📄 Loading unified captions file...")
            with open(unified_file, 'r', encoding='utf-8') as f:
                self.all_captions = json.load(f)

            # Create dummy image IDs and sources
            self.all_image_ids = list(range(len(self.all_captions)))
            self.dataset_sources = ['unified'] * len(self.all_captions)

            logger.info(f"✅ Loaded {len(self.all_captions):,} captions from unified file")
            return

        # Load Localized Narratives
        self._load_localized_narratives()

        # Load COCO captions
        self._load_coco_captions()

        logger.info(f"✅ Total loaded: {len(self.all_captions):,} image-caption pairs")

        # Create dummy vision features if needed
        if self.use_dummy_vision:
            self.all_features = np.random.randn(len(self.all_captions), 768).astype(np.float32)
            logger.info("🎨 Created dummy vision features (768-dim) for all samples")

    def _load_localized_narratives(self):
        """Load Localized Narratives from JSONL files"""
        ln_dir = self.dataset_dir / "localized_narratives"
        if not ln_dir.exists():
            logger.warning("⚠️  Localized Narratives directory not found")
            return

        ln_count = 0
        for dataset_dir in ln_dir.iterdir():
            if dataset_dir.is_dir():
                for jsonl_file in dataset_dir.glob("*.jsonl"):
                    logger.info(f"📄 Loading {jsonl_file}")
                    try:
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f):
                                if line.strip():
                                    try:
                                        data = json.loads(line)
                                        # Extract caption/narrative
                                        caption = data.get('caption') or data.get('narrative', '')
                                        if caption and len(caption.strip()) > 0:
                                            self.all_captions.append(caption.strip())
                                            self.all_image_ids.append(data.get('image_id', f"ln_{line_num}"))
                                            self.dataset_sources.append(f"localized_narratives_{dataset_dir.name}")
                                            ln_count += 1
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"Skipping malformed JSON on line {line_num}: {e}")
                                        continue
                    except Exception as e:
                        logger.warning(f"Failed to load {jsonl_file}: {e}")

        logger.info(f"✅ Localized Narratives: {ln_count:,} samples loaded")

    def _load_coco_captions(self):
        """Load COCO captions from annotation files"""
        coco_dir = self.dataset_dir / "coco" / "annotations"
        if not coco_dir.exists():
            logger.warning("⚠️  COCO annotations directory not found")
            return

        coco_count = 0
        for split in ['train2017', 'val2017']:
            captions_file = coco_dir / f"captions_{split}.json"
            if captions_file.exists():
                logger.info(f"📄 Loading {captions_file}")
                try:
                    with open(captions_file, 'r') as f:
                        data = json.load(f)

                    for annotation in data['annotations']:
                        caption = annotation['caption'].strip()
                        if len(caption) > 0:
                            self.all_captions.append(caption)
                            self.all_image_ids.append(annotation['image_id'])
                            self.dataset_sources.append(f"coco_{split}")
                            coco_count += 1

                except Exception as e:
                    logger.warning(f"Failed to load {captions_file}: {e}")

        logger.info(f"✅ COCO: {coco_count:,} samples loaded")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample"""
        real_idx = self.indices[idx]
        caption = self.all_captions[real_idx]

        # Get dummy vision features or image ID
        if self.use_dummy_vision and hasattr(self, 'all_features'):
            vision_feature = self.all_features[real_idx]
        else:
            # Create random vision features
            vision_feature = np.random.randn(768).astype(np.float32)

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
        vision_tensor = torch.tensor(vision_feature.copy(), dtype=torch.float32)
        if vision_tensor.dim() == 1:
            vision_tensor = vision_tensor.unsqueeze(0)  # Add batch dimension if needed

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'vision_features': vision_tensor,
            'has_vision': True,
            'vision_index': real_idx,
            'sample_type': 'caption',
            'sample_index': idx,
            'text': caption,
            'image_id': self.all_image_ids[real_idx] if hasattr(self, 'all_image_ids') else real_idx,
            'source': self.dataset_sources[real_idx] if hasattr(self, 'dataset_sources') else 'unknown'
        }


def create_data_module(config: Dict) -> 'LocalizedNarrativesCOCODataModule':
    """Create data module for Localized Narratives + COCO"""
    return LocalizedNarrativesCOCODataModule(config)


class LocalizedNarrativesCOCODataModule:
    """Data module for Localized Narratives and COCO datasets"""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset = None
        self.train_loader = None

    def setup(self, rebuild_cache: bool = False):
        """Setup the dataset"""
        logger.info("🔧 Setting up Localized Narratives + COCO data module...")

        self.dataset = LocalizedNarrativesCOCODataset(
            dataset_dir=self.config['dataset_dir'],
            tokenizer_name=self.config['text_encoder_name'],
            max_seq_length=self.config['max_seq_length'],
            use_dummy_vision=True  # Use dummy vision features for now
        )

        logger.info(f"📊 Dataset ready with {len(self.dataset):,} samples")

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

    def get_dataset_info(self) -> Dict[str, any]:
        """Get dataset information"""
        if not self.dataset:
            return {}

        return {
            'total_samples': len(self.dataset),
            'tokenizer': self.config['text_encoder_name'],
            'max_seq_length': self.config['max_seq_length'],
            'datasets': 'Localized Narratives + COCO'
        }


def test_dataset(config: Dict):
    """Test dataset loading and processing"""
    logger.info("🧪 Testing Localized Narratives + COCO dataset...")

    # Create data module
    data_module = create_data_module(config)
    data_module.setup()

    # Test sample
    sample = data_module.dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
    logger.info(f"Vision features shape: {sample['vision_features'].shape}")
    logger.info(f"Caption: {sample['text'][:100]}...")
    logger.info(f"Source: {sample['source']}")

    logger.info("✅ Dataset test completed successfully!")
    return data_module


if __name__ == "__main__":
    # Test configuration
    test_config = {
        'dataset_dir': "./data",
        'text_encoder_name': "gpt2",
        'max_seq_length': 256,
        'batch_size': 4,
        'num_workers': 0,
        'pin_memory': False
    }

    # Test dataset
    test_dataset(test_config)
