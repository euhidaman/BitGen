"""
Unlimited Multimodal Dataset Handler for BitMar
Supports unlimited multimodal datasets without BabyLM constraints
Enhanced with Facebook DINOv2-large feature extraction
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from transformers import AutoTokenizer, AutoImageProcessor
import requests
from PIL import Image
import io
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from transformers import Dinov2Model, AutoImageProcessor
    DINOV2_AVAILABLE = True
    logger.info("✅ DINOv2 model available")
except ImportError:
    DINOV2_AVAILABLE = False
    logger.warning("⚠️  DINOv2 model not available")


class UnlimitedMultimodalDataset(Dataset):
    """
    Unlimited multimodal dataset without BabyLM constraints
    Supports multiple high-quality dataset sources with Facebook DINOv2-large
    """

    def __init__(
        self,
        dataset_sources: Dict[str, Any],
        tokenizer: AutoTokenizer,
        vision_processor: Optional[AutoImageProcessor] = None,
        max_seq_length: int = 512,
        vision_encoder_name: str = "facebook/dinov2-large",
        dynamic_loading: bool = True
    ):
        self.dataset_sources = dataset_sources
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.max_seq_length = max_seq_length
        self.vision_encoder_name = vision_encoder_name
        self.dynamic_loading = dynamic_loading

        # Initialize DINOv2-large if available
        self.setup_vision_encoder()

        # Load all available datasets
        self.samples = []
        self.load_datasets()

        logger.info(f"✅ Unlimited dataset initialized with {len(self.samples):,} samples")
        logger.info(f"🔥 Enhanced with {vision_encoder_name}")

    def setup_vision_encoder(self):
        """Setup Facebook DINOv2-large for feature extraction"""
        if not DINOV2_AVAILABLE:
            self.vision_encoder = None
            self.vision_processor = None
            logger.warning("⚠️  DINOv2 not available - using pre-computed features only")
            return

        try:
            logger.info(f"🔥 Loading {self.vision_encoder_name}...")
            self.vision_encoder = Dinov2Model.from_pretrained(self.vision_encoder_name)
            self.vision_processor = AutoImageProcessor.from_pretrained(self.vision_encoder_name)
            self.vision_encoder.eval()

            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vision_encoder = self.vision_encoder.to(device)

            logger.info(f"✅ {self.vision_encoder_name} loaded successfully")
            logger.info(f"   Device: {device}")
            logger.info(f"   Output dimension: {self.vision_encoder.config.hidden_size}")

        except Exception as e:
            logger.error(f"❌ Failed to load {self.vision_encoder_name}: {e}")
            self.vision_encoder = None
            self.vision_processor = None

    def load_datasets(self):
        """Load multiple high-quality dataset sources"""
        logger.info("📚 Loading unlimited multimodal datasets...")

        # Load Conceptual Captions if enabled
        if self.dataset_sources.get('conceptual_captions', {}).get('enabled', False):
            self.load_conceptual_captions()

        # Load Visual Genome if enabled
        if self.dataset_sources.get('visual_genome', {}).get('enabled', False):
            self.load_visual_genome()

        # Load COCO Captions if enabled
        if self.dataset_sources.get('coco_captions', {}).get('enabled', False):
            self.load_coco_captions()

        # Load Flickr30k if enabled
        if self.dataset_sources.get('flickr30k', {}).get('enabled', False):
            self.load_flickr30k()

        # Load custom datasets if enabled
        if self.dataset_sources.get('custom_datasets', {}).get('enabled', False):
            self.load_custom_datasets()

        # Shuffle all samples for better mixing
        random.shuffle(self.samples)

        logger.info(f"📊 Dataset loading summary:")
        logger.info(f"   Total samples: {len(self.samples):,}")
        self.log_dataset_statistics()

    def load_conceptual_captions(self):
        """Load Conceptual Captions dataset"""
        logger.info("📸 Loading Conceptual Captions...")

        # Try to load from multiple possible locations
        possible_paths = [
            "../babylm_dataset/cc_3M_captions.json",
            "../dataset/conceptual_captions/captions.json",
            "./data/conceptual_captions.json"
        ]

        captions_path = None
        for path in possible_paths:
            if Path(path).exists():
                captions_path = Path(path)
                break

        if captions_path is None:
            logger.warning("⚠️  Conceptual Captions not found")
            return

        try:
            with open(captions_path, 'r') as f:
                captions = json.load(f)

            # Load corresponding features if available
            features_paths = [
                "../babylm_dataset/cc_3M_dino_v2_states_1of2.npy",
                "../babylm_dataset/cc_3M_dino_v2_states_2of2.npy"
            ]

            features = []
            for feat_path in features_paths:
                if Path(feat_path).exists():
                    feat_data = np.load(feat_path, mmap_mode='r')
                    features.append(feat_data)

            if features:
                all_features = np.concatenate(features, axis=0)
                logger.info(f"   Features shape: {all_features.shape}")
            else:
                all_features = None
                logger.warning("   No pre-computed features found")

            # Create samples
            max_samples = self.dataset_sources['conceptual_captions'].get('max_samples')
            min_caption_length = self.dataset_sources['conceptual_captions'].get('min_caption_length', 10)

            for i, caption in enumerate(captions):
                if max_samples and i >= max_samples:
                    break

                if len(caption.split()) >= min_caption_length:
                    sample = {
                        'text': caption,
                        'image_id': f'cc_{i}',
                        'source': 'conceptual_captions',
                        'features': all_features[i] if all_features is not None else None
                    }
                    self.samples.append(sample)

            logger.info(f"✅ Loaded {len([s for s in self.samples if s['source'] == 'conceptual_captions']):,} CC samples")

        except Exception as e:
            logger.error(f"❌ Failed to load Conceptual Captions: {e}")

    def load_visual_genome(self):
        """Load Visual Genome dataset (placeholder - implement based on your data)"""
        logger.info("🔍 Visual Genome dataset loading - implement based on your data structure")
        # Implement based on your Visual Genome data format
        pass

    def load_coco_captions(self):
        """Load COCO Captions dataset (placeholder - implement based on your data)"""
        logger.info("🏞️  COCO Captions dataset loading - implement based on your data structure")
        # Implement based on your COCO data format
        pass

    def load_flickr30k(self):
        """Load Flickr30k dataset (placeholder - implement based on your data)"""
        logger.info("📷 Flickr30k dataset loading - implement based on your data structure")
        # Implement based on your Flickr30k data format
        pass

    def load_custom_datasets(self):
        """Load custom datasets"""
        logger.info("🔧 Loading custom datasets...")

        custom_config = self.dataset_sources.get('custom_datasets', {})
        # Implement based on your custom dataset format

        # Example implementation for a generic image-caption format
        # You would adapt this to your specific data structure
        pass

    def log_dataset_statistics(self):
        """Log dataset statistics"""
        sources = {}
        for sample in self.samples:
            source = sample['source']
            sources[source] = sources.get(source, 0) + 1

        for source, count in sources.items():
            logger.info(f"   {source}: {count:,} samples")

    def extract_vision_features(self, image_path_or_url: str) -> Optional[torch.Tensor]:
        """Extract vision features using DINOv2-large"""
        if self.vision_encoder is None:
            return None

        try:
            # Load image
            if image_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(image_path_or_url)
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path_or_url).convert('RGB')

            # Process image
            inputs = self.vision_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.vision_encoder.device) for k, v in inputs.items()}

            # Extract features
            with torch.no_grad():
                outputs = self.vision_encoder(**inputs)
                # Get pooled features (CLS token)
                features = outputs.pooler_output  # Shape: [1, hidden_size]
                return features.squeeze(0)  # Shape: [hidden_size]

        except Exception as e:
            logger.warning(f"Failed to extract vision features: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the unlimited dataset"""
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
        vision_features = sample.get('features')
        if vision_features is not None:
            if isinstance(vision_features, np.ndarray):
                vision_features = torch.from_numpy(vision_features).float()
        else:
            # Create dummy features if none available
            vision_dim = 1024  # DINOv2-large dimension
            vision_features = torch.zeros(vision_dim)

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0).clone(),
            'vision_features': vision_features,
            'has_vision': torch.tensor(True),
            'vision_index': torch.tensor(idx),
            'source': sample['source'],
            'image_id': sample['image_id']
        }


class UnlimitedDataModule:
    """Data module for unlimited multimodal training"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.setup_tokenizer()

    def setup_tokenizer(self):
        """Setup tokenizer"""
        tokenizer_name = self.config.get('text_encoder_name', 'gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"✅ Tokenizer setup: {tokenizer_name}")

    def setup(self):
        """Setup datasets"""
        logger.info("🚀 Setting up unlimited multimodal datasets...")

        # Get dataset sources from config
        dataset_sources = self.config.get('dataset_sources', {
            'conceptual_captions': {'enabled': True}
        })

        # Create full dataset
        full_dataset = UnlimitedMultimodalDataset(
            dataset_sources=dataset_sources,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.get('max_seq_length', 512),
            vision_encoder_name=self.config.get('vision_model_name', 'facebook/dinov2-large'),
            dynamic_loading=self.config.get('dynamic_loading', True)
        )

        # Split into train/validation if needed
        if self.config.get('use_validation', True):
            val_split = self.config.get('validation_split', 0.05)
            total_samples = len(full_dataset)
            val_size = int(total_samples * val_split)
            train_size = total_samples - val_size

            from torch.utils.data import random_split
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

            logger.info(f"📊 Dataset split:")
            logger.info(f"   Train: {train_size:,} samples")
            logger.info(f"   Validation: {val_size:,} samples")
        else:
            self.train_dataset = full_dataset
            self.val_dataset = None
            logger.info(f"📊 Single dataset: {len(full_dataset):,} samples")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 48),
            shuffle=True,
            num_workers=self.config.get('num_workers', 8),
            pin_memory=self.config.get('pin_memory', True),
            persistent_workers=self.config.get('persistent_workers', True) and self.config.get('num_workers', 8) > 0,
            prefetch_factor=self.config.get('prefetch_factor', 4),
            drop_last=True
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader"""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 48),
            shuffle=False,
            num_workers=self.config.get('num_workers', 8),
            pin_memory=self.config.get('pin_memory', True),
            persistent_workers=self.config.get('persistent_workers', True) and self.config.get('num_workers', 8) > 0,
            drop_last=False
        )


def create_unlimited_data_module(config: Dict[str, Any]) -> UnlimitedDataModule:
    """Create unlimited data module"""
    logger.info("🚀 Creating unlimited multimodal data module...")
    logger.info("✅ Removed all BabyLM constraints")
    logger.info("🔥 Enhanced with Facebook DINOv2-large support")

    return UnlimitedDataModule(config)
