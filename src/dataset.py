"""
Dataset processing for BitMar
Handles HuggingFace LocalizedNarratives and COCO datasets with vision feature extraction
Loads image-caption pairs and processes images into 768-dim features
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from typing import Dict, List, Tuple, Optional
import logging
import random
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import pickle
import hashlib

# Setup logger first
logger = logging.getLogger(__name__)

# HuggingFace datasets import
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
    logger.info("✅ HuggingFace datasets available")
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning(
        "⚠️ HuggingFace datasets not available - install with: pip install datasets")


class LocalizedNarrativesCOCODataset(Dataset):
    """Dataset for HuggingFace LocalizedNarratives and COCO with vision feature extraction"""

    def __init__(
        self,
        dataset_dir: str,
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 256,
        # Updated to DiNOv3 small (faster)
        vision_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        extract_vision_features: bool = True,
        use_dummy_vision: bool = False,
        cache_vision_features: bool = True,
        force_rebuild_cache: bool = False,
        use_hf_localized_narratives: bool = False,  # Disabled due to deprecated scripts
        hf_dataset_config: str = "open_images",  # open_images, ait, flickr, coco
        max_samples_per_split: int = 10000  # Limit samples for faster loading
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_seq_length = max_seq_length
        self.extract_vision_features = extract_vision_features
        self.use_dummy_vision = use_dummy_vision
        self.cache_vision_features = cache_vision_features
        self.force_rebuild_cache = force_rebuild_cache
        self.use_hf_localized_narratives = use_hf_localized_narratives and HF_DATASETS_AVAILABLE
        self.hf_dataset_config = hf_dataset_config
        self.max_samples_per_split = max_samples_per_split

        # Create cache directory
        self.cache_dir = self.dataset_dir / "vision_features_cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Look for pre-cached features from download step (standard naming)
        self.download_cache_file = self.cache_dir / "all_features.npy"
        self.download_metadata_file = self.cache_dir / "cache_metadata.pkl"

        # Fallback to training-time cache (legacy naming scheme)
        self.cache_key = hashlib.md5(
            f"{vision_model}_{extract_vision_features}_{use_dummy_vision}".encode()
        ).hexdigest()[:8]
        self.training_cache_file = self.cache_dir / \
            f"features_{self.cache_key}.npy"
        self.training_metadata_file = self.cache_dir / \
            f"metadata_{self.cache_key}.pkl"

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Vision model initialization - only if we're forced to rebuild and no download cache exists
        if (force_rebuild_cache and not self.download_cache_file.exists() and
                extract_vision_features and not use_dummy_vision):
            try:
                logger.info(f"Loading vision model: {vision_model}")
                self.vision_processor = AutoImageProcessor.from_pretrained(
                    vision_model)
                self.vision_model = AutoModel.from_pretrained(vision_model)
                self.vision_model.eval()
                logger.info("✅ Vision model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load vision model: {e}")
                logger.warning("Falling back to dummy vision features")
                self.use_dummy_vision = True
                self.vision_processor = None
                self.vision_model = None
        else:
            self.vision_processor = None
            self.vision_model = None

        # Load all datasets
        self._load_datasets()

        # Try to load cached features (prioritize download cache over training cache)
        self._load_or_create_vision_features()

        # Create indices for dataset
        self.indices = list(range(len(self.all_captions)))

        logger.info(
            f"✅ Loaded {len(self.indices):,} image-caption pairs from Localized Narratives + COCO")

    def _load_datasets(self):
        """Load HuggingFace LocalizedNarratives and COCO datasets"""
        logger.info("📥 Loading datasets from HuggingFace and local files...")

        self.all_captions = []
        self.all_image_ids = []
        self.all_image_urls = []
        self.dataset_sources = []

        # First, try to load unified captions if available
        unified_file = self.dataset_dir / "all_captions.json"
        if unified_file.exists():
            logger.info("📄 Loading unified captions file...")
            with open(unified_file, 'r', encoding='utf-8') as f:
                self.all_captions = json.load(f)

            # Create dummy image data
            self.all_image_ids = list(range(len(self.all_captions)))
            self.all_image_urls = [None] * len(self.all_captions)
            self.dataset_sources = ['unified'] * len(self.all_captions)

            logger.info(
                f"✅ Loaded {len(self.all_captions):,} captions from unified file")
            return

        # Load HuggingFace LocalizedNarratives if enabled
        if self.use_hf_localized_narratives:
            logger.warning(
                "⚠️ HuggingFace LocalizedNarratives disabled due to deprecated scripts")
            self._load_localized_narratives()
        else:
            # Use local Localized Narratives
            self._load_localized_narratives()
            # Load COCO captions with image URLs (only when not using HuggingFace)
            self._load_coco_captions()

        logger.info(
            f"✅ Total loaded: {len(self.all_captions):,} image-caption pairs")

    def _load_or_create_vision_features(self):
        """Load cached vision features - prioritize download cache, never create during training"""

        # Priority 1: Look for download cache (created by download_multimodal_data.py)
        if self.download_cache_file.exists() and self.download_metadata_file.exists():
            try:
                logger.info(
                    f"🚀 Loading pre-cached vision features from download step: {self.download_cache_file}")

                # Load metadata to verify compatibility
                with open(self.download_metadata_file, 'rb') as f:
                    metadata = pickle.load(f)

                # Load cached features
                self.all_features = np.load(self.download_cache_file)
                logger.info(
                    f"✅ Loaded {len(self.all_features):,} pre-cached vision features from download")
                logger.info(
                    f"   Feature type: {metadata.get('feature_type', 'unknown')}")
                logger.info(
                    f"   Feature shape per sample: {metadata.get('feature_shape', 'unknown')}")
                return

            except Exception as e:
                logger.warning(f"Failed to load download cache: {e}")

        # Priority 2: Look for training cache (legacy)
        if (self.cache_vision_features and not self.force_rebuild_cache and
                self.training_cache_file.exists() and self.training_metadata_file.exists()):
            try:
                logger.info(
                    f"🚀 Loading cached vision features from training cache: {self.training_cache_file}")

                # Load metadata to verify compatibility
                with open(self.training_metadata_file, 'rb') as f:
                    metadata = pickle.load(f)

                # Verify cache is compatible
                if (metadata['num_samples'] == len(self.all_captions) and
                    metadata['use_dummy_vision'] == self.use_dummy_vision and
                        metadata['extract_vision_features'] == self.extract_vision_features):

                    # Load cached features
                    self.all_features = np.load(self.training_cache_file)
                    logger.info(
                        f"✅ Loaded {len(self.all_features):,} cached vision features from training cache")
                    return
                else:
                    logger.info("⚠️  Training cache metadata mismatch")
            except Exception as e:
                logger.warning(f"Failed to load training cache: {e}")

        # Priority 3: Error - no cache available and we don't create features during training
        logger.error("❌ No pre-cached vision features found!")
        logger.error(
            "   Please run download script with --cache_vision_features first:")
        logger.error(
            "   python download_multimodal_data.py --dataset both --data_dir ./data --cache_vision_features")
        logger.error(
            "   Or use the manage_vision_cache.py script to create cache")

        # Create dummy features as absolute fallback to prevent crashes
        logger.warning(
            "⚠️  Creating dummy features as fallback - training may not work properly")
        self.all_features = np.random.randn(
            len(self.all_captions), 768).astype(np.float32)

    def _save_vision_features_cache(self):
        """Save vision features to disk cache"""
        try:
            logger.info(f"💾 Saving vision features cache to {self.cache_file}")

            # Save features
            np.save(self.cache_file, self.all_features)

            # Save metadata
            metadata = {
                'num_samples': len(self.all_captions),
                'use_dummy_vision': self.use_dummy_vision,
                'extract_vision_features': self.extract_vision_features,
                'cache_key': self.cache_key,
                # File modification time
                'created_at': str(Path(__file__).stat().st_mtime)
            }

            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info("✅ Vision features cached successfully")

        except Exception as e:
            logger.warning(f"Failed to save vision features cache: {e}")

        # Pre-compute vision features if requested
        if self.extract_vision_features and not self.use_dummy_vision:
            self._precompute_vision_features()
        elif self.use_dummy_vision:
            self.all_features = np.random.randn(
                len(self.all_captions), 768).astype(np.float32)
            logger.info(
                "🎨 Created dummy vision features (768-dim) for all samples")

    def _load_localized_narratives(self):
        """Load Localized Narratives from JSONL files with image URLs"""
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
                                        caption = data.get('caption') or data.get(
                                            'narrative', '')
                                        if caption and len(caption.strip()) > 0:
                                            self.all_captions.append(
                                                caption.strip())
                                            self.all_image_ids.append(
                                                data.get('image_id', f"ln_{line_num}"))
                                            self.all_image_urls.append(
                                                data.get('image_url'))
                                            self.dataset_sources.append(
                                                f"localized_narratives_{dataset_dir.name}")
                                            ln_count += 1
                                    except json.JSONDecodeError as e:
                                        logger.debug(
                                            f"Skipping malformed JSON on line {line_num}: {e}")
                                        continue
                    except Exception as e:
                        logger.warning(f"Failed to load {jsonl_file}: {e}")

        logger.info(f"✅ Localized Narratives: {ln_count:,} samples loaded")

    def _load_hf_localized_narratives(self):
        """Load LocalizedNarratives from HuggingFace datasets"""
        if not HF_DATASETS_AVAILABLE:
            logger.warning(
                "⚠️ HuggingFace datasets not available, falling back to local loading")
            self._load_localized_narratives()
            return

        try:
            logger.info(
                f"🤗 Loading LocalizedNarratives from HuggingFace (config: {self.hf_dataset_config})...")

            # Load the dataset from HuggingFace
            dataset = load_dataset(
                "HuggingFaceM4/LocalizedNarratives", self.hf_dataset_config)

            ln_count = 0

            # Process train split
            if 'train' in dataset:
                train_data = dataset['train']
                max_train_samples = min(
                    len(train_data), self.max_samples_per_split)

                logger.info(
                    f"📖 Processing {max_train_samples:,} train samples...")

                for i in tqdm(range(max_train_samples), desc="Loading train samples"):
                    try:
                        sample = train_data[i]

                        # Extract caption
                        caption = sample.get('caption', '')
                        if caption and len(caption.strip()) > 0:
                            self.all_captions.append(caption.strip())

                            # Extract image information
                            image_id = sample.get('image_id', f"hf_train_{i}")
                            self.all_image_ids.append(image_id)

                            # Handle image - it's a PIL Image object from HuggingFace
                            image = sample.get('image')
                            if image is not None:
                                # Store the PIL image directly for later processing
                                self.all_image_urls.append(image)
                            else:
                                self.all_image_urls.append(None)

                            self.dataset_sources.append(
                                f"hf_localized_narratives_{self.hf_dataset_config}_train")
                            ln_count += 1

                    except Exception as e:
                        logger.debug(f"Error processing train sample {i}: {e}")
                        continue

            # Process validation split if available
            if 'validation' in dataset:
                val_data = dataset['validation']
                max_val_samples = min(
                    len(val_data), self.max_samples_per_split // 2)

                logger.info(
                    f"📖 Processing {max_val_samples:,} validation samples...")

                for i in tqdm(range(max_val_samples), desc="Loading validation samples"):
                    try:
                        sample = val_data[i]

                        # Extract caption
                        caption = sample.get('caption', '')
                        if caption and len(caption.strip()) > 0:
                            self.all_captions.append(caption.strip())

                            # Extract image information
                            image_id = sample.get('image_id', f"hf_val_{i}")
                            self.all_image_ids.append(image_id)

                            # Handle image - it's a PIL Image object from HuggingFace
                            image = sample.get('image')
                            if image is not None:
                                # Store the PIL image directly for later processing
                                self.all_image_urls.append(image)
                            else:
                                self.all_image_urls.append(None)

                            self.dataset_sources.append(
                                f"hf_localized_narratives_{self.hf_dataset_config}_val")
                            ln_count += 1

                    except Exception as e:
                        logger.debug(
                            f"Error processing validation sample {i}: {e}")
                        continue

            logger.info(
                f"✅ HuggingFace LocalizedNarratives: {ln_count:,} samples loaded")

        except Exception as e:
            logger.error(
                f"❌ Failed to load HuggingFace LocalizedNarratives: {e}")
            logger.info("🔄 Falling back to local loading...")
            self._load_localized_narratives()

    def _load_coco_captions(self):
        """Load COCO captions from annotation files with image URLs"""
        coco_dir = self.dataset_dir / "coco" / "annotations"
        if not coco_dir.exists():
            logger.warning("⚠️  COCO annotations directory not found")
            return

        coco_count = 0

        # Load image info first to get URLs
        image_info = {}
        for split in ['train2017', 'val2017']:
            instances_file = coco_dir / f"instances_{split}.json"
            if instances_file.exists():
                try:
                    with open(instances_file, 'r') as f:
                        data = json.load(f)
                    for img in data['images']:
                        # COCO images follow pattern: http://images.cocodataset.org/{split}/{id:012d}.jpg
                        image_url = f"http://images.cocodataset.org/zips/{split}/{img['file_name']}"
                        image_info[img['id']] = image_url
                except Exception as e:
                    logger.warning(
                        f"Failed to load image info from {instances_file}: {e}")

        # Load captions with image URLs
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
                            image_id = annotation['image_id']
                            self.all_image_ids.append(image_id)
                            self.all_image_urls.append(
                                image_info.get(image_id))
                            self.dataset_sources.append(f"coco_{split}")
                            coco_count += 1

                except Exception as e:
                    logger.warning(f"Failed to load {captions_file}: {e}")

        logger.info(f"✅ COCO: {coco_count:,} samples loaded")

    def _precompute_vision_features(self):
        """Pre-compute vision features for all images (supports both URLs and PIL images)"""
        logger.info("🔄 Pre-computing vision features from images...")

        self.all_features = []

        for idx, image_source in enumerate(tqdm(self.all_image_urls, desc="Processing images")):
            try:
                if image_source:
                    # Handle PIL Image objects from HuggingFace
                    if isinstance(image_source, Image.Image):
                        image = image_source.convert('RGB')
                    elif isinstance(image_source, str):
                        # Download and process image from URL
                        response = requests.get(image_source, timeout=10)
                        image = Image.open(
                            BytesIO(response.content)).convert('RGB')
                    else:
                        # Unknown image source type
                        logger.debug(
                            f"Unknown image source type for idx {idx}: {type(image_source)}")
                        self.all_features.append(
                            np.random.randn(768).astype(np.float32))
                        continue

                    # Process with vision model
                    inputs = self.vision_processor(image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.vision_model(**inputs)
                        # Get pooled features (768-dim)
                        features = outputs.pooler_output.squeeze().numpy()
                        self.all_features.append(features)
                else:
                    # No image source, use dummy features
                    self.all_features.append(
                        np.random.randn(768).astype(np.float32))

            except Exception as e:
                logger.debug(f"Failed to process image {idx}: {e}")
                # Use dummy features for failed images
                self.all_features.append(
                    np.random.randn(768).astype(np.float32))

        self.all_features = np.array(self.all_features)
        logger.info(f"✅ Pre-computed {len(self.all_features)} vision features")

    def _get_vision_features(self, idx: int) -> np.ndarray:
        """Get vision features for a sample (always use cached/pre-computed now)"""
        # Always use pre-computed/cached features now
        if hasattr(self, 'all_features'):
            return self.all_features[idx]

        # Fallback to dummy if no features available (shouldn't happen)
        logger.warning(
            f"No pre-computed features available for sample {idx}, using dummy")
        return np.random.randn(768).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample with vision features"""
        real_idx = self.indices[idx]
        caption = self.all_captions[real_idx]

        # Get vision features
        vision_feature = self._get_vision_features(real_idx)

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

        # Convert attention mask to boolean for PyTorch compatibility
        attention_mask = attention_mask.bool()

        # Create labels for text generation
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Handle vision features - ensure proper 768-dim format
        vision_tensor = torch.tensor(
            vision_feature.copy(), dtype=torch.float32)
        if vision_tensor.dim() == 1:
            # Ensure it's exactly 768 dimensions
            if vision_tensor.size(0) != 768:
                if vision_tensor.size(0) > 768:
                    vision_tensor = vision_tensor[:768]
                else:
                    # Pad to 768
                    pad_size = 768 - vision_tensor.size(0)
                    vision_tensor = torch.cat(
                        [vision_tensor, torch.zeros(pad_size)])
        else:
            # Flatten and ensure 768 dims
            vision_tensor = vision_tensor.flatten()
            if vision_tensor.size(0) != 768:
                if vision_tensor.size(0) > 768:
                    vision_tensor = vision_tensor[:768]
                else:
                    pad_size = 768 - vision_tensor.size(0)
                    vision_tensor = torch.cat(
                        [vision_tensor, torch.zeros(pad_size)])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'vision_features': vision_tensor,  # Always 768-dimensional
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
    """Data module for Localized Narratives and COCO datasets with vision processing"""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset = None
        self.train_loader = None

    def setup(self, rebuild_cache: bool = False):
        """Setup the dataset with vision feature extraction"""
        logger.info(
            "🔧 Setting up Localized Narratives + COCO data module with vision processing...")

        # Determine if we should extract real vision features or use dummy ones
        extract_vision = self.config.get('extract_vision_features', False)
        # Default to dummy for speed
        use_dummy = self.config.get('use_dummy_vision', True)
        cache_features = self.config.get(
            'cache_vision_features', True)  # Default to caching

        if extract_vision and not use_dummy:
            logger.info(
                "🎨 Real vision feature extraction enabled (slower but better quality)")
            if cache_features:
                logger.info(
                    "💾 Vision feature caching enabled - features will be computed once and cached")
        else:
            logger.info("⚡ Using dummy vision features (faster training)")

        self.dataset = LocalizedNarrativesCOCODataset(
            dataset_dir=self.config['dataset_dir'],
            tokenizer_name=self.config['text_encoder_name'],
            max_seq_length=self.config['max_seq_length'],
            extract_vision_features=extract_vision,
            use_dummy_vision=use_dummy,
            cache_vision_features=cache_features,
            force_rebuild_cache=rebuild_cache
        )

        logger.info(f"📊 Dataset ready with {len(self.dataset):,} samples")

    def clear_vision_cache(self):
        """Clear cached vision features"""
        if hasattr(self, 'dataset') and self.dataset:
            cache_dir = self.dataset.cache_dir
            if cache_dir.exists():
                logger.info(
                    f"🗑️  Clearing vision features cache from {cache_dir}")
                for cache_file in cache_dir.glob("*.npy"):
                    cache_file.unlink()
                    logger.info(f"   Deleted: {cache_file}")
                for metadata_file in cache_dir.glob("*.pkl"):
                    metadata_file.unlink()
                    logger.info(f"   Deleted: {metadata_file}")
                logger.info("✅ Vision features cache cleared")
            else:
                logger.info("ℹ️  No vision features cache to clear")


def create_data_module(config: Dict) -> LocalizedNarrativesCOCODataModule:
    """Create data module from config"""
    return LocalizedNarrativesCOCODataModule(config)


class LocalizedNarrativesCOCODataModule:
    """Data module for Localized Narratives and COCO datasets with vision processing"""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset = None
        self.train_loader = None

    def setup(self, rebuild_cache: bool = False):
        """Setup the dataset with vision feature extraction"""
        logger.info(
            "🔧 Setting up Localized Narratives + COCO data module with vision processing...")

        # Determine if we should extract real vision features or use dummy ones
        extract_vision = self.config.get('extract_vision_features', False)
        # Default to dummy for speed
        use_dummy = self.config.get('use_dummy_vision', True)
        cache_features = self.config.get(
            'cache_vision_features', True)  # Default to caching

        if extract_vision and not use_dummy:
            logger.info(
                "🎨 Real vision feature extraction enabled (slower but better quality)")
            if cache_features:
                logger.info(
                    "💾 Vision feature caching enabled - features will be computed once and cached")
        else:
            logger.info("⚡ Using dummy vision features (faster training)")

        self.dataset = LocalizedNarrativesCOCODataset(
            dataset_dir=self.config['dataset_dir'],
            tokenizer_name=self.config['text_encoder_name'],
            max_seq_length=self.config['max_seq_length'],
            extract_vision_features=extract_vision,
            use_dummy_vision=use_dummy,
            cache_vision_features=cache_features,
            force_rebuild_cache=rebuild_cache
        )

        logger.info(f"📊 Dataset ready with {len(self.dataset):,} samples")

    def clear_vision_cache(self):
        """Clear cached vision features"""
        if hasattr(self, 'dataset') and self.dataset:
            cache_dir = self.dataset.cache_dir
            if cache_dir.exists():
                logger.info(
                    f"🗑️  Clearing vision features cache from {cache_dir}")
                for cache_file in cache_dir.glob("*.npy"):
                    cache_file.unlink()
                    logger.info(f"   Deleted: {cache_file}")
                for metadata_file in cache_dir.glob("*.pkl"):
                    metadata_file.unlink()
                    logger.info(f"   Deleted: {metadata_file}")
                logger.info("✅ Vision features cache cleared")
            else:
                logger.info("ℹ️  No vision features cache to clear")
        else:
            logger.warning("⚠️  No dataset loaded, cannot clear cache")

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
            'datasets': 'Localized Narratives + COCO',
            'vision_features': '768-dimensional (model compatible)',
            'extract_real_vision': not self.dataset.use_dummy_vision
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
    logger.info(f"Vision features type: {sample['vision_features'].dtype}")
    logger.info(f"Caption: {sample['text'][:100]}...")
    logger.info(f"Source: {sample['source']}")

    # Verify vision features are exactly 768-dim
    assert sample['vision_features'].shape == torch.Size(
        [768]), f"Expected [768], got {sample['vision_features'].shape}"
    logger.info("✅ Vision features are correctly formatted for model")

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
        'pin_memory': False,
        'extract_vision_features': False,  # Set to True for real vision features
        'use_dummy_vision': True  # Set to False for real vision features
    }

    # Test dataset
    test_dataset(test_config)
