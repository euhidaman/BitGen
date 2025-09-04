#!/usr/bin/env python3
"""
Download and prepare Localized Narratives and COCO datasets
Creates image-caption pairs for multimodal training
With optional vision features pre-caching
"""

import os
import json
import requests
import zipfile
import tarfile
import gzip
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse
import numpy as np
import hashlib
import pickle

# HuggingFace datasets import
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDatasetDownloader:
    """Download and prepare Localized Narratives and COCO datasets"""

    def __init__(self, data_dir: str = "./data", cache_vision_features: bool = False, use_real_vision: bool = False,
                 use_hf_localized_narratives: bool = False, hf_dataset_config: str = "open_images", max_samples: int = 50000):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Sample limit configuration
        self.max_samples = max_samples if max_samples > 0 else float('inf')  # 0 or negative means unlimited

        # HuggingFace options (disabled by default due to deprecated scripts)
        self.use_hf_localized_narratives = use_hf_localized_narratives and HF_DATASETS_AVAILABLE
        self.hf_dataset_config = hf_dataset_config

        if self.use_hf_localized_narratives:
            logger.warning(
                f"⚠️ HuggingFace LocalizedNarratives uses deprecated scripts - may not work")
        else:
            logger.info(
                "📁 Will download LocalizedNarratives from original sources")

        # Vision caching options
        self.cache_vision_features = cache_vision_features
        self.use_real_vision = use_real_vision
        self.vision_cache_dir = self.data_dir / "vision_features_cache"

        if self.cache_vision_features:
            self.vision_cache_dir.mkdir(exist_ok=True)
            logger.info(
                f"Vision features caching enabled: {self.vision_cache_dir}")

            if self.use_real_vision:
                # Check if vision dependencies are available for real features
                try:
                    import torch
                    from transformers import Dinov2Model, AutoImageProcessor
                    from PIL import Image
                    import io
                    self.torch = torch
                    self.Dinov2Model = Dinov2Model
                    self.AutoImageProcessor = AutoImageProcessor
                    self.Image = Image
                    self.io = io

                    logger.info("Will extract real DiNOv2 features")
                    self._load_vision_model()
                except ImportError as e:
                    logger.error(
                        f"❌ Real vision features requested but dependencies not available: {e}")
                    logger.error(
                        "   Install with: pip install torch transformers pillow")
                    raise ImportError(
                        "Vision dependencies required for real features")
            else:
                logger.info(
                    "Will use dummy vision features (no additional dependencies required)")

        # Dataset URLs
        self.datasets = {
            'localized_narratives': {
                'open_images': {
                    'train': [
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00000-of-00010.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00001-of-00010.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00002-of-00010.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00003-of-00010.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00004-of-00010.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00005-of-00010.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00006-of-00010.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00007-of-00010.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00008-of-00010.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00009-of-00010.jsonl'
                    ],
                    'validation': 'https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_localized_narratives.jsonl',
                    'test': 'https://storage.googleapis.com/localized-narratives/annotations/open_images_test_localized_narratives.jsonl'
                },
                'coco': {
                    'train': [
                        'https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00000-of-00004.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00001-of-00004.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00002-of-00004.jsonl',
                        'https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00003-of-00004.jsonl'
                    ],
                    'validation': 'https://storage.googleapis.com/localized-narratives/annotations/coco_val_localized_narratives.jsonl'
                }
            },
            'coco': {
                'annotations': {
                    'train2017': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                },
                'images': {
                    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
                    'val2017': 'http://images.cocodataset.org/zips/val2017.zip'
                }
            }
        }

    def _load_vision_model(self):
        """Load DiNOv2 vision model for real feature extraction"""
        try:
            logger.info("Loading DiNOv2 vision model...")
            # Use a working DiNOv2 model that exists and is compatible
            model_name = "facebook/dinov2-base"
            self.vision_model = self.Dinov2Model.from_pretrained(model_name)
            self.vision_processor = self.AutoImageProcessor.from_pretrained(
                model_name)

            # Check for GPU availability and move model
            if self.torch.cuda.is_available():
                device_name = self.torch.cuda.get_device_name(0)
                self.vision_model = self.vision_model.cuda()
                logger.info(f"✅ DiNOv2 model loaded on GPU: {device_name}")
                logger.info(
                    f"   GPU Memory Available: {self.torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                logger.warning(
                    "⚠️  No GPU detected, using CPU for DiNOv2 feature extraction (will be slower)")

            self.vision_model.eval()

        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            logger.info("Falling back to dummy vision features")
            self.use_real_vision = False

    def _generate_cache_key(self, image_url: str, dataset_name: str) -> str:
        """Generate a unique cache key for an image"""
        key_string = f"{image_url}_{dataset_name}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _cache_vision_features(self, image_sources: List[str], dataset_name: str) -> int:
        """Cache vision features for a list of image sources (URLs or local files)"""
        if not self.cache_vision_features:
            return 0

        # Filter out None values (images not found locally)
        valid_image_sources = [src for src in image_sources if src is not None]

        if not valid_image_sources:
            logger.warning(f"No valid image sources found for {dataset_name}")
            return 0

        logger.info(
            f"Caching vision features for {len(valid_image_sources)} images from {dataset_name}")

        cached_count = 0

        with tqdm(total=len(valid_image_sources), desc=f"Caching {dataset_name} features") as pbar:
            for image_source in valid_image_sources:
                cache_key = self._generate_cache_key(
                    str(image_source), dataset_name)
                cache_file = self.vision_cache_dir / f"{cache_key}.npy"
                metadata_file = self.vision_cache_dir / f"{cache_key}.pkl"

                # Skip if already cached
                if cache_file.exists() and metadata_file.exists():
                    pbar.update(1)
                    continue

                try:
                    if self.use_real_vision:
                        # Extract real features
                        if image_source and os.path.exists(str(image_source)):
                            # Local file - use file extraction method
                            features = self._extract_real_vision_features_from_file(
                                image_source)
                        elif image_source and image_source.startswith('http'):
                            # URL - but handle 403 gracefully
                            features = self._extract_real_vision_features(
                                image_source)
                        else:
                            # No valid source, use dummy
                            features = self.torch.randn(768)
                        features_array = features.cpu().numpy()
                    else:
                        # Generate dummy features (768-dimensional like DiNOv2) using numpy
                        features_array = np.random.randn(
                            768).astype(np.float32)

                    # Save features and metadata
                    np.save(cache_file, features_array)

                    metadata = {
                        'image_source': str(image_source),
                        'dataset_name': dataset_name,
                        'feature_type': 'real_dinov2' if self.use_real_vision else 'dummy',
                        'feature_shape': features_array.shape
                    }

                    with open(metadata_file, 'wb') as f:
                        pickle.dump(metadata, f)

                    cached_count += 1

                except Exception as e:
                    logger.debug(
                        f"Failed to cache features for {image_source}: {e}")
                    # Create dummy features as fallback
                    features_array = np.random.randn(768).astype(np.float32)

                    try:
                        np.save(cache_file, features_array)
                        metadata = {
                            'image_source': str(image_source),
                            'dataset_name': dataset_name,
                            'feature_type': 'dummy_fallback',
                            'feature_shape': features_array.shape
                        }
                        with open(metadata_file, 'wb') as f:
                            pickle.dump(metadata, f)
                        cached_count += 1
                    except Exception as e2:
                        logger.warning(
                            f"Failed to save fallback features: {e2}")

                pbar.update(1)

        logger.info(
            f"✅ Cached {cached_count} new vision features for {dataset_name}")
        return cached_count

    def _extract_real_vision_features_from_file(self, image_path: str):
        """Extract real DiNOv2 features from local image file"""
        try:
            # Load local image
            image = self.Image.open(image_path).convert('RGB')
            inputs = self.vision_processor(images=image, return_tensors="pt")

            # Move inputs to GPU if available
            if self.torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Extract features using GPU
            with self.torch.no_grad():
                outputs = self.vision_model(**inputs)
                features = outputs.last_hidden_state.mean(
                    dim=1).squeeze()  # Global average pooling

            return features.cpu()

        except Exception as e:
            logger.warning(
                f"Failed to extract features from local file {image_path}: {e}")
            # Return dummy features as fallback
            return self.torch.randn(768)

    def _extract_real_vision_features(self, image_url: str):
        """Extract real DiNOv2 features from image URL using GPU if available"""
        try:
            # Download image with timeout and handle errors gracefully
            response = requests.get(image_url, timeout=10)
            if response.status_code == 403:
                logger.debug(
                    f"403 Forbidden for {image_url} - using dummy features")
                return self.torch.randn(768)
            response.raise_for_status()

            # Process with DiNOv2
            image = self.Image.open(self.io.BytesIO(
                response.content)).convert('RGB')
            inputs = self.vision_processor(images=image, return_tensors="pt")

            # Move inputs to GPU if available
            if self.torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Extract features using GPU
            with self.torch.no_grad():
                outputs = self.vision_model(**inputs)
                features = outputs.last_hidden_state.mean(
                    dim=1).squeeze()  # Global average pooling

            return features.cpu()

        except requests.exceptions.RequestException as e:
            if "403" in str(e) or "Forbidden" in str(e):
                logger.debug(
                    f"Image forbidden/inaccessible: {image_url} - using dummy features")
            else:
                logger.debug(
                    f"Failed to download {image_url}: {e} - using dummy features")
            return self.torch.randn(768)
        except Exception as e:
            logger.debug(
                f"Failed to extract features from {image_url}: {e} - using dummy features")
            # Return dummy features as fallback
            return self.torch.randn(768)

    def _cache_hf_vision_features(self, images: List, dataset_name: str) -> int:
        """Cache vision features for HuggingFace PIL images"""
        if not self.cache_vision_features:
            return 0

        logger.info(
            f"Caching vision features for {len(images)} HuggingFace images from {dataset_name}")

        # Save all features to a single file for HuggingFace data
        all_features = []

        with tqdm(total=len(images), desc=f"Caching {dataset_name} features") as pbar:
            for i, image in enumerate(images):
                try:
                    if self.use_real_vision:
                        # Extract real features from PIL image
                        features = self._extract_real_vision_features_from_pil(
                            image)
                        features_array = features.cpu().numpy()
                    else:
                        # Generate dummy features (768-dimensional)
                        features_array = np.random.randn(
                            768).astype(np.float32)

                    all_features.append(features_array)

                except Exception as e:
                    logger.warning(
                        f"Failed to cache features for image {i}: {e}")
                    # Add dummy features as fallback
                    all_features.append(
                        np.random.randn(768).astype(np.float32))

                pbar.update(1)

        # Save all features to cache
        try:
            all_features_array = np.array(all_features)
            cache_file = self.vision_cache_dir / "all_features.npy"
            metadata_file = self.vision_cache_dir / "cache_metadata.pkl"

            np.save(cache_file, all_features_array)

            metadata = {
                'dataset_name': dataset_name,
                'num_samples': len(all_features),
                'feature_type': 'real_dinov2' if self.use_real_vision else 'dummy',
                'feature_shape': all_features_array[0].shape,
                'source': 'huggingface'
            }

            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(
                f"✅ Cached {len(all_features)} HuggingFace vision features to {cache_file}")
            return len(all_features)

        except Exception as e:
            logger.error(
                f"Failed to save HuggingFace vision features cache: {e}")
            return 0

    def _extract_real_vision_features_from_pil(self, image):
        """Extract real DiNOv2 features from PIL image"""
        try:
            # Process with DiNOv2
            inputs = self.vision_processor(images=image, return_tensors="pt")

            # Move inputs to GPU if available
            if self.torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Extract features using GPU
            with self.torch.no_grad():
                outputs = self.vision_model(**inputs)
                features = outputs.last_hidden_state.mean(
                    dim=1).squeeze()  # Global average pooling

            return features.cpu()

        except Exception as e:
            logger.error(f"Failed to extract features from PIL image: {e}")
            # Return dummy features as fallback
            return self.torch.randn(768)

    def download_file(self, url: str, filepath: Path, desc: str = None) -> bool:
        """Download a file with progress bar"""
        try:
            logger.info(f"Downloading {desc or url} to {filepath}")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or "Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"✅ Downloaded {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to download {url}: {e}")
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract archive file"""
        try:
            logger.info(f"Extracting {archive_path} to {extract_to}")
            extract_to.mkdir(exist_ok=True)

            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.error(
                    f"Unsupported archive format: {archive_path.suffix}")
                return False

            logger.info(f"✅ Extracted {archive_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to extract {archive_path}: {e}")
            return False

    def download_localized_narratives(self) -> Dict[str, int]:
        """Download Localized Narratives annotations or load from HuggingFace"""
        logger.info(
            f"🔧 Debug: use_hf_localized_narratives = {self.use_hf_localized_narratives}")
        logger.info(
            f"🔧 Debug: HF_DATASETS_AVAILABLE = {HF_DATASETS_AVAILABLE}")

        if self.use_hf_localized_narratives:
            logger.info("🤗 Attempting HuggingFace LocalizedNarratives...")
            return self._load_hf_localized_narratives()
        else:
            logger.info("📁 Using local LocalizedNarratives download...")
            return self._download_local_localized_narratives()

    def _load_hf_localized_narratives(self) -> Dict[str, int]:
        """Load LocalizedNarratives from HuggingFace and cache vision features"""
        logger.warning(
            "⚠️ HuggingFace LocalizedNarratives has deprecated dataset scripts")
        logger.info(
            "� Falling back to local processing with dummy vision features...")
        return self._download_local_localized_narratives()

    def _download_local_localized_narratives(self) -> Dict[str, int]:
        """Download Localized Narratives from original sources and get images from source datasets"""
        logger.info("📥 Downloading Localized Narratives...")

        ln_dir = self.data_dir / "localized_narratives"
        ln_dir.mkdir(exist_ok=True)

        total_samples = 0
        dataset_info = {}
        all_image_info = []  # Store image_id and dataset info for downloading actual images

        # Use configurable sample limit
        samples_collected = 0

        for dataset_name, splits in self.datasets['localized_narratives'].items():
            # Only process Open Images (revert back to original working approach)
            # But limit to first 50k samples for faster downloads
            if dataset_name != 'open_images':
                logger.info(
                    f"⏭️ Skipping {dataset_name} dataset (only using Open Images subset)")
                continue

            dataset_dir = ln_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)

            dataset_samples = 0

            # Only process training split to reduce dataset size
            train_splits = {k: v for k,
                            v in splits.items() if 'train' in k.lower()}
            for split, urls in train_splits.items():
                # Handle sharded URLs (list) or single URL (string)
                if isinstance(urls, list):
                    # Sharded files (like Open Images train)
                    dataset_samples_split = 0
                    for i, url in enumerate(urls):
                        filename = f"{dataset_name}_{split}_shard_{i:02d}.jsonl"
                        filepath = dataset_dir / filename

                        if filepath.exists():
                            logger.info(
                                f"✅ {filename} already exists, skipping download")
                        else:
                            success = self.download_file(
                                url, filepath, f"Localized Narratives {dataset_name} {split} shard {i}")
                            if not success:
                                continue

                        # Parse annotations and collect image info for downloading
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                samples = 0
                                for line in f:
                                    if line.strip() and samples_collected < self.max_samples:
                                        try:
                                            data = json.loads(line)
                                            samples += 1
                                            samples_collected += 1
                                            # Store image info for later downloading
                                            all_image_info.append({
                                                'dataset_id': data.get('dataset_id', ''),
                                                'image_id': data.get('image_id', ''),
                                                'source_file': filepath
                                            })

                                            # Stop if we've collected enough samples
                                            if samples_collected >= self.max_samples:
                                                logger.info(
                                                    f"🛑 Reached limit of {self.max_samples:,} samples, stopping...")
                                                break
                                        except json.JSONDecodeError:
                                            continue
                            dataset_samples_split += samples
                            logger.info(
                                f"    ◦ shard {i}: {samples:,} samples")

                            # Break shard loop if we've hit the limit
                            if samples_collected >= self.max_samples:
                                break

                        except Exception as e:
                            logger.warning(f"Failed to parse {filename}: {e}")

                    dataset_samples += dataset_samples_split
                    logger.info(
                        f"  • {split}: {dataset_samples_split:,} samples total")
                else:
                    # Single file
                    filename = f"{dataset_name}_{split}.jsonl"
                    filepath = dataset_dir / filename

                    if filepath.exists():
                        logger.info(
                            f"✅ {filename} already exists, skipping download")
                    else:
                        success = self.download_file(
                            urls, filepath, f"Localized Narratives {dataset_name} {split}")
                        if not success:
                            continue

                    # Parse annotations and collect image info
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            samples = 0
                            for line in f:
                                if line.strip() and samples_collected < self.max_samples:
                                    try:
                                        data = json.loads(line)
                                        samples += 1
                                        samples_collected += 1

                                        # Debug: Log dataset_id for first few samples
                                        if samples <= 3:
                                            logger.info(
                                                f"🔍 Debug sample {samples} from {filename}:")
                                            logger.info(
                                                f"  dataset_id: '{data.get('dataset_id', 'MISSING')}'")
                                            logger.info(
                                                f"  image_id: '{data.get('image_id', 'MISSING')}'")

                                        # Store image info for later downloading
                                        all_image_info.append({
                                            'dataset_id': data.get('dataset_id', ''),
                                            'image_id': data.get('image_id', ''),
                                            'source_file': filepath
                                        })

                                        # Stop if we've collected enough samples
                                        if samples_collected >= self.max_samples:
                                            logger.info(
                                                f"🛑 Reached limit of {self.max_samples:,} samples, stopping...")
                                            break
                                    except json.JSONDecodeError:
                                        continue
                        dataset_samples += samples
                        logger.info(f"  • {split}: {samples:,} samples")

                        # Break outer loop if we've hit the limit
                        if samples_collected >= self.max_samples:
                            break

                    except Exception as e:
                        logger.warning(f"Failed to parse {filename}: {e}")

            dataset_info[dataset_name] = dataset_samples
            total_samples += dataset_samples

            # Break dataset loop if we've hit the limit
            if samples_collected >= self.max_samples:
                logger.info(
                    f"🛑 Sample limit reached, stopping dataset processing...")
                break

        logger.info(
            f"✅ Localized Narratives: {total_samples:,} samples loaded")

        # Now download actual images from source datasets based on image_ids
        if all_image_info:
            self._download_source_images(all_image_info)

        return dataset_info

    def _download_source_images(self, image_info_list: List[Dict]):
        """Download actual images from source datasets (Open Images, COCO, etc.) based on image_ids"""
        logger.info(
            f"📥 Downloading {len(image_info_list):,} images from source datasets...")

        # Group by dataset for efficient downloading
        datasets_to_download = {}
        for info in image_info_list:
            dataset_id = info['dataset_id']
            if dataset_id not in datasets_to_download:
                datasets_to_download[dataset_id] = []
            datasets_to_download[dataset_id].append(info['image_id'])

        # Debug: Show what dataset_ids we found
        logger.info(
            f"🔍 Debug: Found dataset_ids: {list(datasets_to_download.keys())}")
        for dataset_id, image_ids in datasets_to_download.items():
            logger.info(f"🔍 Debug: {dataset_id}: {len(image_ids):,} images")

        for dataset_id, image_ids in datasets_to_download.items():
            logger.info(
                f"📂 Downloading {len(image_ids):,} images from {dataset_id}")

            if 'open_images' in dataset_id.lower() or dataset_id == '':
                # Handle both explicit open_images and empty dataset_id (assume open_images)
                self._download_open_images_by_ids(image_ids)
            elif 'coco' in dataset_id.lower():
                self._download_coco_images_by_ids(image_ids, dataset_id)
            elif 'flickr30k' in dataset_id.lower():
                flickr30k_dir = self.data_dir / "flickr30k"
                flickr30k_dir.mkdir(exist_ok=True)
                self._download_flickr30k_images_by_ids(
                    image_ids, flickr30k_dir)
            elif 'ade20k' in dataset_id.lower():
                ade20k_dir = self.data_dir / "ade20k"
                ade20k_dir.mkdir(exist_ok=True)
                self._download_ade20k_images_by_ids(image_ids, ade20k_dir)
            else:
                logger.warning(
                    f"⚠️ Unknown dataset type: {dataset_id} - treating as Open Images")
                self._download_open_images_by_ids(image_ids)

    def _download_open_images_by_ids(self, image_ids: List[str]):
        """Download Open Images using image IDs"""
        logger.info(f"🖼️ Processing {len(image_ids)} Open Images...")

        # Create Open Images download directory
        oi_dir = self.data_dir / "open_images"
        oi_dir.mkdir(exist_ok=True)

        # Quick pre-check: count existing images first (BEFORE CSV download)
        logger.info("🔍 Quick check: scanning existing images...")
        existing_files = list(oi_dir.glob("*.jpg"))
        existing_ids = {f.stem for f in existing_files if f.stat().st_size > 0}
        needed_ids = [
            img_id for img_id in image_ids if img_id not in existing_ids]

        logger.info(f"📊 Found {len(existing_ids)} existing images in folder")
        logger.info(f"📊 Need to check {len(needed_ids)} more images")

        if len(needed_ids) == 0:
            logger.info(
                "✅ All requested images already exist! Skipping download entirely.")
            logger.info(f"📁 All {len(image_ids)} images found in {oi_dir}")
            return len(existing_ids)

        # Open Images URLs follow pattern: https://c{bucket}.staticflickr.com/{server}/{id}_{secret}_{size}.jpg
        # But we need the CSV files to get the actual URLs
        logger.info(
            f"📄 Only {len(needed_ids)} images missing - downloading CSV to get URLs...")

        # Download image URLs CSV (this contains the mapping from image_id to actual URL)
        # Only use training set to keep it manageable
        train_csv_url = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"

        # Download only training CSV file to get image URLs
        for csv_name, csv_url in [("train", train_csv_url)]:
            csv_file = oi_dir / f"{csv_name}_images.csv"
            if not csv_file.exists():
                logger.info(f"📥 Downloading {csv_name} CSV...")
                if not self.download_file(csv_url, csv_file, f"Open Images {csv_name} CSV"):
                    continue

        # Parse CSV files to build image_id -> URL mapping
        logger.info("🔗 Building image_id to URL mapping from training CSV...")
        id_to_url = {}

        # Only process training CSV
        csv_file = oi_dir / "train_images.csv"
        if csv_file.exists():
            try:
                import csv
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        image_id = row.get('ImageID', '')
                        original_url = row.get('OriginalURL', '')
                        if image_id and original_url:
                            id_to_url[image_id] = original_url
                logger.info(
                    f"✅ Loaded {len(id_to_url)} image URLs from training CSV")
            except Exception as e:
                logger.warning(f"Failed to parse {csv_file}: {e}")

        # Download images based on IDs (only the needed ones)
        downloaded_count = 0
        existing_count = 0

        logger.info(
            f"🔍 Now downloading the {len(needed_ids)} missing images...")

        for image_id in tqdm(needed_ids, desc="Downloading missing Open Images"):
            if image_id in id_to_url:
                image_url = id_to_url[image_id]
                image_file = oi_dir / f"{image_id}.jpg"

                # Since we pre-filtered, this should not exist, but double-check anyway
                if image_file.exists() and image_file.stat().st_size > 0:
                    existing_count += 1
                    continue

                # Download the missing image
                try:
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        with open(image_file, 'wb') as f:
                            f.write(response.content)
                        downloaded_count += 1
                    else:
                        logger.debug(
                            f"Failed to download {image_url}: {response.status_code}")
                except Exception as e:
                    logger.debug(f"Error downloading {image_url}: {e}")

        total_existing = len(existing_ids)  # From pre-check
        logger.info(
            f"✅ Found {total_existing} existing images, downloaded {downloaded_count} new images")
        logger.info(
            f"📊 Total images available: {total_existing + downloaded_count}")

        return total_existing + downloaded_count

    def _download_coco_images_by_ids(self, image_ids: List[str], dataset_id: str):
        """Download COCO images using bulk zip download for much faster downloads"""
        logger.info(
            f"🖼️ Downloading {len(image_ids)} COCO images for {dataset_id}...")

        coco_dir = self.data_dir / "coco_images"
        coco_dir.mkdir(exist_ok=True)

        # Determine split from dataset_id
        split = "train2017" if "train" in dataset_id else "val2017"

        # Try bulk download first - much faster than individual downloads
        zip_url = f"http://images.cocodataset.org/zips/{split}.zip"
        zip_file = coco_dir / f"{split}.zip"
        extract_dir = coco_dir / split

        if not extract_dir.exists():
            logger.info(
                f"🗜️ Attempting bulk download of COCO {split} (~20GB zip file)...")
            logger.info("This is MUCH faster than individual image downloads!")

            # Download the entire COCO split as zip
            if self.download_file(zip_url, zip_file, f"COCO {split} bulk download"):
                logger.info(f"📦 Extracting COCO {split} zip file...")
                if self.extract_archive(zip_file, coco_dir):
                    logger.info(f"✅ Successfully extracted COCO {split}")
                    # Clean up zip file after extraction
                    zip_file.unlink()
                else:
                    logger.error(f"❌ Failed to extract {zip_file}")
                    # Fall back to individual downloads
                    return self._download_coco_images_individually(image_ids, dataset_id)
            else:
                logger.warning(
                    "❌ Bulk download failed, falling back to individual image downloads...")
                return self._download_coco_images_individually(image_ids, dataset_id)
        else:
            logger.info(
                f"✅ COCO {split} directory already exists, skipping bulk download")

        # Count how many of our needed images are present
        downloaded_count = 0
        for image_id in image_ids:
            try:
                padded_id = f"{int(image_id):012d}"
                image_file = extract_dir / f"{padded_id}.jpg"
                if image_file.exists():
                    downloaded_count += 1
            except Exception:
                continue

        logger.info(
            f"✅ Found {downloaded_count}/{len(image_ids)} COCO images in bulk download")

    def _download_coco_images_individually(self, image_ids: List[str], dataset_id: str):
        """Fallback: Download COCO images individually (slower method)"""
        logger.info(
            f"🐌 Fallback: Downloading {len(image_ids)} COCO images individually...")

        coco_dir = self.data_dir / "coco_images"
        coco_dir.mkdir(exist_ok=True)

        # Determine split from dataset_id
        split = "train2017" if "train" in dataset_id else "val2017"

        downloaded_count = 0
        for image_id in tqdm(image_ids, desc=f"Downloading COCO {split}"):
            # COCO image_id needs to be zero-padded to 12 digits
            try:
                padded_id = f"{int(image_id):012d}"
                image_url = f"http://images.cocodataset.org/{split}/{padded_id}.jpg"
                image_file = coco_dir / f"{padded_id}.jpg"

                if not image_file.exists():
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        with open(image_file, 'wb') as f:
                            f.write(response.content)
                        downloaded_count += 1
                    else:
                        logger.debug(
                            f"Failed to download {image_url}: {response.status_code}")
            except Exception as e:
                logger.debug(f"Error downloading COCO image {image_id}: {e}")

        logger.info(f"✅ Downloaded {downloaded_count} COCO images")

    def _download_flickr30k_images_by_ids(self, image_ids: List[str], download_dir):
        """Download Flickr30k images using Flickr API"""
        logger.info(
            f"🖼️ Downloading {len(image_ids)} Flickr30k images using Flickr API...")

        flickr30k_dir = self.data_dir / "flickr30k"
        flickr30k_dir.mkdir(exist_ok=True)

        try:
            import flickrapi

            # You need to get your own API key from https://www.flickr.com/services/apps/create/apply
            api_key = os.environ.get('FLICKR_API_KEY', '')
            api_secret = os.environ.get('FLICKR_API_SECRET', '')

            if not api_key or not api_secret:
                logger.info(
                    "FLICKR_API_KEY and FLICKR_API_SECRET environment variables not set.")
                logger.info("Please set them to enable Flickr30k downloading.")
                logger.info(
                    "Get your API keys from: https://www.flickr.com/services/apps/create/apply")

                # Create placeholder files for now
                for image_id in image_ids:
                    placeholder_path = download_dir / f"{image_id}.jpg"
                    if not placeholder_path.exists():
                        placeholder_path.write_text("")
                return len(image_ids)

            flickr = flickrapi.FlickrAPI(api_key, api_secret, cache=True)
            success_count = 0

            for i, image_id in enumerate(image_ids):
                try:
                    if i % 100 == 0:
                        logger.info(
                            f"Progress: {i}/{len(image_ids)} Flickr30k images")

                    image_path = download_dir / f"{image_id}.jpg"
                    if image_path.exists() and image_path.stat().st_size > 0:
                        success_count += 1
                        continue

                    # Get photo sizes to find the image URL
                    sizes = flickr.photos.getSizes(
                        photo_id=image_id, format='etree')

                    # Find the largest size
                    size_elements = sizes.findall('.//size')
                    if not size_elements:
                        logger.debug(
                            f"No sizes found for Flickr image {image_id}")
                        continue

                    # Get the URL for the largest size
                    largest_size = max(size_elements, key=lambda x: int(
                        x.get('width', 0)) * int(x.get('height', 0)))
                    image_url = largest_size.get('source')

                    if image_url:
                        response = requests.get(image_url, timeout=30)
                        response.raise_for_status()

                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                        success_count += 1

                except Exception as e:
                    logger.debug(
                        f"Failed to download Flickr30k image {image_id}: {e}")
                    # Create placeholder for failed downloads
                    placeholder_path = download_dir / f"{image_id}.jpg"
                    if not placeholder_path.exists():
                        placeholder_path.write_text("")

            logger.info(
                f"✅ Successfully downloaded {success_count}/{len(image_ids)} Flickr30k images")
            return success_count

        except ImportError:
            logger.info(
                "flickrapi library not installed. Install with: pip install flickrapi")
            logger.info("Creating placeholder files for Flickr30k images...")

            # Create placeholder files
            for image_id in image_ids:
                placeholder_path = download_dir / f"{image_id}.jpg"
                if not placeholder_path.exists():
                    placeholder_path.write_text("")
            return len(image_ids)

    def _download_ade20k_images_by_ids(self, image_ids: List[str], download_dir):
        """Download ADE20K images using direct URLs and API endpoints"""
        logger.info(
            f"🖼️ Attempting to download {len(image_ids)} ADE20K images...")

        # ADE20K images are often available through multiple mirrors and patterns
        base_urls = [
            "http://data.csail.mit.edu/places/ADE20K/images/training/",
            "http://data.csail.mit.edu/places/ADE20K/images/validation/",
            "https://data.csail.mit.edu/places/ADE20K/images/training/",
            "https://data.csail.mit.edu/places/ADE20K/images/validation/",
            "http://groups.csail.mit.edu/vision/datasets/ADE20K/images/training/",
            "http://groups.csail.mit.edu/vision/datasets/ADE20K/images/validation/",
        ]

        # Common folder structures in ADE20K
        folder_patterns = [
            "",  # Direct in base
            "urban/street/",
            "urban/bridge/",
            "urban/highway/",
            "rural/forest/",
            "rural/field/",
            "indoor/office/",
            "indoor/bedroom/",
            "indoor/kitchen/",
            "indoor/living_room/",
            "outdoor/playground/",
            "outdoor/park/",
            "outdoor/garden/",
        ]

        success_count = 0

        for i, image_id in enumerate(image_ids):
            try:
                if i % 100 == 0:
                    logger.info(
                        f"Progress: {i}/{len(image_ids)} ADE20K images")

                image_path = download_dir / f"{image_id}.jpg"
                if image_path.exists() and image_path.stat().st_size > 0:
                    success_count += 1
                    continue

                downloaded = False

                # Try different URL patterns and folder structures
                for base_url in base_urls:
                    for folder in folder_patterns:
                        full_url = f"{base_url}{folder}{image_id}.jpg"

                        try:
                            response = requests.get(full_url, timeout=10)
                            # Valid image
                            if response.status_code == 200 and len(response.content) > 1000:
                                with open(image_path, 'wb') as f:
                                    f.write(response.content)
                                success_count += 1
                                downloaded = True
                                logger.debug(
                                    f"Downloaded ADE20K image {image_id} from {full_url}")
                                break
                        except Exception as e:
                            logger.debug(
                                f"Failed to download from {full_url}: {e}")
                            continue

                    if downloaded:
                        break

                if not downloaded:
                    logger.debug(
                        f"Could not download ADE20K image {image_id}, creating placeholder")
                    # Create placeholder for failed downloads
                    with open(image_path, 'w') as f:
                        f.write("")

            except Exception as e:
                logger.debug(f"Error downloading ADE20K image {image_id}: {e}")
                # Create placeholder for failed downloads
                placeholder_path = download_dir / f"{image_id}.jpg"
                if not placeholder_path.exists():
                    placeholder_path.write_text("")

        if success_count == 0:
            logger.info("⚠️ Unable to download ADE20K images automatically.")
            logger.info(
                "📋 Manual download required from: https://ade20k.csail.mit.edu/request_data/")
        else:
            logger.info(
                f"✅ Successfully downloaded {success_count}/{len(image_ids)} ADE20K images")

        return len(image_ids)  # Return total attempted, not just successful

    def download_coco_annotations(self) -> Dict[str, int]:
        """Download COCO annotations"""
        logger.info("📥 Downloading COCO annotations...")

        coco_dir = self.data_dir / "coco"
        coco_dir.mkdir(exist_ok=True)

        annotations_dir = coco_dir / "annotations"

        # Download annotations
        ann_url = self.datasets['coco']['annotations']['train2017']
        ann_zip = coco_dir / "annotations_trainval2017.zip"

        if ann_zip.exists():
            logger.info("✅ COCO annotations already downloaded")
        else:
            success = self.download_file(ann_url, ann_zip, "COCO Annotations")
            if not success:
                return {}

        # Extract annotations
        if not annotations_dir.exists():
            self.extract_archive(ann_zip, coco_dir)

        # Count COCO samples
        dataset_info = {}
        total_samples = 0

        # Count captions in train and val
        for split in ['train2017', 'val2017']:
            captions_file = annotations_dir / f"captions_{split}.json"
            if captions_file.exists():
                try:
                    with open(captions_file, 'r') as f:
                        data = json.load(f)

                    num_annotations = len(data['annotations'])
                    num_images = len(data['images'])

                    dataset_info[f"coco_{split}_captions"] = num_annotations
                    dataset_info[f"coco_{split}_images"] = num_images
                    total_samples += num_annotations

                    logger.info(
                        f"  • COCO {split}: {num_images:,} images, {num_annotations:,} captions")

                except Exception as e:
                    logger.warning(
                        f"Failed to count COCO {split} samples: {e}")

        logger.info(f"✅ COCO total: {total_samples:,} caption samples")
        return dataset_info

    def prepare_unified_dataset(self) -> Tuple[List[str], int]:
        """Prepare unified caption dataset from all sources and optionally cache vision features"""
        logger.info("🔄 Preparing unified dataset...")

        all_captions = []
        all_image_urls = []

        # Process Localized Narratives
        ln_dir = self.data_dir / "localized_narratives"
        if ln_dir.exists():
            for dataset_dir in ln_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    dataset_image_urls = []

                    for jsonl_file in dataset_dir.glob("*.jsonl"):
                        logger.info(f"Processing {jsonl_file}")
                        try:
                            with open(jsonl_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        data = json.loads(line)

                                        # Extract caption/narrative
                                        if 'caption' in data:
                                            all_captions.append(
                                                data['caption'])
                                        elif 'narrative' in data:
                                            all_captions.append(
                                                data['narrative'])

                                        # Extract image path for caching using downloaded local images
                                        if self.cache_vision_features and 'image_id' in data:
                                            image_id = data['image_id']
                                            dataset_id = data.get(
                                                'dataset_id', '')

                                            # Check if we have this image downloaded locally
                                            local_image_path = None

                                            if 'open_images' in dataset_id.lower():
                                                # Check if Open Images file exists
                                                oi_image_path = self.data_dir / \
                                                    "open_images" / \
                                                    f"{image_id}.jpg"
                                                if oi_image_path.exists():
                                                    local_image_path = str(
                                                        oi_image_path)
                                            elif 'coco' in dataset_id.lower():
                                                # Check if COCO file exists (look in extracted train2017/val2017 directory first, then fallback)
                                                split = "train2017" if "train" in dataset_id else "val2017"
                                                coco_extracted_path = self.data_dir / "coco_images" / \
                                                    split / \
                                                    f"{int(image_id):012d}.jpg"
                                                coco_flat_path = self.data_dir / \
                                                    "coco_images" / \
                                                    f"{int(image_id):012d}.jpg"

                                                if coco_extracted_path.exists():
                                                    local_image_path = str(
                                                        coco_extracted_path)
                                                elif coco_flat_path.exists():
                                                    local_image_path = str(
                                                        coco_flat_path)
                                            elif 'flickr30k' in dataset_id.lower():
                                                # Check if Flickr30k file exists
                                                flickr_image_path = self.data_dir / \
                                                    "flickr30k" / \
                                                    f"{image_id}.jpg"
                                                if flickr_image_path.exists():
                                                    local_image_path = str(
                                                        flickr_image_path)
                                            elif 'ade20k' in dataset_id.lower():
                                                # Check if ADE20K file exists
                                                ade_image_path = self.data_dir / \
                                                    "ade20k" / \
                                                    f"{image_id}.jpg"
                                                if ade_image_path.exists():
                                                    local_image_path = str(
                                                        ade_image_path)

                                            # Add local path or None if not found
                                            all_image_urls.append(
                                                local_image_path)
                                            dataset_image_urls.append(
                                                local_image_path)

                        except Exception as e:
                            logger.warning(
                                f"Failed to process {jsonl_file}: {e}")

                    # Cache vision features for this Localized Narratives dataset
                    if self.cache_vision_features and dataset_image_urls:
                        # Remove duplicates
                        unique_dataset_urls = list(set(dataset_image_urls))
                        self._cache_vision_features(
                            unique_dataset_urls, f"localized_narratives_{dataset_name}")

        # Process COCO captions - REMOVED: Only use COCO from Localized Narratives
        # The COCO data in Localized Narratives is sufficient and avoids duplication

        # Save unified captions
        captions_file = self.data_dir / "all_captions.json"
        with open(captions_file, 'w', encoding='utf-8') as f:
            json.dump(all_captions, f, indent=2, ensure_ascii=False)

        logger.info(
            f"✅ Unified dataset created: {len(all_captions):,} captions")
        logger.info(f"   Saved to: {captions_file}")

        if self.cache_vision_features:
            unique_urls = len(set(all_image_urls))
            logger.info(
                f"✅ Vision features processed for {unique_urls:,} unique images")

            # Create unified cache file for training
            self._create_unified_vision_cache(all_image_urls)

        return all_captions, len(all_captions)

    def _create_unified_vision_cache(self, all_image_urls: List[str]):
        """Create unified vision cache file that training can use directly"""
        logger.info("🔄 Creating unified vision cache for training...")

        # Collect all cached features in the same order as captions
        unified_features = []
        feature_type = 'dummy'  # Default

        for i, image_url in enumerate(all_image_urls):
            if image_url is None:
                # No image URL available, use dummy
                features = np.random.randn(768).astype(np.float32)
            else:
                # Look for cached feature for this image
                found_feature = False

                # Try different dataset cache keys
                for dataset_name in ['coco', 'localized_narratives_coco', 'localized_narratives_open_images']:
                    cache_key = self._generate_cache_key(
                        image_url, dataset_name)
                    cache_file = self.vision_cache_dir / f"{cache_key}.npy"
                    metadata_file = self.vision_cache_dir / f"{cache_key}.pkl"

                    if cache_file.exists():
                        try:
                            features = np.load(cache_file)

                            # Load metadata to get feature type
                            if metadata_file.exists():
                                with open(metadata_file, 'rb') as f:
                                    metadata = pickle.load(f)
                                feature_type = metadata.get(
                                    'feature_type', 'unknown')

                            found_feature = True
                            break
                        except Exception as e:
                            logger.debug(
                                f"Failed to load cache {cache_file}: {e}")
                            continue

                if not found_feature:
                    # No cached feature found, create dummy
                    features = np.random.randn(768).astype(np.float32)

            unified_features.append(features)

        # Convert to numpy array
        unified_features_array = np.array(unified_features)

        # Save unified cache
        unified_cache_file = self.vision_cache_dir / "all_features.npy"
        unified_metadata_file = self.vision_cache_dir / "cache_metadata.pkl"

        np.save(unified_cache_file, unified_features_array)

        # Save metadata
        metadata = {
            'num_samples': len(unified_features_array),
            'feature_type': feature_type,
            'feature_shape': (768,),
            'created_by': 'download_script',
            'use_real_vision': self.use_real_vision
        }

        with open(unified_metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(
            f"✅ Created unified vision cache: {len(unified_features_array):,} features")
        logger.info(f"   Saved to: {unified_cache_file}")
        logger.info(f"   Feature type: {feature_type}")

    def download_all(self, download_localized_narratives: bool = True) -> Dict[str, int]:
        """Download Localized Narratives dataset and return statistics"""
        logger.info("🚀 Starting multimodal dataset download...")

        total_stats = {}

        # Download Localized Narratives (contains both Open Images and COCO data)
        if download_localized_narratives:
            logger.info("📥 Downloading Localized Narratives...")
            ln_stats = self.download_localized_narratives()
            total_stats.update(ln_stats)
        else:
            logger.info("⏭️  Skipping Localized Narratives download")
            logger.warning("⚠️  No datasets selected for download!")
            return {}

        # Prepare unified dataset from Localized Narratives
        all_captions, total_captions = self.prepare_unified_dataset()
        total_stats['total_unified_captions'] = total_captions

        # Print summary
        logger.info("📊 Dataset Download Summary:")
        total_samples = 0
        for key, count in total_stats.items():
            logger.info(f"  • {key}: {count:,}")
            total_samples += count

        logger.info(f"🎯 TOTAL IMAGE-CAPTION PAIRS: {total_samples:,}")

        return total_stats


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download multimodal datasets with optional vision feature caching")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to download data to")
    parser.add_argument("--skip_images", action="store_true",
                        help="Skip downloading actual images (annotations only)")

    # Dataset selection arguments
    dataset_group = parser.add_argument_group(
        "Dataset Selection", "Localized Narratives dataset (includes Open Images and COCO)")
    dataset_group.add_argument("--dataset", type=str, choices=['localized_narratives'], default='localized_narratives',
                               help="Dataset to download: 'localized_narratives' (~1.16M samples with Open Images + COCO)")
    dataset_group.add_argument("--max_samples", type=int, default=50000,
                               help="Maximum number of samples to download (default: 50000, use 0 or -1 for unlimited)")
    dataset_group.add_argument("--download_full_openimages", action="store_true",
                               help="Download the entire OpenImages dataset overnight (removes 50k sample limit)")

    # HuggingFace options
    hf_group = parser.add_argument_group(
        "HuggingFace Options", "Use HuggingFace LocalizedNarratives instead of downloading")
    hf_group.add_argument("--use_hf_localized_narratives", action="store_true", default=False,
                          help="Use HuggingFace LocalizedNarratives dataset (deprecated scripts - may not work)")
    hf_group.add_argument("--no_hf_localized_narratives", action="store_true",
                          help="Force disable HuggingFace and download original LocalizedNarratives")
    hf_group.add_argument("--hf_dataset_config", type=str, default="open_images",
                          choices=["open_images", "ait", "flickr", "coco"],
                          help="HuggingFace LocalizedNarratives config (default: open_images)")

    # Vision feature caching arguments
    vision_group = parser.add_argument_group(
        "Vision Features Caching", "Pre-cache vision features during download")
    vision_group.add_argument("--cache_vision_features", action="store_true",
                              help="Enable vision features caching during download")
    vision_group.add_argument("--real_vision_features", action="store_true",
                              help="Extract real DiNOv2 features (slower but better quality). Default uses dummy features")

    args = parser.parse_args()

    # Dataset is required and defaults to localized_narratives
    logger.info("📄 BitGen Localized Narratives Dataset Downloader")
    logger.info(
        "ℹ️  This will download Localized Narratives containing both Open Images and COCO data")
    logger.info("")

    try:
        # Handle HuggingFace options
        use_hf = args.use_hf_localized_narratives and not args.no_hf_localized_narratives

        # Force dummy vision to avoid URL issues unless explicitly requested
        use_real_vision = args.real_vision_features
        if not use_real_vision:
            logger.info(
                "🎨 Using dummy vision features to avoid URL download issues")

        # Handle full dataset download options
        max_samples = args.max_samples
        if args.download_full_openimages or max_samples <= 0:
            max_samples = 0  # Unlimited
            logger.info("🌍 FULL OPENIMAGES DOWNLOAD: No sample limit - downloading entire dataset overnight!")
            logger.info("⚠️  This will take several hours and require significant disk space")
        else:
            logger.info(f"📊 Sample limit: {max_samples:,} samples")

        # Create downloader with vision caching options
        downloader = MultimodalDatasetDownloader(
            data_dir=args.data_dir,
            cache_vision_features=args.cache_vision_features,
            use_real_vision=use_real_vision,
            use_hf_localized_narratives=use_hf,
            hf_dataset_config=args.hf_dataset_config,
            max_samples=max_samples
        )

        # Determine which datasets to download
        if args.dataset == 'localized_narratives':
            if max_samples <= 0:
                logger.info("🎯 Downloading ENTIRE Localized Narratives dataset")
                logger.info("📊 Estimated download: ~1.16M+ image-caption pairs (Full OpenImages + COCO)")
                logger.info("💾 Expected size: ~100+ GB (images) + ~10 GB (features)")
                logger.info("⏰ Estimated time: 6-12 hours depending on connection")
            else:
                logger.info("🎯 Downloading LIMITED Localized Narratives dataset")
                logger.info(f"📊 Sample limit: {max_samples:,} image-caption pairs")
            logger.info("ℹ️  This includes both Open Images and COCO data from Localized Narratives")
        else:
            logger.error(
                "❌ Only 'localized_narratives' dataset is supported in this version")
            return 1

        if args.cache_vision_features:
            if args.real_vision_features:
                logger.info(
                    "🔄 Will extract and cache REAL DiNOv2 vision features (slower, better quality)")
                logger.info(
                    "   ⚠️  This requires GPU and internet connection for each image")
            else:
                logger.info(
                    "🔄 Will generate and cache DUMMY vision features (faster, for development)")

        # Download Localized Narratives dataset
        stats = downloader.download_all()

        if stats:
            logger.info("✅ Download completed successfully!")
            logger.info("📁 Data structure:")
            logger.info(f"  📂 {args.data_dir}/")
            logger.info(
                f"    📄 all_captions.json ({stats.get('total_unified_captions', 0):,} captions)")
            logger.info(f"    📂 localized_narratives/")
            if args.cache_vision_features:
                logger.info(f"    📂 vision_features_cache/ (cached features)")

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
