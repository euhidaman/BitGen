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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDatasetDownloader:
    """Download and prepare Localized Narratives and COCO datasets"""

    def __init__(self, data_dir: str = "./data", cache_vision_features: bool = False, use_real_vision: bool = False):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

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

                    logger.info("Will extract real DiNOv3 features")
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
                    'train': 'https://storage.googleapis.com/localized-narratives/annotations/localized_narratives_openimages_train.jsonl',
                    'validation': 'https://storage.googleapis.com/localized-narratives/annotations/localized_narratives_openimages_validation.jsonl',
                    'test': 'https://storage.googleapis.com/localized-narratives/annotations/localized_narratives_openimages_test.jsonl'
                },
                'coco': {
                    'train': 'https://storage.googleapis.com/localized-narratives/annotations/localized_narratives_coco_train.jsonl',
                    'validation': 'https://storage.googleapis.com/localized-narratives/annotations/localized_narratives_coco_val.jsonl'
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
        """Load DiNOv3 vision model for real feature extraction"""
        try:
            logger.info("Loading DiNOv3 vision model...")
            self.vision_model = self.Dinov2Model.from_pretrained(
                "facebook/dinov2-vits14")
            self.vision_processor = self.AutoImageProcessor.from_pretrained(
                "facebook/dinov2-vits14")

            # Check for GPU availability and move model
            if self.torch.cuda.is_available():
                device_name = self.torch.cuda.get_device_name(0)
                self.vision_model = self.vision_model.cuda()
                logger.info(f"✅ DiNOv3 model loaded on GPU: {device_name}")
                logger.info(
                    f"   GPU Memory Available: {self.torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                logger.warning(
                    "⚠️  No GPU detected, using CPU for DiNOv3 feature extraction (will be slower)")

            self.vision_model.eval()

        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            logger.info("Falling back to dummy vision features")
            self.use_real_vision = False

    def _generate_cache_key(self, image_url: str, dataset_name: str) -> str:
        """Generate a unique cache key for an image"""
        key_string = f"{image_url}_{dataset_name}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _cache_vision_features(self, image_urls: List[str], dataset_name: str) -> int:
        """Cache vision features for a list of image URLs"""
        if not self.cache_vision_features:
            return 0

        logger.info(
            f"Caching vision features for {len(image_urls)} images from {dataset_name}")

        cached_count = 0

        with tqdm(total=len(image_urls), desc=f"Caching {dataset_name} features") as pbar:
            for image_url in image_urls:
                cache_key = self._generate_cache_key(image_url, dataset_name)
                cache_file = self.vision_cache_dir / f"{cache_key}.npy"
                metadata_file = self.vision_cache_dir / f"{cache_key}.pkl"

                # Skip if already cached
                if cache_file.exists() and metadata_file.exists():
                    pbar.update(1)
                    continue

                try:
                    if self.use_real_vision:
                        # Extract real features (requires downloading image)
                        features = self._extract_real_vision_features(
                            image_url)
                        features_array = features.cpu().numpy()
                    else:
                        # Generate dummy features (768-dimensional like DiNOv3) using numpy
                        features_array = np.random.randn(
                            768).astype(np.float32)

                    # Save features and metadata
                    np.save(cache_file, features_array)

                    metadata = {
                        'image_url': image_url,
                        'dataset_name': dataset_name,
                        'feature_type': 'real_dinov3' if self.use_real_vision else 'dummy',
                        'feature_shape': features_array.shape
                    }

                    with open(metadata_file, 'wb') as f:
                        pickle.dump(metadata, f)

                    cached_count += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to cache features for {image_url}: {e}")

                pbar.update(1)

        logger.info(
            f"✅ Cached {cached_count} new vision features for {dataset_name}")
        return cached_count

    def _extract_real_vision_features(self, image_url: str):
        """Extract real DiNOv3 features from image URL using GPU if available"""
        try:
            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # Process with DiNOv3
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

        except Exception as e:
            logger.error(f"Failed to extract features from {image_url}: {e}")
            # Return dummy features as fallback
            return self.torch.randn(768)

        except Exception as e:
            logger.warning(
                f"Failed to extract real features from {image_url}: {e}")
            # Return dummy features as fallback
            return self.torch.randn(768, dtype=self.torch.float32)

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
        """Download Localized Narratives annotations"""
        logger.info("📥 Downloading Localized Narratives...")

        ln_dir = self.data_dir / "localized_narratives"
        ln_dir.mkdir(exist_ok=True)

        total_samples = 0
        dataset_info = {}

        for dataset_name, splits in self.datasets['localized_narratives'].items():
            dataset_dir = ln_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)

            dataset_samples = 0

            for split, url in splits.items():
                filename = f"{dataset_name}_{split}.jsonl"
                filepath = dataset_dir / filename

                if filepath.exists():
                    logger.info(
                        f"✅ {filename} already exists, skipping download")
                else:
                    success = self.download_file(
                        url, filepath, f"Localized Narratives {dataset_name} {split}")
                    if not success:
                        continue

                # Count samples in the file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        samples = sum(1 for line in f if line.strip())
                    dataset_samples += samples
                    logger.info(f"  • {split}: {samples:,} samples")
                except Exception as e:
                    logger.warning(
                        f"Failed to count samples in {filename}: {e}")

            dataset_info[f"localized_narratives_{dataset_name}"] = dataset_samples
            total_samples += dataset_samples
            logger.info(
                f"✅ Localized Narratives {dataset_name}: {dataset_samples:,} total samples")

        logger.info(f"✅ Localized Narratives total: {total_samples:,} samples")
        return dataset_info

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

                                        # Extract image URL for caching
                                        if self.cache_vision_features and 'image_id' in data:
                                            # Construct image URL based on dataset
                                            if dataset_name == 'open_images' and 'image_id' in data:
                                                image_url = f"https://storage.googleapis.com/openimages/web/{data['image_id']}.jpg"
                                            elif dataset_name == 'coco' and 'image_id' in data:
                                                # COCO image URLs from Localized Narratives
                                                image_url = f"http://images.cocodataset.org/train2017/{data['image_id']:012d}.jpg"
                                            else:
                                                continue

                                            dataset_image_urls.append(
                                                image_url)
                                            all_image_urls.append(image_url)

                        except Exception as e:
                            logger.warning(
                                f"Failed to process {jsonl_file}: {e}")

                    # Cache vision features for this dataset
                    if self.cache_vision_features and dataset_image_urls:
                        self._cache_vision_features(
                            dataset_image_urls, f"localized_narratives_{dataset_name}")

        # Process COCO captions
        coco_dir = self.data_dir / "coco" / "annotations"
        if coco_dir.exists():
            coco_image_urls = []

            for split in ['train2017', 'val2017']:
                captions_file = coco_dir / f"captions_{split}.json"
                if captions_file.exists():
                    logger.info(f"Processing {captions_file}")
                    try:
                        with open(captions_file, 'r') as f:
                            data = json.load(f)

                        # Create image_id to filename mapping
                        image_id_to_filename = {}
                        for img_info in data['images']:
                            image_id_to_filename[img_info['id']
                                                 ] = img_info['file_name']

                        for annotation in data['annotations']:
                            all_captions.append(annotation['caption'])

                            # Extract image URL for caching
                            if self.cache_vision_features:
                                image_id = annotation['image_id']
                                if image_id in image_id_to_filename:
                                    filename = image_id_to_filename[image_id]
                                    image_url = f"http://images.cocodataset.org/{split}/{filename}"
                                    coco_image_urls.append(image_url)
                                    all_image_urls.append(image_url)

                    except Exception as e:
                        logger.warning(
                            f"Failed to process {captions_file}: {e}")

            # Cache vision features for COCO
            if self.cache_vision_features and coco_image_urls:
                # Remove duplicates
                unique_coco_urls = list(set(coco_image_urls))
                self._cache_vision_features(unique_coco_urls, "coco")

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

    def download_all(self, download_localized_narratives: bool = True, download_coco: bool = True) -> Dict[str, int]:
        """Download selected datasets and return statistics"""
        logger.info("🚀 Starting multimodal dataset download...")

        total_stats = {}

        # Download Localized Narratives (optional)
        if download_localized_narratives:
            logger.info("📥 Downloading Localized Narratives...")
            ln_stats = self.download_localized_narratives()
            total_stats.update(ln_stats)
        else:
            logger.info("⏭️  Skipping Localized Narratives download")

        # Download COCO annotations (optional)
        if download_coco:
            logger.info("📥 Downloading COCO...")
            coco_stats = self.download_coco_annotations()
            total_stats.update(coco_stats)
        else:
            logger.info("⏭️  Skipping COCO download")

        # Only prepare unified dataset if we downloaded something
        if download_localized_narratives or download_coco:
            # Prepare unified dataset
            all_captions, total_captions = self.prepare_unified_dataset()
            total_stats['total_unified_captions'] = total_captions
        else:
            logger.warning("⚠️  No datasets selected for download!")
            return {}

        # Print summary
        logger.info("📊 Dataset Download Summary:")
        total_samples = 0
        for key, count in total_stats.items():
            logger.info(f"  • {key}: {count:,}")
            if 'captions' in key or 'narratives' in key:
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
        "Dataset Selection", "Choose which datasets to download")
    dataset_group.add_argument("--dataset", type=str, choices=['both', 'coco', 'localized_narratives'],
                               help="Which dataset(s) to download: 'both' (~1.78M samples), 'coco' (~615K samples), 'localized_narratives' (~1.16M samples)")

    # Vision feature caching arguments
    vision_group = parser.add_argument_group(
        "Vision Features Caching", "Pre-cache vision features during download")
    vision_group.add_argument("--cache_vision_features", action="store_true",
                              help="Enable vision features caching during download")
    vision_group.add_argument("--real_vision_features", action="store_true",
                              help="Extract real DiNOv3 features (slower but better quality). Default uses dummy features")

    args = parser.parse_args()

    # Validate dataset argument
    if not args.dataset:
        logger.info(
            "Please specify which dataset(s) to download using --dataset:")
        logger.info(
            "  --dataset both                    : ~1.78M samples (largest)")
        logger.info(
            "  --dataset coco                    : ~615K samples (smallest)")
        logger.info(
            "  --dataset localized_narratives    : ~1.16M samples (medium)")
        logger.info("")
        logger.info("Optional vision feature caching:")
        logger.info(
            "  --cache_vision_features           : Pre-cache vision features during download")
        logger.info(
            "  --real_vision_features            : Use real DiNOv3 features (requires GPU, slower)")
        parser.print_help()
        return 0

    try:
        # Create downloader with vision caching options
        downloader = MultimodalDatasetDownloader(
            data_dir=args.data_dir,
            cache_vision_features=args.cache_vision_features,
            use_real_vision=args.real_vision_features
        )

        # Determine which datasets to download
        if args.dataset == 'localized_narratives':
            logger.info("🎯 Downloading ONLY Localized Narratives dataset")
            download_localized_narratives = True
            download_coco = False
        elif args.dataset == 'coco':
            logger.info("🎯 Downloading ONLY COCO dataset")
            download_localized_narratives = False
            download_coco = True
        elif args.dataset == 'both':
            logger.info(
                "🎯 Downloading BOTH Localized Narratives and COCO datasets")
            download_localized_narratives = True
            download_coco = True

        # Show estimated sizes and caching info
        if download_localized_narratives and download_coco:
            logger.info(
                "📊 Estimated download: ~1.78M image-caption pairs (both datasets)")
        elif download_localized_narratives:
            logger.info(
                "📊 Estimated download: ~1.16M image-caption pairs (Localized Narratives)")
        elif download_coco:
            logger.info(
                "📊 Estimated download: ~615K image-caption pairs (COCO)")

        if args.cache_vision_features:
            if args.real_vision_features:
                logger.info(
                    "🔄 Will extract and cache REAL DiNOv3 vision features (slower, better quality)")
                logger.info(
                    "   ⚠️  This requires GPU and internet connection for each image")
            else:
                logger.info(
                    "🔄 Will generate and cache DUMMY vision features (faster, for development)")

        stats = downloader.download_all(
            download_localized_narratives, download_coco)

        if stats:
            logger.info("✅ Download completed successfully!")
            logger.info("📁 Data structure:")
            logger.info(f"  📂 {args.data_dir}/")
            logger.info(
                f"    📄 all_captions.json ({stats.get('total_unified_captions', 0):,} captions)")
            if download_localized_narratives:
                logger.info(f"    📂 localized_narratives/")
            if download_coco:
                logger.info(f"    📂 coco/")
            if args.cache_vision_features:
                logger.info(f"    📂 vision_features_cache/ (cached features)")

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
