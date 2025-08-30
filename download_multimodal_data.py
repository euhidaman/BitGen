#!/usr/bin/env python3
"""
Download and prepare Localized Narratives and COCO datasets
Creates image-caption pairs for multimodal training
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDatasetDownloader:
    """Download and prepare Localized Narratives and COCO datasets"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

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
                logger.error(f"Unsupported archive format: {archive_path.suffix}")
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
                    logger.info(f"✅ {filename} already exists, skipping download")
                else:
                    success = self.download_file(url, filepath, f"Localized Narratives {dataset_name} {split}")
                    if not success:
                        continue

                # Count samples in the file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        samples = sum(1 for line in f if line.strip())
                    dataset_samples += samples
                    logger.info(f"  • {split}: {samples:,} samples")
                except Exception as e:
                    logger.warning(f"Failed to count samples in {filename}: {e}")

            dataset_info[f"localized_narratives_{dataset_name}"] = dataset_samples
            total_samples += dataset_samples
            logger.info(f"✅ Localized Narratives {dataset_name}: {dataset_samples:,} total samples")

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

                    logger.info(f"  • COCO {split}: {num_images:,} images, {num_annotations:,} captions")

                except Exception as e:
                    logger.warning(f"Failed to count COCO {split} samples: {e}")

        logger.info(f"✅ COCO total: {total_samples:,} caption samples")
        return dataset_info

    def prepare_unified_dataset(self) -> Tuple[List[str], int]:
        """Prepare unified caption dataset from all sources"""
        logger.info("🔄 Preparing unified dataset...")

        all_captions = []

        # Process Localized Narratives
        ln_dir = self.data_dir / "localized_narratives"
        if ln_dir.exists():
            for dataset_dir in ln_dir.iterdir():
                if dataset_dir.is_dir():
                    for jsonl_file in dataset_dir.glob("*.jsonl"):
                        logger.info(f"Processing {jsonl_file}")
                        try:
                            with open(jsonl_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        data = json.loads(line)
                                        if 'caption' in data:
                                            all_captions.append(data['caption'])
                                        elif 'narrative' in data:
                                            all_captions.append(data['narrative'])
                        except Exception as e:
                            logger.warning(f"Failed to process {jsonl_file}: {e}")

        # Process COCO captions
        coco_dir = self.data_dir / "coco" / "annotations"
        if coco_dir.exists():
            for split in ['train2017', 'val2017']:
                captions_file = coco_dir / f"captions_{split}.json"
                if captions_file.exists():
                    logger.info(f"Processing {captions_file}")
                    try:
                        with open(captions_file, 'r') as f:
                            data = json.load(f)

                        for annotation in data['annotations']:
                            all_captions.append(annotation['caption'])

                    except Exception as e:
                        logger.warning(f"Failed to process {captions_file}: {e}")

        # Save unified captions
        captions_file = self.data_dir / "all_captions.json"
        with open(captions_file, 'w', encoding='utf-8') as f:
            json.dump(all_captions, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Unified dataset created: {len(all_captions):,} captions")
        logger.info(f"   Saved to: {captions_file}")

        return all_captions, len(all_captions)

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
    parser = argparse.ArgumentParser(description="Download multimodal datasets")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory to download data to")
    parser.add_argument("--skip_images", action="store_true",
                       help="Skip downloading actual images (annotations only)")

    # Dataset selection arguments
    dataset_group = parser.add_argument_group("Dataset Selection", "Choose which datasets to download")
    dataset_group.add_argument("--localized-narratives-only", action="store_true",
                              help="Download only Localized Narratives dataset (~1.16M samples)")
    dataset_group.add_argument("--coco-only", action="store_true",
                              help="Download only COCO dataset (~615K samples)")
    dataset_group.add_argument("--both", action="store_true",
                              help="Download both datasets (~1.78M total samples)")

    args = parser.parse_args()

    # Validate arguments
    selected_options = [args.localized_narratives_only, args.coco_only, args.both]
    if sum(selected_options) > 1:
        logger.error("❌ Please select only one dataset option")
        parser.print_help()
        return 1

    try:
        downloader = MultimodalDatasetDownloader(args.data_dir)

        # Determine which datasets to download based on arguments
        if args.localized_narratives_only:
            logger.info("🎯 Downloading ONLY Localized Narratives dataset")
            download_localized_narratives = True
            download_coco = False
        elif args.coco_only:
            logger.info("🎯 Downloading ONLY COCO dataset")
            download_localized_narratives = False
            download_coco = True
        elif args.both:
            logger.info("🎯 Downloading BOTH Localized Narratives and COCO datasets")
            download_localized_narratives = True
            download_coco = True
        else:
            # Default behavior: show help and ask user to choose
            logger.info("Please specify which dataset(s) to download:")
            logger.info("  --localized-narratives-only  : ~1.16M samples (smaller)")
            logger.info("  --coco-only                  : ~615K samples (smallest)")
            logger.info("  --both                       : ~1.78M samples (largest)")
            parser.print_help()
            return 0

        # Show estimated sizes
        if download_localized_narratives and download_coco:
            logger.info("📊 Estimated download: ~1.78M image-caption pairs (both datasets)")
        elif download_localized_narratives:
            logger.info("📊 Estimated download: ~1.16M image-caption pairs (Localized Narratives)")
        elif download_coco:
            logger.info("📊 Estimated download: ~615K image-caption pairs (COCO)")

        stats = downloader.download_all(download_localized_narratives, download_coco)

        if stats:
            logger.info("✅ Download completed successfully!")
            logger.info("📁 Data structure:")
            logger.info(f"  📂 {args.data_dir}/")
            logger.info(f"    📄 all_captions.json ({stats.get('total_unified_captions', 0):,} captions)")
            if download_localized_narratives:
                logger.info(f"    📂 localized_narratives/")
            if download_coco:
                logger.info(f"    📂 coco/")

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
