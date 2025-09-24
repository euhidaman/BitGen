"""
COCO Dataset Downloader for BitGen
Simple cross-platform dataset downloader without Pi dependencies
"""

import os
import json
import requests
import zipfile
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time

class COCODownloader:
    """Simple COCO dataset downloader"""

    def __init__(self, output_dir: str = "data/coco"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_from_kaggle(self) -> bool:
        """Download COCO dataset from Kaggle"""
        try:
            import kaggle

            self.logger.info("Downloading COCO dataset from Kaggle...")

            # Download using Kaggle API
            kaggle.api.dataset_download_files(
                'awsaf49/coco-2017-dataset',
                path=str(self.output_dir),
                unzip=True
            )

            self.logger.info("✅ COCO dataset downloaded successfully")
            return True

        except ImportError:
            self.logger.error("Kaggle API not available. Install with: pip install kaggle")
            return False
        except Exception as e:
            self.logger.error(f"Failed to download from Kaggle: {e}")
            return False

    def download_sample_data(self) -> bool:
        """Download a small sample of COCO data for testing"""
        try:
            self.logger.info("Creating sample COCO dataset for testing...")

            # Create sample data structure
            sample_data = {
                "images": [
                    {
                        "id": 1,
                        "file_name": "sample_001.jpg",
                        "width": 640,
                        "height": 480
                    },
                    {
                        "id": 2,
                        "file_name": "sample_002.jpg",
                        "width": 640,
                        "height": 480
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "caption": "A robot arm picking up a red cube from a table"
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "caption": "Industrial robot performing pick and place operation"
                    },
                    {
                        "id": 3,
                        "image_id": 2,
                        "caption": "Mobile robot navigating through a warehouse corridor"
                    },
                    {
                        "id": 4,
                        "image_id": 2,
                        "caption": "Autonomous robot avoiding obstacles while moving"
                    }
                ]
            }

            # Save sample data
            sample_file = self.output_dir / "sample_coco.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_data, f, indent=2)

            self.logger.info(f"✅ Sample COCO data created: {sample_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create sample data: {e}")
            return False

    def process_dataset(self):
        """Process downloaded dataset for BitGen training"""
        try:
            self.logger.info("Processing COCO dataset for BitGen...")

            # Look for COCO annotation files
            coco_files = list(self.output_dir.rglob("*.json"))

            if not coco_files:
                # Create sample data if no COCO files found
                self.logger.info("No COCO files found, creating sample data...")
                self.download_sample_data()
                return

            # Process the first valid COCO file found
            for coco_file in coco_files:
                if self._process_coco_file(coco_file):
                    break

            self.logger.info("✅ Dataset processing completed")

        except Exception as e:
            self.logger.error(f"Dataset processing failed: {e}")

    def _process_coco_file(self, coco_file: Path) -> bool:
        """Process a single COCO annotation file"""
        try:
            with open(coco_file, 'r') as f:
                data = json.load(f)

            # Extract images and annotations
            images = data.get('images', [])
            annotations = data.get('annotations', [])

            if not images or not annotations:
                return False

            # Create processed dataset
            processed_data = []

            for img in images[:100]:  # Limit to first 100 images for quick testing
                img_id = img['id']

                # Find captions for this image
                img_captions = [ann['caption'] for ann in annotations
                              if ann.get('image_id') == img_id and 'caption' in ann]

                if img_captions:
                    for caption in img_captions[:2]:  # Max 2 captions per image
                        processed_data.append({
                            'image_id': img_id,
                            'image_file': img.get('file_name', f"image_{img_id}.jpg"),
                            'caption': caption,
                            'width': img.get('width', 640),
                            'height': img.get('height', 480)
                        })

            # Save processed data
            output_file = self.output_dir / "validated_coco.json"
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)

            self.logger.info(f"✅ Processed {len(processed_data)} samples to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {coco_file}: {e}")
            return False

    def validate_dataset(self) -> bool:
        """Validate that dataset is ready for training"""
        validation_file = self.output_dir / "validated_coco.json"

        if not validation_file.exists():
            self.logger.error("No validated dataset found. Run download and process first.")
            return False

        try:
            with open(validation_file, 'r') as f:
                data = json.load(f)

            if len(data) < 10:
                self.logger.warning(f"Dataset only has {len(data)} samples. Consider downloading more data.")

            self.logger.info(f"✅ Dataset validated: {len(data)} training samples available")
            return True

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False

def download_and_prepare_coco(output_dir: str = "data/coco"):
    """Main function to download and prepare COCO dataset"""

    downloader = COCODownloader(output_dir)

    print("📥 BitGen COCO Dataset Setup")
    print("=" * 40)

    # Try to download from Kaggle first
    if downloader.download_from_kaggle():
        print("✅ Downloaded COCO dataset from Kaggle")
    else:
        print("ℹ️ Kaggle download failed, creating sample dataset...")
        if downloader.download_sample_data():
            print("✅ Created sample dataset for testing")
        else:
            print("❌ Failed to create sample dataset")
            return False

    # Process the dataset
    downloader.process_dataset()

    # Validate the result
    if downloader.validate_dataset():
        print("🎯 Dataset is ready for BitGen training!")
        return True
    else:
        print("❌ Dataset validation failed")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download COCO dataset for BitGen")
    parser.add_argument("--output_dir", type=str, default="data/coco",
                       help="Output directory for dataset")

    args = parser.parse_args()

    success = download_and_prepare_coco(args.output_dir)

    if success:
        print(f"\n🚀 Ready to train! Use:")
        print(f"python bitgen_cli.py train --coco_data {args.output_dir}/validated_coco.json")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
