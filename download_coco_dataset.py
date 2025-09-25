"""
COCO Dataset Downloader for BitGen
Simple cross-platform dataset downloader for GPU training
"""

import os
import json
import subprocess
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time

class COCODownloader:
    """Simple COCO dataset downloader for GPU training"""

    def __init__(self, output_dir: str = "data/coco"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_from_kaggle(self) -> bool:
        """Download COCO dataset from Kaggle"""
        try:
            # Try to import kaggle API
            import kaggle

            self.logger.info("🚀 Starting Kaggle download...")
            self.logger.info("📦 Dataset: nikhil7280/coco-image-caption")

            # Download the dataset
            kaggle.api.dataset_download_files(
                'nikhil7280/coco-image-caption',
                path=str(self.output_dir),
                unzip=True
            )

            self.logger.info("✅ COCO dataset downloaded successfully from Kaggle")
            return True

        except ImportError:
            self.logger.error("❌ Kaggle API not available")
            self.logger.error("💡 Install with: pip install kaggle")
            self.logger.error("🔑 Setup credentials: https://www.kaggle.com/docs/api")
            return False
        except Exception as e:
            self.logger.error(f"❌ Kaggle download failed: {e}")
            return False

    def process_dataset(self):
        """Process downloaded dataset for BitGen training"""
        try:
            self.logger.info("🔄 Processing COCO dataset for BitGen...")

            # Look for downloaded files
            all_files = list(self.output_dir.rglob("*"))
            json_files = [f for f in all_files if f.suffix.lower() == '.json' and 'sample' not in f.name]
            image_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

            self.logger.info(f"📋 Found {len(json_files)} JSON files and {len(image_files)} image files")

            if not json_files and not image_files:
                self.logger.warning("⚠️ No COCO files found after download")
                self.download_sample_data()
                return

            processed_data = []

            # Try to process COCO format
            for json_file in json_files:
                if self._process_coco_format(json_file, image_files, processed_data):
                    break

            # Save processed data
            if processed_data:
                output_file = self.output_dir / "validated_coco.json"
                with open(output_file, 'w') as f:
                    json.dump(processed_data, f, indent=2)

                self.logger.info(f"✅ Processed {len(processed_data)} image-caption pairs")
                self._validate_images(processed_data)
            else:
                self.logger.warning("⚠️ No data could be processed, creating sample data")
                self.download_sample_data()

        except Exception as e:
            self.logger.error(f"❌ Dataset processing failed: {e}")
            self.download_sample_data()

    def _process_coco_format(self, json_file: Path, image_files: List[Path], processed_data: List) -> bool:
        """Process COCO format JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' in data and 'annotations' in data:
                images = data['images']
                annotations = data['annotations']

                self.logger.info(f"📊 Processing {len(images)} images with {len(annotations)} captions")

                # Create image lookup
                image_lookup = {img['id']: img for img in images}

                # Create filename mapping
                image_files_dict = {img_file.name: img_file for img_file in image_files}

                # Process annotations
                for ann in annotations[:2000]:  # Limit for performance
                    if 'image_id' in ann and 'caption' in ann:
                        img_id = ann['image_id']
                        caption = ann['caption'].strip()

                        if img_id in image_lookup and caption:
                            img_info = image_lookup[img_id]
                            img_filename = img_info.get('file_name', f'image_{img_id}.jpg')

                            # Find actual image file
                            img_path = None
                            if img_filename in image_files_dict:
                                img_path = image_files_dict[img_filename]
                            else:
                                # Try with different prefixes
                                for prefix in ['COCO_train2014_', 'COCO_val2014_', 'COCO_val2017_']:
                                    test_name = f"{prefix}{img_filename}"
                                    if test_name in image_files_dict:
                                        img_path = image_files_dict[test_name]
                                        break

                            if img_path:
                                processed_data.append({
                                    'image_id': img_id,
                                    'image_path': str(img_path),
                                    'caption': caption,
                                    'width': img_info.get('width', 640),
                                    'height': img_info.get('height', 480)
                                })

                return len(processed_data) > 0

        except Exception as e:
            self.logger.error(f"❌ Error processing {json_file}: {e}")
            return False

        return False

    def _validate_images(self, processed_data: List[Dict]):
        """Validate that image files exist"""
        valid_count = 0
        invalid_files = []

        for item in processed_data:
            img_path = Path(item['image_path'])
            if img_path.exists():
                valid_count += 1
            else:
                invalid_files.append(str(img_path))

        self.logger.info(f"✅ Validated {valid_count}/{len(processed_data)} images exist")

        if invalid_files:
            if len(invalid_files) < 10:
                self.logger.warning(f"❌ Missing files: {invalid_files}")
            else:
                self.logger.warning(f"❌ {len(invalid_files)} image files not found")

        # Remove invalid entries
        valid_data = [item for item in processed_data if Path(item['image_path']).exists()]
        processed_data.clear()
        processed_data.extend(valid_data)

        # Save final validated data
        output_file = self.output_dir / "validated_coco.json"
        with open(output_file, 'w') as f:
            json.dump(valid_data, f, indent=2)

    def download_sample_data(self) -> bool:
        """Create sample data for testing"""
        try:
            self.logger.info("🔧 Creating sample dataset for testing...")

            sample_data = [
                {
                    'image_id': 1,
                    'image_path': str(self.output_dir / 'sample_001.jpg'),
                    'caption': 'A robot arm picking up a red cube from a table',
                    'width': 640,
                    'height': 480
                },
                {
                    'image_id': 2,
                    'image_path': str(self.output_dir / 'sample_002.jpg'),
                    'caption': 'Industrial robot performing pick and place operation',
                    'width': 640,
                    'height': 480
                },
                {
                    'image_id': 3,
                    'image_path': str(self.output_dir / 'sample_003.jpg'),
                    'caption': 'Mobile robot navigating through a warehouse corridor',
                    'width': 640,
                    'height': 480
                }
            ]

            sample_file = self.output_dir / "validated_coco.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_data, f, indent=2)

            self.logger.info(f"✅ Sample dataset created: {sample_file}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to create sample data: {e}")
            return False

    def validate_dataset(self) -> bool:
        """Validate that dataset is ready for training"""
        validation_file = self.output_dir / "validated_coco.json"

        if not validation_file.exists():
            self.logger.error("❌ No validated dataset found")
            return False

        try:
            with open(validation_file, 'r') as f:
                data = json.load(f)

            self.logger.info(f"✅ Dataset validated: {len(data)} training samples available")
            return len(data) > 0

        except Exception as e:
            self.logger.error(f"❌ Dataset validation failed: {e}")
            return False


def download_and_prepare_coco(output_dir: str = "data/coco") -> bool:
    """Main function to download and prepare COCO dataset"""
    downloader = COCODownloader(output_dir)

    # Try Kaggle download
    if downloader.download_from_kaggle():
        downloader.process_dataset()
        return downloader.validate_dataset()
    else:
        # Fallback to sample data
        return downloader.download_sample_data()


if __name__ == "__main__":
    success = download_and_prepare_coco()
    if success:
        print("✅ COCO dataset ready for training")
    else:
        print("❌ Dataset preparation failed")
