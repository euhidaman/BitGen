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

            self.logger.info("Downloading COCO Image Caption dataset from Kaggle...")

            # Download the specific dataset you mentioned
            kaggle.api.dataset_download_files(
                'nikhil7280/coco-image-caption',
                path=str(self.output_dir),
                unzip=True
            )

            self.logger.info("✅ COCO Image Caption dataset downloaded successfully")
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
            self.logger.info("Processing COCO Image Caption dataset for BitGen...")

            # Look for the downloaded files - this dataset might have different structure
            all_files = list(self.output_dir.rglob("*"))

            # Find JSON files (captions) and image files
            json_files = [f for f in all_files if f.suffix.lower() == '.json']
            image_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]

            self.logger.info(f"Found {len(json_files)} JSON files and {len(image_files)} image files")

            if not json_files and not image_files:
                self.logger.warning("No COCO files found, creating sample data...")
                self.download_sample_data()
                return

            # Try to process different possible structures
            processed_data = []

            # Method 1: Look for standard COCO format JSON
            for json_file in json_files:
                if self._process_coco_format(json_file, image_files, processed_data):
                    break

            # Method 2: Try to process CSV format if available
            csv_files = [f for f in all_files if f.suffix.lower() == '.csv']
            if not processed_data and csv_files:
                for csv_file in csv_files:
                    if self._process_csv_format(csv_file, image_files, processed_data):
                        break

            # Method 3: Try to process directory structure with images and captions
            if not processed_data:
                self._process_directory_structure(image_files, processed_data)

            if processed_data:
                # Save processed data
                output_file = self.output_dir / "validated_coco.json"
                with open(output_file, 'w') as f:
                    json.dump(processed_data, f, indent=2)

                self.logger.info(f"✅ Processed {len(processed_data)} image-caption pairs")

                # Validate images exist
                self._validate_images(processed_data)
            else:
                self.logger.warning("Could not process dataset, creating sample data...")
                self.download_sample_data()

        except Exception as e:
            self.logger.error(f"Dataset processing failed: {e}")
            # Fallback to sample data
            self.download_sample_data()

    def _process_coco_format(self, json_file: Path, image_files: List[Path], processed_data: List) -> bool:
        """Process standard COCO format JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if it's COCO format
            if 'images' in data and 'annotations' in data:
                images = data['images']
                annotations = data['annotations']

                self.logger.info(f"Processing COCO format from {json_file.name}")
                self.logger.info(f"Found {len(images)} images and {len(annotations)} annotations")

                # Use pandas for efficient merging (like in the notebook)
                try:
                    import pandas as pd

                    # Create DataFrames (following the notebook approach)
                    images_df = pd.DataFrame(images)
                    annotations_df = pd.DataFrame(annotations)

                    # Merge on image_id (like in the notebook)
                    merged_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id', suffixes=('_annotation', '_image'))

                    # Select relevant columns
                    coco_df = merged_df[['image_id', 'file_name', 'caption', 'width', 'height']].copy()

                    self.logger.info(f"Merged data: {len(coco_df)} image-caption pairs")

                    # Create image filename mapping for actual files
                    image_files_dict = {}
                    for img_file in image_files:
                        # Try multiple naming patterns
                        image_files_dict[img_file.name] = img_file
                        # Also map without path prefixes
                        base_name = img_file.name
                        if base_name.startswith('COCO_'):
                            clean_name = base_name
                        else:
                            # Try to match COCO naming patterns
                            clean_name = base_name
                        image_files_dict[clean_name] = img_file

                    # Process the merged data (limit for performance)
                    processed_count = 0
                    for _, row in coco_df.head(1000).iterrows():  # Process up to 1000 samples
                        img_filename = row['file_name']
                        caption = str(row['caption']).st