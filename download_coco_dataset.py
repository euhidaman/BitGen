#!/usr/bin/env python3
"""
COCO Dataset Download Script for BitGen
Downloads COCO images and captions from Kaggle for cross-modal training.
"""

import os
import sys
import zipfile
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import kaggle
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class COCODownloader:
    def __init__(self, data_dir: str = "data/coco"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_from_kaggle(self):
        """Download COCO dataset from Kaggle"""
        print("Downloading COCO dataset from Kaggle...")

        # Ensure Kaggle API credentials are set
        if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            print("Please set up Kaggle API credentials first:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Create new API token")
            print("3. Place kaggle.json in ~/.kaggle/")
            print("4. chmod 600 ~/.kaggle/kaggle.json")
            sys.exit(1)

        try:
            # Download the COCO image-caption dataset
            kaggle.api.dataset_download_files(
                'nikhil7280/coco-image-caption',
                path=str(self.data_dir),
                unzip=True
            )
            print(f"Dataset downloaded to {self.data_dir}")

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Fallback: Please manually download from https://www.kaggle.com/datasets/nikhil7280/coco-image-caption")
            return False

        return True

    def process_dataset(self):
        """Process downloaded COCO dataset for BitGen training"""
        print("Processing COCO dataset...")

        # Find the dataset files
        dataset_files = list(self.data_dir.rglob("*.json"))
        image_dirs = [d for d in self.data_dir.rglob("*") if d.is_dir() and any(d.glob("*.jpg"))]

        if not dataset_files or not image_dirs:
            print("Dataset files not found. Please check the download.")
            return False

        # Process annotations
        processed_data = []

        for json_file in dataset_files:
            print(f"Processing {json_file}")

            with open(json_file, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            if 'annotations' in data and 'images' in data:
                # COCO format
                image_info = {img['id']: img for img in data['images']}

                for ann in data['annotations']:
                    if ann['image_id'] in image_info:
                        img_info = image_info[ann['image_id']]
                        processed_data.append({
                            'image_id': ann['image_id'],
                            'image_file': img_info['file_name'],
                            'caption': ann['caption'],
                            'width': img_info.get('width', 0),
                            'height': img_info.get('height', 0)
                        })

            elif isinstance(data, list):
                # Simple list format
                for item in data:
                    if 'image' in item and 'caption' in item:
                        processed_data.append({
                            'image_id': item.get('id', len(processed_data)),
                            'image_file': item['image'],
                            'caption': item['caption'],
                            'width': 0,
                            'height': 0
                        })

        # Save processed data
        output_file = self.data_dir / "processed_coco.json"
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)

        print(f"Processed {len(processed_data)} image-caption pairs")
        print(f"Saved to {output_file}")

        # Validate images
        self.validate_images(processed_data)

        return True

    def validate_images(self, data: List[Dict]):
        """Validate that images exist and are readable"""
        print("Validating images...")

        valid_data = []
        image_dirs = [d for d in self.data_dir.rglob("*") if d.is_dir() and any(d.glob("*.jpg"))]

        for item in tqdm(data):
            image_found = False

            # Search for image in all subdirectories
            for img_dir in image_dirs:
                img_path = img_dir / item['image_file']
                if img_path.exists():
                    try:
                        # Try to open image
                        with Image.open(img_path) as img:
                            width, height = img.size
                            item['width'] = width
                            item['height'] = height
                            item['image_path'] = str(img_path)
                            valid_data.append(item)
                            image_found = True
                            break
                    except Exception as e:
                        print(f"Error opening {img_path}: {e}")

            if not image_found:
                print(f"Image not found: {item['image_file']}")

        print(f"Found {len(valid_data)} valid image-caption pairs")

        # Save validated data
        output_file = self.data_dir / "validated_coco.json"
        with open(output_file, 'w') as f:
            json.dump(valid_data, f, indent=2)

        return valid_data

class COCODataset(Dataset):
    """COCO Dataset for BitGen training"""

    def __init__(self, data_file: str, transform=None, max_caption_length: int = 77):
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.transform = transform
        self.max_caption_length = max_caption_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Process caption
        caption = item['caption']
        if len(caption) > self.max_caption_length:
            caption = caption[:self.max_caption_length]

        return {
            'image': image,
            'caption': caption,
            'image_id': item['image_id'],
            'width': item['width'],
            'height': item['height']
        }

def create_embedded_splits(data_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Create train/val/test splits optimized for embedded training"""
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Shuffle data
    import random
    random.shuffle(data)

    # Calculate split sizes
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # Create splits
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Save splits
    base_path = Path(data_file).parent

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, split_data in splits.items():
        output_file = base_path / f"coco_{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(split_data, f)
        print(f"Saved {len(split_data)} samples to {output_file}")

    return splits

def main():
    """Main download and processing function"""
    print("BitGen COCO Dataset Downloader")
    print("=" * 40)

    # Initialize downloader
    downloader = COCODownloader()

    # Download dataset
    if not downloader.download_from_kaggle():
        return

    # Process dataset
    if not downloader.process_dataset():
        return

    # Create splits for embedded training
    data_file = downloader.data_dir / "validated_coco.json"
    if data_file.exists():
        create_embedded_splits(str(data_file))
        print("Dataset preparation complete!")

        # Print statistics
        with open(data_file, 'r') as f:
            data = json.load(f)
        print(f"Total samples: {len(data)}")
        print(f"Dataset location: {downloader.data_dir}")
    else:
        print("Error: Validated dataset not found")

if __name__ == "__main__":
    main()
