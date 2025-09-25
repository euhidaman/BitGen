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

    def process_dataset(self):
        """Process downloaded dataset for BitGen training - compatible with existing data loader"""
        try:
            self.logger.info("Processing COCO Image Caption dataset for BitGen...")

            # Look for the downloaded files
            all_files = list(self.output_dir.rglob("*"))

            # Find JSON files (captions) and image files
            json_files = [f for f in all_files if f.suffix.lower() == '.json' and 'sample' not in f.name]
            image_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]

            self.logger.info(f"Found {len(json_files)} JSON files and {len(image_files)} image files")

            if not json_files and not image_files:
                self.logger.warning("No COCO files found, creating sample data...")
                self.download_sample_data()
                return

            # Process into BitGen-compatible format
            processed_data = []

            # Try different processing methods
            success = False

            # Method 1: Standard COCO format
            for json_file in json_files:
                if self._process_coco_compatible(json_file, image_files, processed_data):
                    success = True
                    break

            # Method 2: CSV format
            if not success:
                csv_files = [f for f in all_files if f.suffix.lower() == '.csv']
                for csv_file in csv_files:
                    if self._process_csv_format(csv_file, image_files, processed_data):
                        success = True
                        break

            # Method 3: Directory structure
            if not success:
                self._process_directory_structure(image_files, processed_data)
                success = len(processed_data) > 0

            if processed_data:
                # Save in BitGen-compatible format
                output_file = self.output_dir / "validated_coco.json"
                with open(output_file, 'w') as f:
                    json.dump(processed_data, f, indent=2)

                self.logger.info(f"✅ Processed {len(processed_data)} image-caption pairs for BitGen")

                # Validate images exist
                self._validate_images(processed_data)

                # Save final validated data
                self._save_validated_data(processed_data)
            else:
                self.logger.warning("Could not process dataset, creating sample data...")
                self.download_sample_data()

        except Exception as e:
            self.logger.error(f"Dataset processing failed: {e}")
            # Fallback to sample data
            self.download_sample_data()

    def _process_coco_compatible(self, json_file: Path, image_files: List[Path], processed_data: List) -> bool:
        """Process COCO format to be compatible with BitGen data loader"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' in data and 'annotations' in data:
                images = data['images']
                annotations = data['annotations']

                self.logger.info(f"Processing {len(images)} images with {len(annotations)} captions from {json_file.name}")

                # Create image lookup
                image_lookup = {img['id']: img for img in images}

                # Create image filename mapping
                image_files_dict = {}
                for img_file in image_files:
                    base_name = img_file.name
                    image_files_dict[base_name] = img_file

                    # Also try without common prefixes
                    for prefix in ['COCO_train2014_', 'COCO_val2014_', 'COCO_val2017_', 'train2014_', 'val2014_', 'val2017_']:
                        if base_name.startswith(prefix):
                            clean_name = base_name[len(prefix):]
                            image_files_dict[clean_name] = img_file
                            break

                # Process annotations and create BitGen-compatible entries
                processed_count = 0
                for ann in annotations[:1000]:  # Limit for performance
                    if 'image_id' in ann and 'caption' in ann:
                        img_id = ann['image_id']
                        caption = ann['caption'].strip()

                        if img_id in image_lookup and caption:
                            img_info = image_lookup[img_id]
                            img_filename = img_info.get('file_name', f'image_{img_id}.jpg')

                            # Try to find the actual image file
                            img_path = None

                            # Direct match
                            if img_filename in image_files_dict:
                                img_path = image_files_dict[img_filename]
                            else:
                                # Try with different prefixes
                                for prefix in ['', 'COCO_train2014_', 'COCO_val2014_', 'COCO_val2017_']:
                                    test_name = f"{prefix}{img_filename}"
                                    if test_name in image_files_dict:
                                        img_path = image_files_dict[test_name]
                                        break

                            if img_path:
                                # Create BitGen-compatible entry
                                processed_data.append({
                                    'image_id': img_id,
                                    'image_file': str(img_path.relative_to(self.output_dir)),
                                    'image_path': str(img_path),
                                    'caption': caption,
                                    'width': img_info.get('width', 640),
                                    'height': img_info.get('height', 480)
                                })
                                processed_count += 1

                self.logger.info(f"Successfully processed {processed_count} image-caption pairs")
                return processed_count > 0

            return False

        except Exception as e:
            self.logger.error(f"Error processing {json_file}: {e}")
            return False

    def _save_validated_data(self, processed_data: List[Dict]):
        """Save data in format compatible with BitGen COCODataset"""
        try:
            # Filter out invalid entries
            valid_data = []
            for item in processed_data:
                if Path(item['image_path']).exists() and item['caption'].strip():
                    valid_data.append(item)

            # Save in final format
            output_file = self.output_dir / "validated_coco.json"
            with open(output_file, 'w') as f:
                json.dump(valid_data, f, indent=2)

            # Also create a summary file for easy reference
            summary = {
                'dataset_info': {
                    'total_samples': len(valid_data),
                    'source': 'nikhil7280/coco-image-caption',
                    'processed_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'format': 'BitGen-compatible'
                },
                'sample_data': valid_data[:5]  # First 5 for reference
            }

            summary_file = self.output_dir / "dataset_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Saved {len(valid_data)} validated samples for BitGen training")

        except Exception as e:
            self.logger.error(f"Error saving validated data: {e}")

    def download_sample_data(self) -> bool:
        """Create sample data compatible with BitGen"""
        try:
            self.logger.info("Creating BitGen-compatible sample dataset...")

            # Create sample data in the format expected by BitGen
            sample_data = [
                {
                    'image_id': 1,
                    'image_file': 'sample_001.jpg',
                    'image_path': str(self.output_dir / 'sample_001.jpg'),
                    'caption': 'A robot arm picking up a red cube from a table',
                    'width': 640,
                    'height': 480
                },
                {
                    'image_id': 2,
                    'image_file': 'sample_002.jpg',
                    'image_path': str(self.output_dir / 'sample_002.jpg'),
                    'caption': 'Industrial robot performing pick and place operation',
                    'width': 640,
                    'height': 480
                },
                {
                    'image_id': 3,
                    'image_file': 'sample_003.jpg',
                    'image_path': str(self.output_dir / 'sample_003.jpg'),
                    'caption': 'Mobile robot navigating through a warehouse corridor',
                    'width': 640,
                    'height': 480
                }
            ]

            # Save in BitGen format
            sample_file = self.output_dir / "validated_coco.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_data, f, indent=2)

            self.logger.info(f"✅ BitGen-compatible sample data created: {sample_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create sample data: {e}")
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

            # Check if images actually exist
            valid_samples = 0
            for item in data:
                if 'image_path' in item and Path(item['image_path']).exists():
                    valid_samples += 1

            self.logger.info(f"✅ Dataset validated: {len(data)} training samples available")
            self.logger.info(f"✅ {valid_samples}/{len(data)} images verified to exist")

            if valid_samples < len(data) * 0.8:  # Less than 80% valid
                self.logger.warning("⚠️ Many image files are missing. Check dataset download.")

            return valid_samples > 5  # Need at least 5 valid samples

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False

    def get_dataset_info(self) -> Dict:
        """Get information about the downloaded dataset"""
        try:
            all_files = list(self.output_dir.rglob("*"))
            json_files = [f for f in all_files if f.suffix.lower() == '.json']
            image_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
            csv_files = [f for f in all_files if f.suffix.lower() == '.csv']

            # Calculate total size
            total_size_mb = sum(f.stat().st_size for f in all_files if f.is_file()) / (1024 * 1024)

            # Check for validated dataset
            validation_file = self.output_dir / "validated_coco.json"
            processed_samples = 0
            if validation_file.exists():
                with open(validation_file, 'r') as f:
                    processed_samples = len(json.load(f))

            return {
                'total_files': len(all_files),
                'json_files': len(json_files),
                'image_files': len(image_files),
                'csv_files': len(csv_files),
                'total_size_mb': total_size_mb,
                'processed_samples': processed_samples,
                'ready_for_training': validation_file.exists() and processed_samples > 5
            }

        except Exception as e:
            self.logger.error(f"Error getting dataset info: {e}")
            return {'error': str(e)}

    def _process_coco_file(self, coco_file: Path) -> bool:
        """Legacy method - kept for compatibility"""
        return self._process_coco_format(coco_file, [], [])

    def _process_csv_format(self, csv_file: Path, image_files: List[Path], processed_data: List) -> bool:
        """Process CSV format with image-caption pairs - BitGen compatible"""
        try:
            import pandas as pd

            df = pd.read_csv(csv_file)
            self.logger.info(f"Processing CSV format with {len(df)} rows")

            # Look for common column names
            image_col = None
            caption_col = None

            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['image', 'img', 'file']):
                    image_col = col
                if any(keyword in col.lower() for keyword in ['caption', 'description', 'text']):
                    caption_col = col

            if not image_col or not caption_col:
                self.logger.warning(f"Could not identify image and caption columns in {csv_file}")
                return False

            self.logger.info(f"Using columns: {image_col} (images), {caption_col} (captions)")

            # Create image filename mapping
            image_files_dict = {img_file.name: img_file for img_file in image_files}

            processed_count = 0
            for idx, row in df.iterrows():
                if processed_count >= 1000:  # Limit for performance
                    break

                img_filename = str(row[image_col])
                caption = str(row[caption_col]).strip()

                # Try to find the image file
                img_path = None
                if img_filename in image_files_dict:
                    img_path = image_files_dict[img_filename]
                else:
                    # Try without extension and with common extensions
                    base_name = Path(img_filename).stem
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        test_name = f"{base_name}{ext}"
                        if test_name in image_files_dict:
                            img_path = image_files_dict[test_name]
                            break

                if img_path and caption:
                    # BitGen-compatible format
                    processed_data.append({
                        'image_id': processed_count + 1,
                        'image_file': str(img_path.relative_to(self.output_dir)),
                        'image_path': str(img_path),
                        'caption': caption,
                        'width': 640,  # Default values
                        'height': 480
                    })
                    processed_count += 1

            self.logger.info(f"Successfully processed {processed_count} samples from CSV")
            return processed_count > 0

        except ImportError:
            self.logger.warning("pandas not available for CSV processing")
            return False
        except Exception as e:
            self.logger.error(f"Error processing CSV format {csv_file}: {e}")
            return False

    def _process_directory_structure(self, image_files: List[Path], processed_data: List):
        """Process images directly from directory structure - BitGen compatible"""
        try:
            self.logger.info(f"Processing {len(image_files)} images from directory structure")

            processed_count = 0
            for img_path in image_files[:500]:  # Limit for performance
                # Generate a caption based on filename
                img_name = img_path.stem
                caption = self._generate_caption_from_filename(img_name)

                # BitGen-compatible format
                processed_data.append({
                    'image_id': processed_count + 1,
                    'image_file': str(img_path.relative_to(self.output_dir)),
                    'image_path': str(img_path),
                    'caption': caption,
                    'width': 640,  # Default values
                    'height': 480
                })
                processed_count += 1

            self.logger.info(f"Generated captions for {processed_count} images")

        except Exception as e:
            self.logger.error(f"Error processing directory structure: {e}")

    def _generate_caption_from_filename(self, filename: str) -> str:
        """Generate a basic caption from image filename"""
        # Remove common prefixes/suffixes and underscores
        name = filename.lower()
        name = name.replace('_', ' ').replace('-', ' ')

        # Generate robot/AI focused captions for BitGen
        if any(word in name for word in ['person', 'people', 'man', 'woman']):
            return f"A person interacting with robotic systems"
        elif any(word in name for word in ['car', 'vehicle', 'truck', 'bus']):
            return f"An autonomous vehicle in operation"
        elif any(word in name for word in ['cat', 'dog', 'animal', 'bird']):
            return f"An animal that robots must navigate around"
        elif any(word in name for word in ['food', 'kitchen']):
            return f"A kitchen environment for service robots"
        else:
            return f"A scene for robotic understanding and navigation"

    def _validate_images(self, processed_data: List[Dict]):
        """Validate that image files actually exist and get real dimensions"""
        valid_count = 0
        invalid_files = []

        for item in processed_data:
            img_path = Path(item['image_path'])
            if img_path.exists():
                valid_count += 1
                # Try to get actual image dimensions
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        item['width'] = img.width
                        item['height'] = img.height
                except:
                    pass  # Keep default dimensions if PIL not available
            else:
                invalid_files.append(str(img_path))

        self.logger.info(f"✅ Validated {valid_count}/{len(processed_data)} images exist")
        if invalid_files and len(invalid_files) < 10:
            self.logger.warning(f"❌ Missing files: {invalid_files}")
        elif invalid_files:
            self.logger.warning(f"❌ {len(invalid_files)} image files not found")

        # Remove invalid entries
        valid_data = [item for item in processed_data if Path(item['image_path']).exists()]
        processed_data.clear()
        processed_data.extend(valid_data)
```
