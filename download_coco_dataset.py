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
                        caption = str(row['caption']).strip()

                        # Try to find the actual image file
                        img_path = None

                        # Method 1: Direct filename match
                        if img_filename in image_files_dict:
                            img_path = image_files_dict[img_filename]
                        else:
                            # Method 2: Try common COCO prefixes
                            for prefix in ['', 'COCO_train2014_', 'COCO_val2014_', 'COCO_val2017_', 'train2014_', 'val2014_', 'val2017_']:
                                test_name = f"{prefix}{img_filename}"
                                if test_name in image_files_dict:
                                    img_path = image_files_dict[test_name]
                                    break

                        if img_path and caption:
                            processed_data.append({
                                'image_id': int(row['image_id']),
                                'image_file': str(img_path.relative_to(self.output_dir)),
                                'image_path': str(img_path),
                                'caption': caption,
                                'width': int(row['width']) if pd.notna(row['width']) else 640,
                                'height': int(row['height']) if pd.notna(row['height']) else 480
                            })
                            processed_count += 1

                    self.logger.info(f"Successfully processed {processed_count} image-caption pairs from {json_file.name}")
                    return processed_count > 0

                except ImportError:
                    # Fallback without pandas
                    self.logger.warning("pandas not available, using slower processing")
                    return self._process_coco_format_without_pandas(data, image_files, processed_data)

            else:
                # Check for other possible formats
                if isinstance(data, list) and len(data) > 0 and 'caption' in data[0]:
                    # Direct list of caption objects
                    self.logger.info(f"Processing direct caption list format with {len(data)} entries")

                    image_files_dict = {img_file.name: img_file for img_file in image_files}

                    for item in data[:500]:  # Limit for performance
                        if 'image_id' in item and 'caption' in item:
                            # Try to find corresponding image
                            img_filename = item.get('file_name', f"image_{item['image_id']}.jpg")

                            if img_filename in image_files_dict:
                                img_path = image_files_dict[img_filename]
                                processed_data.append({
                                    'image_id': item['image_id'],
                                    'image_file': str(img_path.relative_to(self.output_dir)),
                                    'image_path': str(img_path),
                                    'caption': str(item['caption']).strip(),
                                    'width': item.get('width', 640),
                                    'height': item.get('height', 480)
                                })

                    return len(processed_data) > 0

                return False

        except Exception as e:
            self.logger.error(f"Error processing COCO format {json_file}: {e}")
            return False

    def _process_coco_format_without_pandas(self, data: Dict, image_files: List[Path], processed_data: List) -> bool:
        """Fallback method to process COCO format without pandas"""
        try:
            images = data['images']
            annotations = data['annotations']

            # Create lookup dictionaries
            image_lookup = {img['id']: img for img in images}
            image_files_dict = {img_file.name: img_file for img_file in image_files}

            # Process annotations
            for ann in annotations[:500]:  # Limit for performance
                if 'image_id' in ann and 'caption' in ann:
                    img_id = ann['image_id']
                    if img_id in image_lookup:
                        img_info = image_lookup[img_id]
                        img_filename = img_info.get('file_name', f"image_{img_id}.jpg")

                        # Try to find the image file
                        img_path = None
                        if img_filename in image_files_dict:
                            img_path = image_files_dict[img_filename]
                        else:
                            # Try with COCO prefixes
                            for prefix in ['COCO_train2014_', 'COCO_val2014_', 'COCO_val2017_']:
                                test_name = f"{prefix}{img_filename}"
                                if test_name in image_files_dict:
                                    img_path = image_files_dict[test_name]
                                    break

                        if img_path:
                            processed_data.append({
                                'image_id': img_id,
                                'image_file': str(img_path.relative_to(self.output_dir)),
                                'image_path': str(img_path),
                                'caption': str(ann['caption']).strip(),
                                'width': img_info.get('width', 640),
                                'height': img_info.get('height', 480)
                            })

            return len(processed_data) > 0

        except Exception as e:
            self.logger.error(f"Error in fallback COCO processing: {e}")
            return False

    def _process_csv_format(self, csv_file: Path, image_files: List[Path], processed_data: List) -> bool:
        """Process CSV format with image-caption pairs"""
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

            for idx, row in df.iterrows():
                if idx >= 1000:  # Limit for faster processing
                    break

                img_filename = str(row[image_col])
                caption = str(row[caption_col])

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

                if img_path and caption.strip():
                    processed_data.append({
                        'image_id': idx + 1,
                        'image_file': str(img_path.relative_to(self.output_dir)),
                        'image_path': str(img_path),
                        'caption': caption.strip(),
                        'width': 640,  # Default values
                        'height': 480
                    })

            return len(processed_data) > 0

        except ImportError:
            self.logger.warning("pandas not available for CSV processing")
            return False
        except Exception as e:
            self.logger.error(f"Error processing CSV format {csv_file}: {e}")
            return False

    def _process_directory_structure(self, image_files: List[Path], processed_data: List):
        """Process images directly from directory structure"""
        try:
            self.logger.info(f"Processing {len(image_files)} images from directory structure")

            for idx, img_path in enumerate(image_files[:500]):  # Limit for faster processing
                # Generate a simple caption based on filename or use default
                img_name = img_path.stem

                # Try to create a meaningful caption from filename
                caption = self._generate_caption_from_filename(img_name)

                processed_data.append({
                    'image_id': idx + 1,
                    'image_file': str(img_path.relative_to(self.output_dir)),
                    'image_path': str(img_path),
                    'caption': caption,
                    'width': 640,  # Default values - could be extracted from image
                    'height': 480
                })

            self.logger.info(f"Generated captions for {len(processed_data)} images")

        except Exception as e:
            self.logger.error(f"Error processing directory structure: {e}")

    def _generate_caption_from_filename(self, filename: str) -> str:
        """Generate a basic caption from image filename"""
        # Remove common prefixes/suffixes and underscores
        name = filename.lower()
        name = name.replace('_', ' ').replace('-', ' ')

        # Common COCO-style captions based on filename patterns
        if any(word in name for word in ['person', 'people', 'man', 'woman']):
            return f"A person in the image {filename}"
        elif any(word in name for word in ['car', 'vehicle', 'truck', 'bus']):
            return f"A vehicle shown in {filename}"
        elif any(word in name for word in ['cat', 'dog', 'animal', 'bird']):
            return f"An animal captured in {filename}"
        elif any(word in name for word in ['food', 'pizza', 'cake']):
            return f"Food item displayed in {filename}"
        else:
            return f"An image showing the contents of {filename}"

    def _validate_images(self, processed_data: List[Dict]):
        """Validate that image files actually exist"""
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
        if invalid_files:
            self.logger.warning(f"❌ {len(invalid_files)} image files not found")
            # Remove invalid entries
            valid_data = [item for item in processed_data if Path(item['image_path']).exists()]
            processed_data.clear()
            processed_data.extend(valid_data)

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

def download_and_prepare_coco(output_dir: str = "data/coco"):
    """Main function to download and prepare COCO dataset"""

    downloader = COCODownloader(output_dir)

    print("📥 BitGen COCO Image Caption Dataset Setup")
    print("=" * 50)
    print(f"🎯 Target dataset: nikhil7280/coco-image-caption")
    print(f"📁 Output directory: {output_dir}")

    # Check if Kaggle credentials are set up
    try:
        import kaggle
        print("✅ Kaggle API credentials found")
    except ImportError:
        print("❌ Kaggle API not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Kaggle credentials issue: {e}")
        print("💡 Make sure ~/.kaggle/kaggle.json exists with your API token")
        return False

    # Try to download from Kaggle
    print("\n🚀 Starting download from Kaggle...")
    if downloader.download_from_kaggle():
        print("✅ COCO Image Caption dataset downloaded from Kaggle")

        # Get initial info about downloaded files
        info = downloader.get_dataset_info()
        print(f"\n📊 Downloaded files:")
        print(f"   📄 JSON files: {info.get('json_files', 0)}")
        print(f"   🖼️  Image files: {info.get('image_files', 0)}")
        print(f"   📋 CSV files: {info.get('csv_files', 0)}")
        print(f"   💾 Total size: {info.get('total_size_mb', 0):.1f} MB")

    else:
        print("❌ Kaggle download failed, creating sample dataset...")
        if downloader.download_sample_data():
            print("✅ Created sample dataset for testing")
        else:
            print("❌ Failed to create sample dataset")
            return False

    # Process the dataset
    print("\n🔄 Processing dataset for BitGen training...")
    downloader.process_dataset()

    # Get final info and validate
    final_info = downloader.get_dataset_info()

    if downloader.validate_dataset():
        print("\n🎉 Dataset is ready for BitGen training!")
        print(f"   ✅ Processed samples: {final_info.get('processed_samples', 0)}")
        print(f"   🖼️  Image files available: {final_info.get('image_files', 0)}")
        print(f"   📁 Dataset location: {output_dir}")

        # Show sample of what was processed
        validation_file = Path(output_dir) / "validated_coco.json"
        if validation_file.exists():
            try:
                with open(validation_file, 'r') as f:
                    sample_data = json.load(f)[:3]  # Show first 3 samples

                print(f"\n📋 Sample processed data:")
                for i, sample in enumerate(sample_data, 1):
                    print(f"   {i}. Image: {sample.get('image_file', 'N/A')}")
                    print(f"      Caption: {sample.get('caption', 'N/A')[:80]}...")
                    print(f"      Size: {sample.get('width', 0)}x{sample.get('height', 0)}")
            except:
                pass

        return True
    else:
        print("❌ Dataset validation failed")
        print("💡 Check if images were properly downloaded and can be accessed")
        return False
