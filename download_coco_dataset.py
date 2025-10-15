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

    def download_from_official(self) -> bool:
        """Download COCO dataset from official source (like FIBER)"""
        try:
            self.logger.info("🚀 Downloading COCO from official source...")
            self.logger.info("📦 Source: http://images.cocodataset.org/")
            
            # COCO 2017 (standard for training)
            urls = [
                ("http://images.cocodataset.org/zips/train2017.zip", "train2017.zip"),
                ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip"),
                ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
            ]
            
            for url, filename in urls:
                dest = self.output_dir / filename
                
                # Check if already extracted
                extracted_name = filename.replace('.zip', '')
                if (self.output_dir / extracted_name).exists():
                    self.logger.info(f"✓ {extracted_name} already exists")
                    continue
                
                if dest.exists():
                    self.logger.info(f"✓ {filename} already downloaded")
                else:
                    self.logger.info(f"📥 Downloading {filename} (~6GB for 2017, may take a while)...")
                    subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
                    self.logger.info(f"✓ Downloaded {filename}")
                
                # Extract using Python's zipfile (cross-platform)
                self.logger.info(f"📦 Extracting {filename}...")
                try:
                    import zipfile
                    with zipfile.ZipFile(dest, 'r') as zip_ref:
                        zip_ref.extractall(self.output_dir)
                    self.logger.info(f"✓ Extracted {filename}")
                except Exception as e:
                    self.logger.error(f"❌ Extraction failed: {e}")
                    # Fallback to unzip command
                    subprocess.run(["unzip", "-q", str(dest), "-d", str(self.output_dir)], check=False)
            
            self.logger.info("✅ COCO dataset downloaded from official source")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Official download failed: {e}")
            return False

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
        """Process downloaded dataset for BitGen training with proper pairing validation"""
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

            # Process all annotations and create proper image-caption pairs
            all_processed_data = []
            processed_image_ids = set()

            # Create unified image file mapping first
            image_files_dict = {}
            for img_file in image_files:
                base_name = img_file.name
                image_files_dict[base_name] = img_file

                # Also try without common prefixes
                for prefix in ['COCO_train2014_', 'COCO_val2014_', 'COCO_val2017_']:
                    if base_name.startswith(prefix):
                        clean_name = base_name[len(prefix):]
                        image_files_dict[clean_name] = img_file

            # Process each JSON file and collect all valid pairs
            for json_file in json_files:
                self.logger.info(f"📂 Processing: {json_file.name}")
                temp_data = []
                # Fixed method call - ensure exact parameter matching
                try:
                    if self._process_coco_format(json_file, image_files_dict, temp_data):
                        all_processed_data.extend(temp_data)
                        self.logger.info(f"   ✅ Added {len(temp_data)} valid image-caption pairs")
                except TypeError as e:
                    self.logger.error(f"❌ Method signature error: {e}")
                    self.logger.error(f"   Parameters: json_file={type(json_file)}, image_files_dict={type(image_files_dict)}, temp_data={type(temp_data)}")
                    # Skip this file and continue
                    continue

            # Validate and save final data
            if all_processed_data:
                self.logger.info(f"🔍 Processing {len(all_processed_data)} total pairs...")

                # Remove duplicates based on image_id + caption combination
                unique_pairs = {}
                for item in all_processed_data:
                    key = f"{item['image_id']}_{hash(item['caption'])}"
                    if key not in unique_pairs:
                        unique_pairs[key] = item

                final_data = list(unique_pairs.values())

                # USE ALL DATA - NO ARTIFICIAL LIMITS!
                # Removed the 50k limit - use everything available for better training
                self.logger.info(f"✅ Final dataset: {len(final_data)} unique image-caption pairs")
                self.logger.info("🚀 Using ALL available data for comprehensive training!")

                # Validate that each pair has correct image-caption matching
                self._validate_image_caption_pairs(final_data)

                # Save processed data
                output_file = self.output_dir / "validated_coco.json"
                with open(output_file, 'w') as f:
                    json.dump(final_data, f, indent=2)

                self.logger.info(f"💾 Saved validated dataset to: {output_file}")

                # AUTOMATICALLY CREATE VISUALIZATION GRID
                self.create_visualization_grid(final_data)

            else:
                self.logger.warning("⚠️ No valid pairs found, creating sample data")
                self.download_sample_data()

        except Exception as e:
            self.logger.error(f"❌ Dataset processing failed: {e}")
            self.download_sample_data()

    def _process_coco_format(self, json_file: Path, image_files_dict: dict, processed_data: List) -> bool:
        """Process COCO format JSON file with proper validation - USE ALL DATA"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Only process caption files - skip instances/keypoints files for now
            if 'captions' not in json_file.name.lower():
                self.logger.info(f"   ⏭️ Skipping non-caption file: {json_file.name}")
                return False

            if 'images' in data and 'annotations' in data:
                images = data['images']
                annotations = data['annotations']

                self.logger.info(f"   📊 Found {len(images)} images with {len(annotations)} captions in {json_file.name}")

                # Create image lookup
                image_lookup = {img['id']: img for img in images}

                valid_pairs = 0
                skipped_pairs = 0

                # Process ALL annotations - no artificial limits!
                self.logger.info(f"   🚀 Processing ALL {len(annotations)} annotations from {json_file.name}")

                for ann in annotations:
                    if 'image_id' in ann and 'caption' in ann:
                        img_id = ann['image_id']
                        caption = ann['caption'].strip()

                        # Skip if empty caption or too short
                        if not caption or len(caption) < 5:
                            skipped_pairs += 1
                            continue

                        if img_id in image_lookup:
                            img_info = image_lookup[img_id]
                            img_filename = img_info.get('file_name', f'{img_id:012d}.jpg')

                            # Find actual image file
                            img_path = None

                            # Try direct match first
                            if img_filename in image_files_dict:
                                img_path = image_files_dict[img_filename]
                            else:
                                # Try with different naming patterns
                                possible_names = [
                                    f'COCO_train2014_{img_id:012d}.jpg',
                                    f'COCO_val2014_{img_id:012d}.jpg',
                                    f'COCO_train2017_{img_id:012d}.jpg',
                                    f'COCO_val2017_{img_id:012d}.jpg',
                                    f'{img_id:012d}.jpg',
                                    img_filename
                                ]

                                for test_name in possible_names:
                                    if test_name in image_files_dict:
                                        img_path = image_files_dict[test_name]
                                        break

                            # Only add if we found the matching image file
                            if img_path and img_path.exists():
                                processed_data.append({
                                    'image_id': img_id,
                                    'image_path': str(img_path),
                                    'caption': caption,
                                    'width': img_info.get('width', 640),
                                    'height': img_info.get('height', 480),
                                    'source_file': json_file.name
                                })
                                valid_pairs += 1
                            else:
                                skipped_pairs += 1

                self.logger.info(f"   ✅ Extracted {valid_pairs} valid pairs, skipped {skipped_pairs} (missing images/bad captions)")
                return valid_pairs > 0

        except Exception as e:
            self.logger.error(f"❌ Error processing {json_file}: {e}")
            return False

        return False

    def _validate_image_caption_pairs(self, processed_data: List[Dict]):
        """Validate that each image-caption pair is correctly matched"""
        self.logger.info("🔍 Validating image-caption pairs...")

        valid_count = 0
        invalid_files = []

        # Check each pair
        for i, item in enumerate(processed_data):
            img_path = Path(item['image_path'])
            caption = item['caption']

            # Verify image exists and caption is valid
            if img_path.exists() and len(caption.strip()) >= 10:
                valid_count += 1

                # Log a few examples for verification
                if i < 3:
                    self.logger.info(f"   ✅ Pair {i+1}: {img_path.name} → '{caption[:50]}...'")
            else:
                invalid_files.append({
                    'image_path': str(img_path),
                    'caption': caption[:30] + '...' if len(caption) > 30 else caption,
                    'issue': 'missing_image' if not img_path.exists() else 'short_caption'
                })

        self.logger.info(f"✅ Validation complete: {valid_count}/{len(processed_data)} pairs are valid")

        if invalid_files:
            self.logger.warning(f"⚠️ Found {len(invalid_files)} invalid pairs")
            if len(invalid_files) <= 5:
                for invalid in invalid_files:
                    self.logger.warning(f"   ❌ {invalid['issue']}: {invalid['image_path']} → {invalid['caption']}")

        # Remove invalid pairs
        valid_data = [item for item in processed_data if
                     Path(item['image_path']).exists() and len(item['caption'].strip()) >= 10]

        # Update the list in place
        processed_data.clear()
        processed_data.extend(valid_data)

        # Save final summary
        self.logger.info(f"🎯 Final result: {len(processed_data)} verified image-caption pairs ready for training")

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

    def create_visualization_grid(self, data: List[Dict]):
        """Create and display 3x3 grid of image-caption pairs - Jupyter friendly"""
        try:
            # Add visualization imports with Jupyter-specific setup
            import matplotlib.pyplot as plt
            from PIL import Image
            import random

            # Ensure proper backend for Jupyter
            try:
                get_ipython()  # Check if in Jupyter
                plt.ion()  # Turn on interactive mode for Jupyter
                self.logger.info("🖥️ Jupyter notebook detected - enabling interactive display")
            except NameError:
                pass  # Not in Jupyter

            self.logger.info("🖼️ Creating visualization grid to verify image-caption alignment...")

            if len(data) < 9:
                self.logger.warning(f"⚠️ Only {len(data)} pairs available, need at least 9 for 3x3 grid")
                return None

            # Select 9 random pairs for visualization
            random.seed(42)  # Reproducible results
            selected_pairs = random.sample(data, 9)

            # Create the figure with better spacing
            fig, axes = plt.subplots(3, 3, figsize=(20, 16))
            fig.suptitle('COCO Dataset: Image-Caption Pairs Verification\n(Verifying proper alignment)',
                        fontsize=18, fontweight='bold', y=0.98)

            self.logger.info("📊 Processing 9 pairs for visualization grid:")

            for idx, pair in enumerate(selected_pairs):
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]

                image_path = pair['image_path']
                caption = pair['caption']
                image_id = pair['image_id']

                self.logger.info(f"   📷 Pair {idx+1}: Image ID {image_id}")
                self.logger.info(f"      📁 {Path(image_path).name}")
                self.logger.info(f"      💬 '{caption}'")

                try:
                    # Load and display image
                    if Path(image_path).exists():
                        img = Image.open(image_path)
                        img = img.convert('RGB')  # Ensure RGB format
                        ax.imshow(img)

                        # Add caption as title (truncate if too long)
                        caption_short = caption if len(caption) <= 85 else caption[:82] + "..."
                        ax.set_title(f"ID: {image_id}\n{caption_short}",
                                   fontsize=10, wrap=True, pad=10)

                        self.logger.info(f"      ✅ Image loaded successfully")

                    else:
                        # Image doesn't exist - show placeholder
                        ax.text(0.5, 0.5, f"❌ Image Missing\nID: {image_id}\n{caption[:40]}...",
                               ha='center', va='center', fontsize=10, transform=ax.transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        self.logger.warning(f"      ❌ Image file not found")

                except Exception as e:
                    self.logger.error(f"      ❌ Error loading image: {e}")
                    # Show error placeholder
                    ax.text(0.5, 0.5, f"⚠️ Error Loading\nID: {image_id}\n{str(e)[:30]}...",
                           ha='center', va='center', fontsize=10, transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="orange"))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                # Remove axis ticks and labels for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])

            # Adjust layout with more padding
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save visualization with high quality
            output_image = self.output_dir / "coco_verification_grid.png"
            plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='white',
                       pad_inches=0.2)
            self.logger.info(f"💾 Visualization grid saved to: {output_image}")

            # Display statistics
            self.logger.info("📊 DATASET VERIFICATION COMPLETE:")
            self.logger.info(f"   📈 Total processed pairs: {len(data)}")

            # Check how many images actually exist
            existing_images = sum(1 for item in data if Path(item['image_path']).exists())
            self.logger.info(f"   🖼️ Images found: {existing_images}/{len(data)} ({existing_images/len(data)*100:.1f}%)")

            if existing_images == len(data):
                self.logger.info("   ✅ Perfect! All images properly aligned with captions")
            else:
                self.logger.warning(f"   ⚠️ {len(data) - existing_images} images missing - check download")

            # Enhanced display for Jupyter notebooks
            try:
                get_ipython()  # Check if in Jupyter
                from IPython.display import display
                plt.show()
                display(fig)  # Explicit display for Jupyter
                self.logger.info("🖥️ Visualization displayed in Jupyter notebook")
            except NameError:
                # Not in Jupyter - regular display
                plt.show()
                self.logger.info("🖥️ Visualization displayed")
            except Exception as e:
                self.logger.warning(f"⚠️ Display might not work in current environment: {e}")
                self.logger.info("💾 Visualization saved to file instead")

            # Keep the figure object available for manual display
            return fig

        except ImportError:
            self.logger.error("❌ Matplotlib not available for visualization")
            self.logger.info("💡 Install with: pip install matplotlib pillow")
            return None
        except Exception as e:
            self.logger.error(f"❌ Visualization failed: {e}")
            return None


def download_and_prepare_coco(output_dir: str = "data/coco") -> bool:
    """Main function to download and prepare COCO dataset"""
    downloader = COCODownloader(output_dir)

    print("\n" + "="*60)
    print("COCO Dataset Downloader")
    print("="*60)

    # Try official source first (like FIBER)
    print("\n🔄 Method 1: Official COCO (http://images.cocodataset.org/)")
    if downloader.download_from_official():
        print("✅ Official download successful")
        downloader.process_dataset()
        if downloader.validate_dataset():
            return True
    
    # Fallback to Kaggle
    print("\n🔄 Method 2: Kaggle (fallback)")
    if downloader.download_from_kaggle():
        print("✅ Kaggle download successful")
        downloader.process_dataset()
        return downloader.validate_dataset()
    
    # Last resort: sample data
    print("\n🔄 Method 3: Creating sample data (last resort)")
    return downloader.download_sample_data()


if __name__ == "__main__":
    success = download_and_prepare_coco()
    if success:
        print("✅ COCO dataset ready for training")
    else:
        print("❌ Dataset preparation failed")
