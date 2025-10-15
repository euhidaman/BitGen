"""
Multi-Dataset Downloader for BitGen Stage 1 Training
Simple cross-platform downloader with visualization (like download_coco_dataset.py)

Downloads and visualizes:
- Visual Genome (regions + captions)  
- RefCOCO/+/g (phrase grounding)
- Flickr30k
- SBU Captions (helper scripts)
- Conceptual Captions 3M (helper scripts)

Usage:
    python download_fiber_datasets.py
"""

import os
import json
import subprocess
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time
import random


class FIBERDatasetDownloader:
    """Simple multi-dataset downloader with visualization"""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_visual_genome(self) -> bool:
        """Download Visual Genome dataset"""
        try:
            self.logger.info("🚀 Downloading Visual Genome dataset...")
            
            vg_dir = self.output_dir / "visual_genome"
            vg_dir.mkdir(parents=True, exist_ok=True)

            urls = [
                ("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "VG_100K.zip"),
                ("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "VG_100K_2.zip"),
            ]

            for url, filename in urls:
                dest = vg_dir / filename
                if dest.exists():
                    self.logger.info(f"✓ {filename} already exists")
                    continue

                self.logger.info(f"📥 Downloading {filename}...")
                subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
                self.logger.info(f"✓ Downloaded {filename}")

                # Extract
                if dest.suffix == '.zip':
                    self.logger.info(f"📦 Extracting {filename}...")
                    subprocess.run(["unzip", "-q", str(dest), "-d", str(vg_dir)], check=False)
                    self.logger.info(f"✓ Extracted {filename}")

            # Download annotations
            ann_urls = [
                ("https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip", "region_descriptions.json.zip"),
                ("https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip", "objects.json.zip"),
            ]

            for url, filename in ann_urls:
                dest = vg_dir / filename
                if dest.exists():
                    self.logger.info(f"✓ {filename} already exists")
                    continue

                self.logger.info(f"📥 Downloading {filename}...")
                subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
                self.logger.info(f"✓ Downloaded {filename}")

                # Extract JSON
                subprocess.run(["unzip", "-q", str(dest), "-d", str(vg_dir)], check=False)

            self.logger.info("✅ Visual Genome downloaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Visual Genome download failed: {e}")
            return False

    def download_coco_official(self) -> bool:
        """Download COCO 2014 and 2017 from official source (like FIBER)"""
        try:
            self.logger.info("🚀 Downloading COCO from official source...")
            
            coco_dir = self.output_dir / "coco"
            coco_dir.mkdir(parents=True, exist_ok=True)
            
            # COCO 2014 (needed for RefCOCO)
            coco_2014_urls = [
                ("http://images.cocodataset.org/zips/train2014.zip", "train2014.zip"),
                ("http://images.cocodataset.org/zips/val2014.zip", "val2014.zip"),
            ]
            
            # COCO 2017 (standard for training)
            coco_2017_urls = [
                ("http://images.cocodataset.org/zips/train2017.zip", "train2017.zip"),
                ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip"),
            ]
            
            # Annotations
            ann_urls = [
                ("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", "annotations_trainval2014.zip"),
                ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip"),
            ]
            
            all_urls = coco_2014_urls + coco_2017_urls + ann_urls
            
            for url, filename in all_urls:
                dest = coco_dir / filename
                
                # Check if already extracted
                extracted_name = filename.replace('.zip', '')
                if (coco_dir / extracted_name).exists():
                    self.logger.info(f"✓ {extracted_name} already exists")
                    continue
                
                if dest.exists():
                    self.logger.info(f"✓ {filename} already downloaded")
                else:
                    self.logger.info(f"📥 Downloading {filename} (~13GB total, may take a while)...")
                    subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
                    self.logger.info(f"✓ Downloaded {filename}")
                
                # Extract
                self.logger.info(f"📦 Extracting {filename}...")
                subprocess.run(["unzip", "-q", str(dest), "-d", str(coco_dir)], check=False)
                self.logger.info(f"✓ Extracted {filename}")
            
            self.logger.info("✅ COCO dataset downloaded from official source")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ COCO official download failed: {e}")
            return False

    def download_refcoco(self) -> bool:
        """Download RefCOCO/+/g annotations (uses COCO 2014 images)"""
        try:
            self.logger.info("🚀 Downloading RefCOCO annotations...")
            
            mdetr_dir = self.output_dir / "mdetr_annotations"
            mdetr_dir.mkdir(parents=True, exist_ok=True)

            url = "https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz"
            dest = mdetr_dir / "mdetr_annotations.tar.gz"

            if dest.exists():
                self.logger.info("✓ MDETR annotations already exist")
                return True

            self.logger.info("📥 Downloading MDETR annotations (RefCOCO/+/g)...")
            subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
            self.logger.info("✓ Downloaded MDETR annotations")

            # Extract
            self.logger.info("📦 Extracting annotations...")
            subprocess.run(["tar", "-xzf", str(dest), "-C", str(mdetr_dir)], check=False)
            self.logger.info("✓ Extracted annotations")

            self.logger.info("✅ RefCOCO annotations downloaded")
            self.logger.info("⚠️  Note: RefCOCO uses COCO 2014 images (downloaded above)")
            return True

        except Exception as e:
            self.logger.error(f"❌ RefCOCO download failed: {e}")
            return False

    def create_helper_scripts(self):
        """Create helper scripts for SBU and CC3M (require manual download)"""
        self.logger.info("📝 Creating helper scripts for SBU and CC3M...")

        # SBU helper
        sbu_dir = self.output_dir / "sbu"
        sbu_dir.mkdir(parents=True, exist_ok=True)

        sbu_script = sbu_dir / "README.txt"
        with open(sbu_script, 'w') as f:
            f.write("""SBU Captions Dataset (~1M image-text pairs)

Manual Download Required:
1. Visit: http://www.cs.virginia.edu/~vicente/sbucaptions/
2. Download:
   - SBU_captioned_photo_dataset_urls.txt
   - SBU_captioned_photo_dataset_captions.txt
3. Place files in this directory (data/sbu/)
4. Images will be downloaded automatically during training

Note: ~1M images, ~400GB total size
BitGen will handle image downloads on-the-fly during training.
""")

        # CC3M helper
        cc3m_dir = self.output_dir / "conceptual_captions"
        cc3m_dir.mkdir(parents=True, exist_ok=True)

        cc3m_script = cc3m_dir / "README.txt"
        with open(cc3m_script, 'w') as f:
            f.write("""Conceptual Captions 3M Dataset

Manual Download Required:
1. Visit: https://ai.google.com/research/ConceptualCaptions/download
2. Download:
   - Train_GCC-training.tsv
   - Validation_GCC-1.1.0-Validation.tsv
3. Place TSV files in this directory (data/conceptual_captions/)
4. Images will be downloaded automatically during training

Note: ~3M images
BitGen will handle image downloads on-the-fly during training.
""")

        self.logger.info(f"✓ Created {sbu_dir}/README.txt")
        self.logger.info(f"✓ Created {cc3m_dir}/README.txt")

    def visualize_datasets(self):
        """Visualize samples from each downloaded dataset (3x3 grids)"""
        try:
            import matplotlib.pyplot as plt
            from PIL import Image

            self.logger.info("🎨 Creating visualization grids...")

            # Visual Genome visualization
            vg_dir = self.output_dir / "visual_genome"
            vg_images = []
            
            for img_dir in [vg_dir / "VG_100K", vg_dir / "VG_100K_2"]:
                if img_dir.exists():
                    vg_images.extend(list(img_dir.glob("*.jpg")))
            
            if vg_images:
                self.logger.info(f"📊 Visualizing Visual Genome (found {len(vg_images)} images)")
                self._show_grid(random.sample(vg_images, min(9, len(vg_images))), 
                               "Visual Genome Dataset Samples", 
                               "visual_genome_grid.png")

            # COCO visualization (if exists)
            coco_dir = self.output_dir / "coco"
            for split_dir in ["train2017", "val2017", "train2014"]:
                coco_split = coco_dir / split_dir
                if coco_split.exists():
                    coco_images = list(coco_split.glob("*.jpg"))
                    if coco_images:
                        self.logger.info(f"📊 Visualizing COCO {split_dir} (found {len(coco_images)} images)")
                        self._show_grid(random.sample(coco_images, min(9, len(coco_images))),
                                       f"COCO {split_dir} Samples",
                                       f"coco_{split_dir}_grid.png")
                        break  # Only visualize one COCO split

            self.logger.info("✅ Visualization complete")

        except ImportError:
            self.logger.error("❌ Matplotlib/PIL not available for visualization")
            self.logger.info("💡 Install with: pip install matplotlib pillow")
        except Exception as e:
            self.logger.error(f"❌ Visualization failed: {e}")

    def _show_grid(self, image_paths: List[Path], title: str, save_name: str):
        """Create and display 3x3 grid of images"""
        import matplotlib.pyplot as plt
        from PIL import Image

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(title, fontsize=16)

        for idx, (ax, img_path) in enumerate(zip(axes.flat, image_paths)):
            try:
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(img_path.name[:20], fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, 'Error loading', ha='center', va='center')
                ax.axis('off')

        # Hide unused subplots
        for idx in range(len(image_paths), 9):
            axes.flat[idx].axis('off')

        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"💾 Saved: {save_path}")

        # Show
        try:
            plt.show()
        except:
            pass

    def create_dataset_summary(self):
        """Create dataset_info.json with what's available"""
        info = {
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": {}
        }

        # Check what's available
        vg_dir = self.output_dir / "visual_genome"
        if (vg_dir / "VG_100K").exists():
            vg_count = len(list((vg_dir / "VG_100K").glob("*.jpg")))
            vg_count += len(list((vg_dir / "VG_100K_2").glob("*.jpg"))) if (vg_dir / "VG_100K_2").exists() else 0
            info["datasets"]["visual_genome"] = {
                "path": str(vg_dir),
                "images": vg_count,
                "type": "coarse-grained + fine-grained"
            }

        mdetr_dir = self.output_dir / "mdetr_annotations"
        if mdetr_dir.exists():
            refcoco_files = list(mdetr_dir.glob("final_refcoco*.json"))
            info["datasets"]["refcoco"] = {
                "path": str(mdetr_dir),
                "files": [f.name for f in refcoco_files],
                "type": "fine-grained (phrase grounding)"
            }

        coco_dir = self.output_dir / "coco"
        if coco_dir.exists():
            coco_splits = []
            for split in ["train2017", "val2017", "train2014"]:
                if (coco_dir / split).exists():
                    coco_splits.append(split)
            if coco_splits:
                info["datasets"]["coco"] = {
                    "path": str(coco_dir),
                    "splits": coco_splits,
                    "type": "coarse-grained + fine-grained"
                }

        # Save
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

        self.logger.info(f"✓ Created {info_file}")
        return info


def download_and_prepare_fiber_datasets(output_dir: str = "data") -> bool:
    """Main function to download FIBER datasets"""
    downloader = FIBERDatasetDownloader(output_dir)

    print("\n" + "="*60)
    print("BitGen Multi-Dataset Downloader")
    print("="*60)

    # Download COCO (official source - 2014 + 2017)
    print("\n1️⃣  COCO 2014 + 2017 (official source, ~13GB)")
    coco_success = downloader.download_coco_official()

    # Download Visual Genome
    print("\n2️⃣  Visual Genome (images + regions + captions)")
    vg_success = downloader.download_visual_genome()

    # Download RefCOCO annotations
    print("\n3️⃣  RefCOCO/+/g (phrase grounding annotations)")
    refcoco_success = downloader.download_refcoco()

    # Create helper scripts for SBU/CC3M
    print("\n4️⃣  SBU & CC3M (helper scripts)")
    downloader.create_helper_scripts()

    # Create summary
    print("\n📋 Creating dataset summary...")
    info = downloader.create_dataset_summary()

    # Visualize
    print("\n🎨 Creating visualization grids...")
    downloader.visualize_datasets()

    print("\n" + "="*60)
    print("Download Summary:")
    print("="*60)
    for dataset, details in info.get("datasets", {}).items():
        print(f"✅ {dataset}: {details.get('path', 'N/A')}")
    print("="*60)

    print("\n📝 Next Steps:")
    print("1. For SBU/CC3M: see README.txt files in data/sbu/ and data/conceptual_captions/")
    print("2. Alternative COCO: run `python download_coco_dataset.py` for Kaggle version (if official fails)")
    print("3. Start training: `python src/train_stage1_vision_language.py`")

    return coco_success or vg_success or refcoco_success


if __name__ == "__main__":
    success = download_and_prepare_fiber_datasets()
    if success:
        print("\n✅ Datasets ready for training")
    else:
        print("\n❌ Some downloads failed (check logs above)")
