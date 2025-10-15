"""
Download FIBER-style datasets for BitGen Stage 1 training
Supports both coarse-grained (image-text pairs) and fine-grained (region annotations)

Datasets:
- Coarse-Grained: COCO, SBU, Visual Genome, Conceptual Captions 3M, Flickr30k
- Fine-Grained: RefCOCO, RefCOCO+, RefCOCOg, Visual Genome regions, Objects365

Usage:
    python download_fiber_datasets.py --datasets coco sbu vg cc3m --data_root ./data
    python download_fiber_datasets.py --all  # Download all datasets
"""

import os
import sys
import json
import argparse
import urllib.request
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import requests


class DatasetDownloader:
    """Download and prepare FIBER-style datasets"""
    
    def __init__(self, data_root="./data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url, dest_path, desc=None):
        """Download file with progress bar"""
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if dest_path.exists():
            print(f"✓ {dest_path.name} already exists, skipping")
            return
        
        print(f"Downloading {desc or dest_path.name}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=desc or dest_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print(f"✓ Downloaded {dest_path.name}")
        except Exception as e:
            print(f"✗ Failed to download {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            raise
    
    def extract_archive(self, archive_path, extract_dir):
        """Extract zip/tar/gz archive"""
        archive_path = Path(archive_path)
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        
        print(f"✓ Extracted to {extract_dir}")
    
    def download_coco(self):
        """Download COCO 2017 train/val for coarse-grained + COCO 2014 for fine-grained"""
        print("\n" + "="*60)
        print("Downloading COCO Dataset")
        print("="*60)
        
        coco_root = self.data_root / "coco"
        coco_root.mkdir(parents=True, exist_ok=True)
        
        # COCO 2017 (for coarse-grained image-text)
        datasets = [
            ("http://images.cocodataset.org/zips/train2017.zip", "train2017.zip", "COCO 2017 Train Images"),
            ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip", "COCO 2017 Val Images"),
            ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations_trainval2017.zip", "COCO 2017 Annotations"),
        ]
        
        # COCO 2014 (for fine-grained grounding)
        datasets.extend([
            ("http://images.cocodataset.org/zips/train2014.zip", "train2014.zip", "COCO 2014 Train Images (for grounding)"),
        ])
        
        for url, filename, desc in datasets:
            dest = coco_root / filename
            self.download_file(url, dest, desc)
            
            # Extract
            if dest.exists() and dest.suffix == '.zip':
                self.extract_archive(dest, coco_root)
                # Remove zip to save space (optional)
                # dest.unlink()
        
        print("\n✓ COCO dataset ready")
        print(f"  Location: {coco_root}")
        print(f"  Contents: train2017/, val2017/, train2014/, annotations/")
    
    def download_sbu(self):
        """Download SBU Captions dataset (~1M image-text pairs)"""
        print("\n" + "="*60)
        print("Downloading SBU Captions Dataset")
        print("="*60)
        
        sbu_root = self.data_root / "sbu"
        sbu_root.mkdir(parents=True, exist_ok=True)
        
        # SBU URLs and instructions
        print("\n📝 SBU Captions Download Instructions:")
        print("   The SBU Captions dataset requires manual download from:")
        print("   http://www.cs.virginia.edu/~vicente/sbucaptions/")
        print("\n   Steps:")
        print("   1. Download SBU_captioned_photo_dataset_urls.txt")
        print("   2. Download SBU_captioned_photo_dataset_captions.txt")
        print(f"   3. Place both files in: {sbu_root}/")
        print("   4. Run the provided download script to fetch images")
        print("\n   Note: ~1M images, ~400GB total size")
        print(f"\n   We will create a download script at: {sbu_root}/download_sbu_images.py")
        
        # Create helper script
        download_script = sbu_root / "download_sbu_images.py"
        with open(download_script, 'w') as f:
            f.write("""
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(url_caption_pair, output_dir, idx):
    url, caption = url_caption_pair
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            ext = url.split('.')[-1].split('?')[0]
            if ext not in ['jpg', 'jpeg', 'png', 'gif']:
                ext = 'jpg'
            filename = f"{idx:08d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

def main():
    # Read URLs and captions
    with open('SBU_captioned_photo_dataset_urls.txt', 'r') as f:
        urls = [line.strip() for line in f]
    
    with open('SBU_captioned_photo_dataset_captions.txt', 'r') as f:
        captions = [line.strip() for line in f]
    
    output_dir = 'images_train'
    os.makedirs(output_dir, exist_ok=True)
    
    # Download with threading
    url_caption_pairs = list(zip(urls, captions))
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_image, pair, output_dir, idx): idx 
                   for idx, pair in enumerate(url_caption_pairs)}
        
        with tqdm(total=len(url_caption_pairs), desc="Downloading SBU images") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
    
    print(f"Downloaded images to {output_dir}/")

if __name__ == "__main__":
    main()
""")
        
        print(f"✓ Created download script: {download_script}")
        print(f"  Run: cd {sbu_root} && python download_sbu_images.py")
    
    def download_visual_genome(self):
        """Download Visual Genome dataset (region descriptions + objects)"""
        print("\n" + "="*60)
        print("Downloading Visual Genome Dataset")
        print("="*60)
        
        vg_root = self.data_root / "visual_genome"
        vg_root.mkdir(parents=True, exist_ok=True)
        
        datasets = [
            ("https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "VG_100K.zip", "VG Images Part 1"),
            ("https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", "VG_100K_2.zip", "VG Images Part 2"),
            ("https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip", "region_descriptions.json.zip", "VG Region Descriptions"),
            ("https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip", "objects.json.zip", "VG Objects"),
        ]
        
        for url, filename, desc in datasets:
            dest = vg_root / filename
            try:
                self.download_file(url, dest, desc)
                if dest.exists() and dest.suffix == '.zip':
                    self.extract_archive(dest, vg_root)
            except Exception as e:
                print(f"⚠️  Failed to download {desc}: {e}")
                print(f"   Please manually download from: {url}")
        
        print("\n✓ Visual Genome dataset ready")
        print(f"  Location: {vg_root}")
        print(f"  Contents: VG_100K/, VG_100K_2/, region_descriptions.json, objects.json")
    
    def download_conceptual_captions(self):
        """Download Conceptual Captions 3M dataset"""
        print("\n" + "="*60)
        print("Downloading Conceptual Captions 3M Dataset")
        print("="*60)
        
        cc3m_root = self.data_root / "conceptual_captions"
        cc3m_root.mkdir(parents=True, exist_ok=True)
        
        print("\n📝 Conceptual Captions 3M Download Instructions:")
        print("   CC3M requires downloading from Google's TSV files:")
        print("   https://ai.google.com/research/ConceptualCaptions/download")
        print("\n   Steps:")
        print("   1. Download Train_GCC-training.tsv")
        print("   2. Download Validation_GCC-1.1.0-Validation.tsv")
        print(f"   3. Place TSV files in: {cc3m_root}/")
        print("   4. Run the provided download script to fetch images from URLs")
        
        # Create helper script
        download_script = cc3m_root / "download_cc3m_images.py"
        with open(download_script, 'w') as f:
            f.write("""
import os
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(row, output_dir, idx):
    caption, url = row['caption'], row['url']
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            ext = url.split('.')[-1].split('?')[0]
            if ext not in ['jpg', 'jpeg', 'png', 'gif']:
                ext = 'jpg'
            filename = f"{idx:08d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

def main():
    # Read TSV
    df = pd.read_csv('Train_GCC-training.tsv', sep='\\t', header=None, names=['caption', 'url'])
    
    output_dir = 'images_train'
    os.makedirs(output_dir, exist_ok=True)
    
    # Download with threading
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_image, row, output_dir, idx): idx 
                   for idx, row in df.iterrows()}
        
        with tqdm(total=len(df), desc="Downloading CC3M images") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
    
    print(f"Downloaded images to {output_dir}/")

if __name__ == "__main__":
    main()
""")
        
        print(f"✓ Created download script: {download_script}")
        print(f"  Run: cd {cc3m_root} && python download_cc3m_images.py")
    
    def download_flickr30k(self):
        """Download Flickr30k dataset (for evaluation)"""
        print("\n" + "="*60)
        print("Downloading Flickr30k Dataset")
        print("="*60)
        
        flickr_root = self.data_root / "flickr30k"
        flickr_root.mkdir(parents=True, exist_ok=True)
        
        print("\n📝 Flickr30k Download Instructions:")
        print("   Flickr30k requires manual request:")
        print("   http://shannon.cs.illinois.edu/DenotationGraph/")
        print("\n   Steps:")
        print("   1. Fill out the form to request dataset access")
        print("   2. Download flickr30k-images.tar")
        print(f"   3. Extract to: {flickr_root}/flickr30k_images/")
        print("\n   For MDETR annotations (grounding):")
        print("   Download: https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz")
        print(f"   Extract to: {self.data_root}/mdetr_annotations/")
    
    def download_refcoco(self):
        """Download RefCOCO, RefCOCO+, RefCOCOg datasets (fine-grained grounding)"""
        print("\n" + "="*60)
        print("Downloading RefCOCO/+/g Datasets (Fine-Grained Grounding)")
        print("="*60)
        
        refcoco_root = self.data_root / "refcoco"
        refcoco_root.mkdir(parents=True, exist_ok=True)
        
        # Download MDETR annotations (includes RefCOCO/+/g)
        mdetr_root = self.data_root / "mdetr_annotations"
        mdetr_root.mkdir(parents=True, exist_ok=True)
        
        mdetr_urls = [
            "https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz",
        ]
        
        for url in mdetr_urls:
            filename = url.split('/')[-1]
            dest = mdetr_root / filename
            try:
                self.download_file(url, dest, "MDETR Annotations (RefCOCO/+/g)")
                if dest.exists():
                    self.extract_archive(dest, mdetr_root)
            except Exception as e:
                print(f"⚠️  Failed to download MDETR annotations: {e}")
                print(f"   Please manually download from: {url}")
        
        print("\n✓ RefCOCO datasets ready")
        print(f"  Location: {mdetr_root}")
        print(f"  Contains: final_refcoco_train.json, final_refcoco+_train.json, final_refcocog_train.json")
        print("\n  Note: RefCOCO uses COCO 2014 images (download COCO first)")
    
    def download_mixed_grounding(self):
        """Download MixedGrounding dataset (MDETR curated)"""
        print("\n" + "="*60)
        print("Downloading MixedGrounding Dataset")
        print("="*60)
        
        mdetr_root = self.data_root / "mdetr_annotations"
        mdetr_root.mkdir(parents=True, exist_ok=True)
        
        # MixedGrounding annotation (no COCO)
        url = "https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/mdetr_annotations/final_mixed_train_no_coco.json"
        dest = mdetr_root / "final_mixed_train_no_coco.json"
        
        try:
            self.download_file(url, dest, "MixedGrounding Annotations (no COCO)")
        except Exception as e:
            print(f"⚠️  Failed to download MixedGrounding: {e}")
        
        # GQA images (part of MixedGrounding)
        gqa_root = self.data_root / "gqa"
        gqa_root.mkdir(parents=True, exist_ok=True)
        
        print("\n📝 GQA Images (for MixedGrounding):")
        print("   Download: https://nlp.stanford.edu/data/gqa/images.zip")
        print(f"   Extract to: {gqa_root}/images/")
        
        print("\n✓ MixedGrounding ready")
        print("  Requires: COCO 2014 train, GQA images")
    
    def create_dataset_structure_info(self):
        """Create a JSON file with dataset structure information"""
        info = {
            "coarse_grained": {
                "coco": {
                    "path": str(self.data_root / "coco"),
                    "splits": ["train2017", "val2017"],
                    "annotations": "annotations/captions_train2017.json",
                    "size": "~120k images",
                    "purpose": "Image-text contrastive learning"
                },
                "sbu": {
                    "path": str(self.data_root / "sbu"),
                    "splits": ["train"],
                    "size": "~1M images",
                    "purpose": "Image-text pairs for scaling"
                },
                "visual_genome": {
                    "path": str(self.data_root / "visual_genome"),
                    "splits": ["train"],
                    "annotations": "region_descriptions.json",
                    "size": "~108k images, ~5M regions",
                    "purpose": "Region descriptions for dense captioning"
                },
                "conceptual_captions": {
                    "path": str(self.data_root / "conceptual_captions"),
                    "splits": ["train", "val"],
                    "size": "~3M images",
                    "purpose": "Large-scale image-text pairs from web"
                },
                "flickr30k": {
                    "path": str(self.data_root / "flickr30k"),
                    "splits": ["train", "val", "test"],
                    "size": "~31k images",
                    "purpose": "Image-text retrieval evaluation"
                }
            },
            "fine_grained": {
                "refcoco": {
                    "path": str(self.data_root / "mdetr_annotations"),
                    "files": [
                        "final_refcoco_train.json",
                        "final_refcoco+_train.json",
                        "final_refcocog_train.json"
                    ],
                    "images": "coco/train2014",
                    "size": "~120k referring expressions",
                    "purpose": "Phrase grounding, referring expression comprehension"
                },
                "mixed_grounding": {
                    "path": str(self.data_root / "mdetr_annotations"),
                    "files": ["final_mixed_train_no_coco.json"],
                    "images": ["coco/train2014", "gqa/images"],
                    "size": "~1M grounding annotations",
                    "purpose": "Large-scale phrase grounding"
                },
                "visual_genome_boxes": {
                    "path": str(self.data_root / "visual_genome"),
                    "annotations": "objects.json",
                    "size": "~108k images, ~3.8M objects",
                    "purpose": "Object detection, region-level understanding"
                }
            }
        }
        
        info_file = self.data_root / "dataset_structure.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✓ Created dataset structure info: {info_file}")
        return info


def main():
    parser = argparse.ArgumentParser(description="Download FIBER-style datasets for BitGen")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["coco", "sbu", "vg", "cc3m", "flickr30k", "refcoco", "mixed_grounding"],
        help="Datasets to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Download all coarse-grained datasets (COCO, SBU, VG, CC3M, Flickr30k)"
    )
    parser.add_argument(
        "--fine",
        action="store_true",
        help="Download all fine-grained datasets (RefCOCO, MixedGrounding)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for datasets"
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(data_root=args.data_root)
    
    # Determine which datasets to download
    if args.all:
        datasets_to_download = ["coco", "sbu", "vg", "cc3m", "flickr30k", "refcoco", "mixed_grounding"]
    elif args.coarse:
        datasets_to_download = ["coco", "sbu", "vg", "cc3m", "flickr30k"]
    elif args.fine:
        datasets_to_download = ["refcoco", "mixed_grounding"]
    elif args.datasets:
        datasets_to_download = args.datasets
    else:
        print("Please specify --datasets, --all, --coarse, or --fine")
        parser.print_help()
        return
    
    print("\n" + "="*60)
    print("BitGen FIBER-Style Dataset Downloader")
    print("="*60)
    print(f"Data root: {downloader.data_root}")
    print(f"Downloading: {', '.join(datasets_to_download)}")
    print("="*60)
    
    # Download datasets
    if "coco" in datasets_to_download:
        downloader.download_coco()
    
    if "sbu" in datasets_to_download:
        downloader.download_sbu()
    
    if "vg" in datasets_to_download:
        downloader.download_visual_genome()
    
    if "cc3m" in datasets_to_download:
        downloader.download_conceptual_captions()
    
    if "flickr30k" in datasets_to_download:
        downloader.download_flickr30k()
    
    if "refcoco" in datasets_to_download:
        downloader.download_refcoco()
    
    if "mixed_grounding" in datasets_to_download:
        downloader.download_mixed_grounding()
    
    # Create dataset structure info
    downloader.create_dataset_structure_info()
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"\n📁 All datasets stored in: {downloader.data_root}")
    print("\n📝 Next Steps:")
    print("   1. For datasets requiring manual download, follow the instructions above")
    print("   2. Run data preprocessing scripts (if needed)")
    print(f"   3. Check {downloader.data_root}/dataset_structure.json for dataset paths")
    print("   4. Start training with: python src/train_stage1_vision_language.py")


if __name__ == "__main__":
    main()
