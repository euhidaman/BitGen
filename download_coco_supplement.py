#!/usr/bin/env python3
"""
COCO Dataset Supplement Download Script
Downloads COCO dataset from Kaggle to supplement existing OpenImages data
Integrates with existing vision features cache without re-downloading
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import pickle
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_existing_data(data_dir: str = "./data") -> Dict:
    """Check what data already exists"""
    data_path = Path(data_dir)
    
    status = {
        'openimages_exists': False,
        'openimages_count': 0,
        'localized_narratives_exists': False,
        'features_cache_exists': False,
        'coco_exists': False,
        'coco_count': 0,
        'existing_features_count': 0
    }
    
    # Check OpenImages
    oi_dir = data_path / "open_images"
    if oi_dir.exists():
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.webp', '*.WEBP']
        existing_images = []
        for ext in image_extensions:
            existing_images.extend(oi_dir.glob(f"**/{ext}"))
        status['openimages_exists'] = len(existing_images) > 0
        status['openimages_count'] = len(existing_images)
    
    # Check Localized Narratives
    ln_dir = data_path / "localized_narratives"
    if ln_dir.exists():
        caption_files = list(ln_dir.glob("**/*.jsonl"))
        status['localized_narratives_exists'] = len(caption_files) > 0
    
    # Check existing features cache
    cache_dir = data_path / "vision_features_cache"
    if cache_dir.exists():
        features_file = cache_dir / "all_features.npy"
        if features_file.exists():
            try:
                features = np.load(features_file)
                status['features_cache_exists'] = True
                status['existing_features_count'] = len(features)
            except:
                pass
    
    # Check COCO
    coco_dir = data_path / "coco"
    if coco_dir.exists():
        coco_images = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in image_extensions:
            coco_images.extend(coco_dir.glob(f"**/{ext}"))
        status['coco_exists'] = len(coco_images) > 0
        status['coco_count'] = len(coco_images)
    
    return status


def install_kagglehub():
    """Install kagglehub if not available"""
    try:
        import kagglehub
        logger.info("✅ kagglehub already installed")
        return True
    except ImportError:
        logger.info("📦 Installing kagglehub...")
        try:
            subprocess.run(["pip", "install", "kagglehub"], check=True)
            logger.info("✅ kagglehub installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install kagglehub: {e}")
            return False


def download_coco_dataset(data_dir: str = "./data") -> Dict:
    """Download COCO dataset from Kaggle"""
    
    if not install_kagglehub():
        return None
    
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError:
        logger.error("❌ kagglehub not available after installation")
        return None
    
    data_path = Path(data_dir)
    coco_dir = data_path / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Downloading COCO dataset from Kaggle...")
    
    try:
        # Download the dataset - this will download to kagglehub cache
        logger.info("📦 Downloading COCO image-caption dataset...")
        downloaded_path = kagglehub.download("nikhil7280/coco-image-caption")
        logger.info(f"✅ Dataset downloaded to: {downloaded_path}")
        
        # Copy to our data directory structure
        source_path = Path(downloaded_path)
        
        # Copy train2014 images
        train_source = source_path / "train2014" / "train2014"
        train_dest = coco_dir / "train2014"
        
        if train_source.exists() and not train_dest.exists():
            logger.info("📂 Copying train2014 images...")
            shutil.copytree(train_source, train_dest)
            logger.info("✅ train2014 images copied")
        elif train_dest.exists():
            logger.info("✅ train2014 images already exist")
        
        # Copy val2017 images (test set)
        val_source = source_path / "val2017" / "val2017"
        val_dest = coco_dir / "val2017"
        
        if val_source.exists() and not val_dest.exists():
            logger.info("📂 Copying val2017 images...")
            shutil.copytree(val_source, val_dest)
            logger.info("✅ val2017 images copied")
        elif val_dest.exists():
            logger.info("✅ val2017 images already exist")
        
        # Copy annotations
        ann_source = source_path / "annotations_trainval2014" / "annotations"
        ann_dest = coco_dir / "annotations_trainval2014"
        
        if ann_source.exists() and not ann_dest.exists():
            logger.info("📂 Copying annotations_trainval2014...")
            shutil.copytree(ann_source, ann_dest)
            logger.info("✅ annotations_trainval2014 copied")
        elif ann_dest.exists():
            logger.info("✅ annotations_trainval2014 already exist")
        
        # Copy val2017 annotations
        ann_source_2017 = source_path / "annotations_trainval2017" / "annotations"
        ann_dest_2017 = coco_dir / "annotations_trainval2017"
        
        if ann_source_2017.exists() and not ann_dest_2017.exists():
            logger.info("📂 Copying annotations_trainval2017...")
            shutil.copytree(ann_source_2017, ann_dest_2017)
            logger.info("✅ annotations_trainval2017 copied")
        elif ann_dest_2017.exists():
            logger.info("✅ annotations_trainval2017 already exist")
        
        return {
            'downloaded_path': downloaded_path,
            'coco_dir': str(coco_dir),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to download COCO dataset: {e}")
        return None


def process_coco_captions(data_dir: str = "./data") -> List[Dict]:
    """Process COCO captions and create aligned pairs"""
    
    data_path = Path(data_dir)
    coco_dir = data_path / "coco"
    
    if not coco_dir.exists():
        logger.error("❌ COCO directory not found")
        return []
    
    logger.info("📝 Processing COCO captions...")
    
    # Process train2014 captions
    trainval_image_dir = coco_dir / "train2014"
    trainval_captions_dir = coco_dir / "annotations_trainval2014"
    trainval_captions_filepath = trainval_captions_dir / "captions_train2014.json"
    
    aligned_pairs = []
    
    if trainval_captions_filepath.exists() and trainval_image_dir.exists():
        logger.info("📖 Processing train2014 captions...")
        
        with open(trainval_captions_filepath, 'r') as f:
            trainval_data = json.load(f)
        
        # Create DataFrame from annotations
        trainval_captions_df = pd.json_normalize(trainval_data, "annotations")
        trainval_captions_df["image_filepath"] = trainval_captions_df["image_id"].apply(
            lambda x: trainval_image_dir / f'COCO_train2014_{x:012d}.jpg'
        )
        
        # Filter to only images that actually exist
        existing_images = []
        for _, row in trainval_captions_df.iterrows():
            image_path = Path(row["image_filepath"])
            if image_path.exists():
                existing_images.append(row)
        
        logger.info(f"   Found {len(existing_images):,} valid train2014 image-caption pairs")
        
        # Process captions
        for item in existing_images:
            caption = item["caption"]
            # Preprocess caption (remove special chars, lowercase)
            processed_caption = caption.lower().strip()
            
            aligned_pairs.append({
                'image_id': f'coco_train_{item["image_id"]}',
                'image_path': str(item["image_filepath"]),
                'caption': processed_caption,
                'original_caption': caption,
                'dataset': 'coco_train2014'
            })
    
    # Process val2017 captions (test set)
    test_image_dir = coco_dir / "val2017"
    test_captions_dir = coco_dir / "annotations_trainval2017"
    test_captions_filepath = test_captions_dir / "captions_val2017.json"
    
    if test_captions_filepath.exists() and test_image_dir.exists():
        logger.info("📖 Processing val2017 captions...")
        
        with open(test_captions_filepath, 'r') as f:
            test_data = json.load(f)
        
        # Create DataFrame from annotations
        test_captions_df = pd.json_normalize(test_data, "annotations")
        test_captions_df["image_filepath"] = test_captions_df["image_id"].apply(
            lambda x: test_image_dir / f'{x:012d}.jpg'
        )
        
        # Filter to only images that actually exist
        existing_test_images = []
        for _, row in test_captions_df.iterrows():
            image_path = Path(row["image_filepath"])
            if image_path.exists():
                existing_test_images.append(row)
        
        logger.info(f"   Found {len(existing_test_images):,} valid val2017 image-caption pairs")
        
        # Process captions
        for item in existing_test_images:
            caption = item["caption"]
            # Preprocess caption
            processed_caption = caption.lower().strip()
            
            aligned_pairs.append({
                'image_id': f'coco_val_{item["image_id"]}',
                'image_path': str(item["image_filepath"]),
                'caption': processed_caption,
                'original_caption': caption,
                'dataset': 'coco_val2017'
            })
    
    logger.info(f"✅ Total COCO aligned pairs: {len(aligned_pairs):,}")
    return aligned_pairs


def extract_coco_features(aligned_pairs: List[Dict], data_dir: str = "./data") -> np.ndarray:
    """Extract DiNOv2 features for COCO images"""
    
    if not aligned_pairs:
        logger.warning("⚠️ No aligned pairs to process")
        return np.array([])
    
    logger.info(f"🧠 Extracting DiNOv2 features for {len(aligned_pairs):,} COCO images...")
    
    try:
        import torch
        from transformers import Dinov2Model, AutoImageProcessor
        from PIL import Image
        
        # Load DiNOv2
        logger.info("   Loading DiNOv2 model...")
        model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        
        if torch.cuda.is_available():
            model = model.cuda()
            device = "cuda"
            logger.info(f"   Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("   Using CPU")
        
        model.eval()
        
        # Extract features in batches
        batch_size = 16
        all_features = []
        valid_pairs = []
        
        logger.info(f"   Processing in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(aligned_pairs), batch_size), desc="COCO Features"):
            batch = aligned_pairs[i:i+batch_size]
            
            # Load batch images
            batch_images = []
            batch_valid_pairs = []
            
            for pair in batch:
                try:
                    img = Image.open(pair['image_path']).convert('RGB')
                    batch_images.append(img)
                    batch_valid_pairs.append(pair)
                except Exception as e:
                    logger.debug(f"Skipping image {pair['image_path']}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Extract features
            try:
                inputs = processor(images=batch_images, return_tensors="pt")
                if device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)  # [batch, 768]
                
                all_features.extend(features.cpu().numpy())
                valid_pairs.extend(batch_valid_pairs)
                
            except Exception as e:
                logger.warning(f"Failed to extract features for batch: {e}")
                continue
        
        logger.info(f"   ✅ Extracted features for {len(all_features):,} COCO images")
        return np.array(all_features), valid_pairs
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies for feature extraction: {e}")
        return np.array([]), []


def integrate_with_existing_data(coco_features: np.ndarray, coco_pairs: List[Dict], data_dir: str = "./data") -> Dict:
    """Integrate COCO data with existing OpenImages data"""
    
    data_path = Path(data_dir)
    cache_dir = data_path / "vision_features_cache"
    cache_dir.mkdir(exist_ok=True)
    
    logger.info("🔗 Integrating COCO data with existing dataset...")
    
    # Load existing features if they exist
    existing_features_file = cache_dir / "all_features.npy"
    existing_pairs_file = data_path / "aligned_pairs.json"
    
    existing_features = np.array([])
    existing_pairs = []
    
    if existing_features_file.exists():
        try:
            existing_features = np.load(existing_features_file)
            logger.info(f"   📂 Loaded {len(existing_features):,} existing features")
        except Exception as e:
            logger.warning(f"Failed to load existing features: {e}")
    
    if existing_pairs_file.exists():
        try:
            with open(existing_pairs_file, 'r') as f:
                existing_pairs = json.load(f)
            logger.info(f"   📂 Loaded {len(existing_pairs):,} existing pairs")
        except Exception as e:
            logger.warning(f"Failed to load existing pairs: {e}")
    
    # Combine features
    if len(existing_features) > 0 and len(coco_features) > 0:
        combined_features = np.vstack([existing_features, coco_features])
        logger.info(f"   🔄 Combined features: {len(existing_features):,} + {len(coco_features):,} = {len(combined_features):,}")
    elif len(coco_features) > 0:
        combined_features = coco_features
        logger.info(f"   ➕ Using COCO features only: {len(coco_features):,}")
    elif len(existing_features) > 0:
        combined_features = existing_features
        logger.info(f"   📂 Keeping existing features only: {len(existing_features):,}")
    else:
        combined_features = np.array([])
        logger.warning("   ⚠️ No features to combine")
    
    # Combine pairs
    combined_pairs = existing_pairs + coco_pairs
    logger.info(f"   📝 Combined pairs: {len(existing_pairs):,} + {len(coco_pairs):,} = {len(combined_pairs):,}")
    
    # Save combined data
    if len(combined_features) > 0:
        # Save features
        np.save(existing_features_file, combined_features)
        logger.info(f"   💾 Saved combined features: {combined_features.shape}")
        
        # Save metadata
        metadata = {
            'num_samples': len(combined_features),
            'feature_shape': combined_features.shape[1:],
            'feature_type': 'real_dinov2',
            'vision_model': 'facebook/dinov2-base',
            'use_dummy_vision': False,
            'extract_vision_features': True,
            'datasets': ['openimages', 'coco'],
            'openimages_count': len(existing_pairs),
            'coco_count': len(coco_pairs),
            'created_by': 'download_coco_supplement.py'
        }
        
        metadata_file = cache_dir / "cache_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save combined pairs
        with open(existing_pairs_file, 'w') as f:
            json.dump(combined_pairs, f, indent=2)
        
        # Also save COCO-specific pairs
        coco_pairs_file = data_path / "coco_aligned_pairs.json"
        with open(coco_pairs_file, 'w') as f:
            json.dump(coco_pairs, f, indent=2)
        
        # Save summary
        summary = {
            'total_samples': len(combined_features),
            'openimages_samples': len(existing_pairs),
            'coco_samples': len(coco_pairs),
            'feature_shape': combined_features.shape,
            'datasets_included': ['openimages', 'coco'],
            'ready_for_training': True
        }
        
        summary_file = data_path / "combined_dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✅ Dataset integration complete!")
        logger.info(f"   📊 Total samples: {len(combined_features):,}")
        logger.info(f"   🖼️ OpenImages: {len(existing_pairs):,}")
        logger.info(f"   🖼️ COCO: {len(coco_pairs):,}")
        logger.info(f"   💾 Ready for training with train_unified.py")
        
        return summary
    
    else:
        logger.error("❌ No features to save")
        return None


def main():
    """Main function to download and integrate COCO dataset"""
    
    logger.info("🚀 COCO Dataset Supplement Download")
    logger.info("=" * 50)
    
    # Check existing data
    logger.info("🔍 Checking existing data...")
    status = check_existing_data()
    
    logger.info(f"   OpenImages: {status['openimages_count']:,} images" if status['openimages_exists'] else "   OpenImages: Not found")
    logger.info(f"   Localized Narratives: {'✅' if status['localized_narratives_exists'] else '❌'}")
    logger.info(f"   Features Cache: {status['existing_features_count']:,} features" if status['features_cache_exists'] else "   Features Cache: Not found")
    logger.info(f"   COCO: {status['coco_count']:,} images" if status['coco_exists'] else "   COCO: Not found")
    
    # Download COCO if not exists
    if not status['coco_exists']:
        logger.info("\n📦 Downloading COCO dataset...")
        download_result = download_coco_dataset()
        
        if not download_result or not download_result['success']:
            logger.error("❌ Failed to download COCO dataset")
            return
        
        logger.info("✅ COCO dataset downloaded successfully")
    else:
        logger.info("\n✅ COCO dataset already exists")
    
    # Process COCO captions
    logger.info("\n📝 Processing COCO captions...")
    coco_pairs = process_coco_captions()
    
    if not coco_pairs:
        logger.error("❌ No COCO pairs found")
        return
    
    # Extract COCO features
    logger.info("\n🧠 Extracting COCO features...")
    coco_features, valid_coco_pairs = extract_coco_features(coco_pairs)
    
    if len(coco_features) == 0:
        logger.error("❌ Failed to extract COCO features")
        return
    
    # Integrate with existing data
    logger.info("\n🔗 Integrating with existing dataset...")
    integration_result = integrate_with_existing_data(coco_features, valid_coco_pairs)
    
    if integration_result:
        logger.info("\n🎉 SUCCESS!")
        logger.info(f"   📊 Total dataset size: {integration_result['total_samples']:,} samples")
        logger.info(f"   🖼️ OpenImages contribution: {integration_result['openimages_samples']:,}")
        logger.info(f"   🖼️ COCO contribution: {integration_result['coco_samples']:,}")
        logger.info(f"   🧠 Feature dimensions: {integration_result['feature_shape']}")
        logger.info(f"   💾 All data saved to: ./data/")
        logger.info(f"\n🚀 Ready to train with: python train_unified.py --config configs/bitmar_with_memory.yaml")
    else:
        logger.error("❌ Integration failed")


if __name__ == "__main__":
    main()
