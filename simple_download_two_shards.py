#!/usr/bin/env python3
"""
Simple script: Download first two OpenImages shards + extract DiNOv2 features
Just does what's needed - no complexity.
"""

import os
import json
import subprocess
import tarfile
import logging
from pathlib import Path
from tqdm import tqdm
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def download_two_shards_with_features(data_dir: str = "./data", max_samples: int = 200000):
    """
    Simple function:
    1. Download train_0.tar.gz + train_1.tar.gz (first two OpenImages shards)
    2. Extract images
    3. Download OpenImages captions from Localized Narratives
    4. Align captions with images
    5. Extract DiNOv2 features
    """
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    logger.info("🚀 Starting simple two-shard download with DiNOv2 features")
    
    # Step 1: Download Localized Narratives captions for OpenImages
    logger.info("📝 Step 1: Download OpenImages captions from Localized Narratives")
    
    ln_dir = data_path / "localized_narratives"
    ln_dir.mkdir(exist_ok=True)
    
    # Download first two caption shards (enough for our image shards)
    caption_urls = [
        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00000-of-00010.jsonl',
        'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00001-of-00010.jsonl'
    ]
    
    captions = []
    for i, url in enumerate(caption_urls):
        filename = f"captions_shard_{i}.jsonl"
        filepath = ln_dir / filename
        
        if not filepath.exists():
            logger.info(f"   Downloading {filename}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
        
        # Parse captions
        logger.info(f"   Processing {filename}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if len(captions) >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    if data.get('image_id') and data.get('caption'):
                        captions.append({
                            'image_id': data['image_id'],
                            'caption': data['caption']
                        })
                except:
                    continue
        
        if len(captions) >= max_samples:
            break
    
    logger.info(f"   ✅ Got {len(captions):,} captions")
    
    # Step 2: Download first two OpenImages shards
    logger.info("📦 Step 2: Download first two OpenImages shards")
    
    oi_dir = data_path / "open_images"
    oi_dir.mkdir(exist_ok=True)
    
    # AWS commands for first two shards
    shard_commands = [
        f"aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_0.tar.gz {oi_dir}/",
        f"aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_1.tar.gz {oi_dir}/"
    ]
    
    for i, cmd in enumerate(shard_commands):
        tar_file = oi_dir / f"train_{i}.tar.gz"
        
        # Download if not exists
        if not tar_file.exists():
            logger.info(f"   Downloading train_{i}.tar.gz (this will take time)...")
            try:
                subprocess.run(cmd.split(), check=True)
                logger.info(f"   ✅ Downloaded train_{i}.tar.gz")
            except subprocess.CalledProcessError:
                logger.error(f"   ❌ Failed to download train_{i}.tar.gz")
                continue
        else:
            logger.info(f"   train_{i}.tar.gz already exists")
        
        # Extract shard
        if tar_file.exists():
            logger.info(f"   Extracting train_{i}.tar.gz...")
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(oi_dir)
            
            # Remove tar file to save space
            tar_file.unlink()
            logger.info(f"   ✅ Extracted train_{i}.tar.gz")
    
    # Count extracted images
    image_files = list(oi_dir.glob("**/*.jpg"))
    logger.info(f"   📁 Total images extracted: {len(image_files):,}")
    
    # Step 3: Align captions with available images
    logger.info("🔗 Step 3: Align captions with images")
    
    # Create image_id to path mapping
    image_map = {}
    for img_file in image_files:
        image_id = img_file.stem
        image_map[image_id] = str(img_file)
    
    # Find aligned pairs
    aligned_pairs = []
    for caption_data in captions:
        image_id = caption_data['image_id']
        if image_id in image_map:
            aligned_pairs.append({
                'image_id': image_id,
                'image_path': image_map[image_id],
                'caption': caption_data['caption']
            })
    
    logger.info(f"   ✅ Aligned {len(aligned_pairs):,} caption-image pairs")
    
    # Step 4: Extract DiNOv2 features
    logger.info("🧠 Step 4: Extract DiNOv2 features")
    
    try:
        import torch
        from transformers import Dinov2Model, AutoImageProcessor
        from PIL import Image
        import numpy as np
        
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
        
        # Extract features
        features_dir = data_path / "features"
        features_dir.mkdir(exist_ok=True)
        
        logger.info(f"   Extracting features for {len(aligned_pairs):,} images...")
        
        batch_size = 16
        all_features = []
        
        for i in tqdm(range(0, len(aligned_pairs), batch_size), desc="Features"):
            batch = aligned_pairs[i:i+batch_size]
            
            # Load batch images
            batch_images = []
            for pair in batch:
                try:
                    img = Image.open(pair['image_path']).convert('RGB')
                    batch_images.append(img)
                except:
                    # Skip broken images
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
                
            except Exception as e:
                logger.warning(f"Failed to extract features for batch: {e}")
                continue
        
        # Save everything
        logger.info("💾 Step 5: Save results")
        
        # Save aligned pairs
        pairs_file = data_path / "aligned_pairs.json"
        with open(pairs_file, 'w') as f:
            json.dump(aligned_pairs[:len(all_features)], f, indent=2)
        
        # Save features
        features_file = data_path / "features.npy"
        features_array = np.array(all_features)
        np.save(features_file, features_array)
        
        # Save summary
        summary = {
            'total_images': len(image_files),
            'aligned_pairs': len(aligned_pairs),
            'features_extracted': len(all_features),
            'feature_shape': features_array.shape,
            'shards_downloaded': ['train_0', 'train_1']
        }
        
        summary_file = data_path / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("🎉 DONE!")
        logger.info(f"   📊 Images: {len(image_files):,}")
        logger.info(f"   📝 Aligned pairs: {len(aligned_pairs):,}")
        logger.info(f"   🧠 Features: {len(all_features):,} x {features_array.shape[1]}")
        logger.info(f"   💾 Saved to: {data_path}")
        
        return summary
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.error("Install with: pip install torch transformers pillow")
        return None


if __name__ == "__main__":
    # Simple usage
    summary = download_two_shards_with_features(
        data_dir="./data_simple",
        max_samples=200000
    )
    
    if summary:
        print(f"\n✅ SUCCESS:")
        print(f"  • {summary['features_extracted']:,} image-caption pairs with DiNOv2 features")
        print(f"  • Features shape: {summary['feature_shape']}")
        print(f"  • Files saved to: ./data_simple/")
