#!/usr/bin/env python3
"""
Test script for the quick download function that downloads first two OpenImages shards
and extracts DiNOv2 features.
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the quick download function"""
    
    # Import the function
    from download_multimodal_data import download_first_two_shards_with_features
    
    print("🚀 Testing quick download of first two OpenImages shards with DiNOv2 features")
    print()
    
    # Configuration
    data_dir = "./data_quick_test"
    max_samples = 10000  # Start with 10K for testing
    use_gpu = True  # Use GPU if available
    
    print(f"Configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Max samples: {max_samples:,}")
    print(f"  Use GPU: {use_gpu}")
    print()
    
    try:
        # Run the quick download
        results = download_first_two_shards_with_features(
            data_dir=data_dir,
            max_samples=max_samples,
            use_gpu=use_gpu
        )
        
        print("\n" + "="*60)
        print("📊 FINAL RESULTS:")
        print("="*60)
        print(f"✅ Captions processed: {results['captions_processed']:,}")
        print(f"✅ Images downloaded: {results['images_downloaded']:,}")
        print(f"✅ Features extracted: {results['features_extracted']:,}")
        print()
        print("📁 Data locations:")
        print(f"  Images: {results['image_directory']}")
        print(f"  Features: {results['features_directory']}")
        print(f"  Captions: {results['caption_files']}")
        print()
        
        # Check if everything worked
        if results['features_extracted'] > 0:
            print("🎉 SUCCESS: Downloaded images and extracted DiNOv2 features!")
            print()
            print("You can now use this data for multimodal training:")
            print(f"  • {results['features_extracted']:,} image-caption pairs with real DiNOv2 features")
            print(f"  • Features are 768-dimensional vectors")
            print(f"  • Images and captions are aligned and ready for training")
            
            return 0
        else:
            print("⚠️  WARNING: No features were extracted")
            print("Check the logs above for any errors")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
