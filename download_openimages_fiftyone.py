#!/usr/bin/env python3
"""
Download OpenImages using FiftyOne - the recommended method
This automatically handles all the download and annotation alignment
"""

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_with_fiftyone(max_samples=15000, split="train"):
    """Download OpenImages using FiftyOne - handles everything automatically"""
    
    if not FIFTYONE_AVAILABLE:
        logger.error("FiftyOne not available. Install with: pip install fiftyone")
        logger.error("Then run: python download_openimages_fiftyone.py")
        return None
    
    logger.info(f"Downloading OpenImages with FiftyOne - {max_samples} samples from {split}")
    logger.info("This will automatically download images AND align with captions/annotations")
    
    # Download OpenImages with localized narratives
    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split=split,
        label_types=["detections", "classifications"],  # Include annotations
        max_samples=max_samples,
        dataset_dir="data/fiftyone_openimages"  # Custom location
    )
    
    logger.info(f"✅ Downloaded {len(dataset)} images with FiftyOne")
    logger.info(f"Dataset location: {dataset.dataset_dir}")
    
    # Export to format compatible with your training
    export_dir = Path("data/openimages_fiftyone_export")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Exporting to training-compatible format...")
    
    # Export images and create caption file
    images_dir = export_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    captions = []
    
    for sample in dataset:
        # Copy image
        image_path = Path(sample.filepath)
        new_image_path = images_dir / f"{sample.id}.jpg"
        
        if image_path.exists():
            import shutil
            shutil.copy2(image_path, new_image_path)
            
            # Create caption from classifications (simplified)
            caption_parts = []
            if hasattr(sample, 'ground_truth') and sample.ground_truth:
                for detection in sample.ground_truth.detections:
                    caption_parts.append(detection.label)
            
            caption = f"An image containing {', '.join(caption_parts[:5])}" if caption_parts else "An image"
            
            captions.append({
                "image_id": sample.id,
                "image_path": str(new_image_path),
                "caption": caption
            })
    
    # Save captions
    import json
    captions_file = export_dir / "captions.json"
    with open(captions_file, 'w') as f:
        json.dump(captions, f, indent=2)
    
    logger.info(f"✅ Exported {len(captions)} image-caption pairs")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Captions: {captions_file}")
    
    return export_dir

def main():
    """Main function"""
    if not FIFTYONE_AVAILABLE:
        print("❌ FiftyOne not installed")
        print("Install with: pip install fiftyone")
        print("Then run this script again")
        return
    
    # Download 15K samples to start
    export_dir = download_with_fiftyone(max_samples=15000)
    
    if export_dir:
        print(f"✅ OpenImages download complete!")
        print(f"Next steps:")
        print(f"1. Images are in: {export_dir}/images/")
        print(f"2. Captions are in: {export_dir}/captions.json")
        print(f"3. Modify your training script to use this data")

if __name__ == "__main__":
    main()
