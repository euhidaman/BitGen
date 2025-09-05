#!/usr/bin/env python3
"""
Official OpenImages downloader using the proper CVDF method
Downloads images from the official mirrors instead of broken Flickr URLs
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_official_openimages_downloader():
    """Download the official OpenImages downloader script"""
    downloader_url = "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
    downloader_path = Path("downloader.py")
    
    logger.info("Downloading official OpenImages downloader...")
    response = requests.get(downloader_url)
    response.raise_for_status()
    
    with open(downloader_path, 'w') as f:
        f.write(response.text)
    
    logger.info("✅ Official downloader downloaded")
    return downloader_path

def create_image_list_file(image_ids, output_file="image_list.txt"):
    """Create image list file in the format required by official downloader"""
    output_path = Path(output_file)
    
    logger.info(f"Creating image list file with {len(image_ids)} images...")
    
    with open(output_path, 'w') as f:
        for image_id in image_ids:
            # Format: train/IMAGE_ID (most images are from train set)
            f.write(f"train/{image_id}\n")
    
    logger.info(f"✅ Image list created: {output_path}")
    return output_path

def download_with_official_downloader(image_list_file, download_folder="data/open_images", num_processes=5):
    """Use the official OpenImages downloader"""
    download_path = Path(download_folder)
    download_path.mkdir(parents=True, exist_ok=True)
    
    downloader_path = download_official_openimages_downloader()
    
    logger.info(f"Starting download with official downloader...")
    logger.info(f"Images will be saved to: {download_path}")
    
    cmd = [
        sys.executable, str(downloader_path),
        str(image_list_file),
        f"--download_folder={download_path}",
        f"--num_processes={num_processes}"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ Download completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        raise

def main():
    """Main function to download OpenImages using official method"""
    # You can customize this list - these are sample image IDs
    # In practice, you'd get these from LocalizedNarratives annotations
    sample_image_ids = [
        "f9e0434389a1d4dd",
        "1a007563ebc18664", 
        "ea8bfd4e765304db",
        # Add more image IDs here...
    ]
    
    # For now, let's download just a few images to test
    logger.info("Creating test download with official OpenImages downloader...")
    
    image_list_file = create_image_list_file(sample_image_ids)
    download_with_official_downloader(image_list_file, num_processes=2)

if __name__ == "__main__":
    main()
