#!/usr/bin/env python3
"""
Updated download_multimodal_data.py with official AWS S3 commands for OpenImages
Uses official shard-based downloads with proper progress tracking
"""

# Just the fixed _download_open_images_by_ids method
def _download_open_images_by_ids(self, image_ids: List[str]):
    """Download Open Images using official AWS S3 commands with shard-based progress tracking"""
    logger.info(f"🖼️ Processing {len(image_ids)} Open Images...")

    # Create Open Images download directory
    oi_dir = self.data_dir / "open_images"
    oi_dir.mkdir(exist_ok=True)

    # Quick pre-check: count existing images first
    logger.info("🔍 Quick check: scanning existing images...")
    existing_files = list(oi_dir.glob("**/*.jpg"))  # Include subdirectories
    existing_ids = {f.stem for f in existing_files if f.stat().st_size > 0}
    needed_ids = [img_id for img_id in image_ids if img_id not in existing_ids]

    logger.info(f"📊 Found {len(existing_ids)} existing images in folder")
    logger.info(f"📊 Need to download {len(needed_ids)} more images")

    if len(needed_ids) == 0:
        logger.info("✅ All images already exist!")
        return len(existing_ids)

    # Use official AWS CLI commands for OpenImages dataset
    logger.info("🚀 Using official AWS S3 commands for OpenImages dataset")
    logger.info("📋 Following official OpenImages download procedure")
    
    try:
        import subprocess
        import time
        import shutil
        import tarfile
        
        # Check if AWS CLI is available
        result = subprocess.run(["aws", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ AWS CLI detected - proceeding with official download")
            logger.info(f"🔧 AWS CLI version: {result.stdout.strip()}")
            
            # Test AWS connection first
            logger.info("🧪 Testing AWS S3 connection...")
            test_cmd = ["aws", "s3", "--no-sign-request", "ls", "s3://open-images-dataset/tar/"]
            try:
                test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
                if test_result.returncode == 0:
                    logger.info("✅ AWS S3 connection successful!")
                    available_files = test_result.stdout.strip().split('\n')[:5]
                    logger.info(f"📋 Available tar files: {len(available_files)} found")
                    for file_line in available_files:
                        if 'train_' in file_line:
                            logger.info(f"   {file_line}")
                else:
                    logger.warning(f"⚠️ AWS S3 test failed: {test_result.stderr}")
                    logger.info("Proceeding anyway - might be a temporary issue...")
            except Exception as e:
                logger.warning(f"⚠️ AWS S3 test error: {e}")
                logger.info("Proceeding anyway...")
            
            # Define official shard downloads (using first few shards to get substantial dataset)
            # Each shard contains images with IDs starting with that character/digit
            shard_configs = [
                {"name": "train_0", "size": "46GB", "desc": "Images with IDs starting with '0'"},
                {"name": "train_1", "size": "34GB", "desc": "Images with IDs starting with '1'"},
                {"name": "train_2", "size": "33GB", "desc": "Images with IDs starting with '2'"},
                {"name": "train_3", "size": "32GB", "desc": "Images with IDs starting with '3'"},
            ]
            
            logger.info(f"📦 Downloading {len(shard_configs)} OpenImages training shards")
            logger.info("🎯 This will provide 200K+ images covering multiple ID prefixes")
            
            successful_shards = 0
            
            for shard_idx, shard in enumerate(shard_configs):
                shard_name = shard["name"]
                expected_size = shard["size"]
                description = shard["desc"]
                
                tar_filename = f"{shard_name}.tar.gz"
                tar_filepath = oi_dir / tar_filename
                
                logger.info(f"\n🔽 [{shard_idx+1}/{len(shard_configs)}] Downloading {shard_name}.tar.gz ({expected_size})")
                logger.info(f"📝 {description}")
                
                # Skip if already downloaded and extracted
                if tar_filepath.exists() and tar_filepath.stat().st_size > 1024*1024*1024:  # >1GB
                    logger.info(f"✅ {tar_filename} already exists ({tar_filepath.stat().st_size / (1024**3):.1f} GB)")
                else:
                    # AWS S3 download command (let AWS CLI show its own progress)
                    aws_cmd = [
                        "aws", "s3", "--no-sign-request", "cp",
                        f"s3://open-images-dataset/tar/{tar_filename}",
                        str(tar_filepath)
                    ]
                    
                    logger.info(f"🚀 Starting AWS S3 download: {' '.join(aws_cmd)}")
                    logger.info("⏱️ Large file download - AWS CLI will show its own progress")
                    
                    try:
                        # Run AWS CLI command and let it show its own progress
                        start_time = time.time()
                        
                        # Use subprocess.run to let AWS CLI show its progress directly
                        result = subprocess.run(aws_cmd, cwd=str(oi_dir))
                        
                        elapsed = time.time() - start_time
                        
                        if result.returncode == 0 and tar_filepath.exists():
                            size_gb = tar_filepath.stat().st_size / (1024**3)
                            logger.info(f"✅ Downloaded {tar_filename} ({size_gb:.1f} GB) in {elapsed/60:.1f} minutes")
                        else:
                            logger.warning(f"❌ Failed to download {tar_filename} (return code: {result.returncode})")
                            continue
                            
                    except Exception as e:
                        logger.warning(f"❌ AWS download error for {tar_filename}: {e}")
                        continue
                
                # Extract the tar file
                if tar_filepath.exists():
                    logger.info(f"📦 Extracting {tar_filename}...")
                    
                    try:
                        extract_start = time.time()
                        
                        with tarfile.open(tar_filepath, 'r:gz') as tar:
                            # Extract to train subdirectory to match OpenImages structure
                            train_dir = oi_dir / "train"
                            train_dir.mkdir(exist_ok=True)
                            
                            members = tar.getmembers()
                            total_members = len(members)
                            logger.info(f"🗂️ Extracting {total_members:,} files from {tar_filename}")
                            
                            for i, member in enumerate(members):
                                tar.extract(member, oi_dir)
                                
                                # Show progress every 5000 files
                                if i % 5000 == 0 and i > 0:
                                    progress_pct = (i / total_members) * 100
                                    logger.info(f"   📁 Extracted {i:,}/{total_members:,} files ({progress_pct:.1f}%)")
                        
                        extract_time = time.time() - extract_start
                        logger.info(f"✅ Extracted {tar_filename} in {extract_time/60:.1f} minutes")
                        
                        # Remove tar file to save disk space
                        logger.info(f"🗑️ Removing {tar_filename} to save disk space...")
                        tar_filepath.unlink()
                        
                        successful_shards += 1
                        
                    except Exception as e:
                        logger.warning(f"❌ Failed to extract {tar_filename}: {e}")
                        continue
                
                # Count images after each shard
                current_files = list(oi_dir.glob("**/*.jpg"))
                current_count = len([f for f in current_files if f.stat().st_size > 0])
                logger.info(f"📊 Total images after shard {shard_idx+1}: {current_count:,}")
            
            logger.info(f"\n🎉 Completed downloading {successful_shards}/{len(shard_configs)} shards")
            
            # Final count
            final_files = list(oi_dir.glob("**/*.jpg"))
            valid_files = [f for f in final_files if f.stat().st_size > 0]
            total_images = len(valid_files)
            
            logger.info(f"✅ Total OpenImages dataset: {total_images:,} images")
            
            if total_images > len(existing_ids):
                new_images = total_images - len(existing_ids)
                logger.info(f"🎯 Successfully downloaded {new_images:,} new images")
                return total_images
            else:
                logger.info("⚠️ No new images added, trying FiftyOne fallback...")
        
        else:
            logger.warning("⚠️ AWS CLI not found")
            logger.info("📋 Install AWS CLI with: pip install awscli")
            logger.info("🔄 Trying FiftyOne method instead...")
            
    except Exception as e:
        logger.warning(f"❌ AWS method failed: {e}")
    
    # METHOD 2: FiftyOne fallback (PROVEN TO WORK)
    logger.info("🚀 Fallback: FiftyOne dataset zoo download")
    logger.info("📥 This is slower but very reliable for smaller datasets")
    
    try:
        import fiftyone.zoo as foz
        
        logger.info("📥 Using FiftyOne to download OpenImages subset...")
        
        # Download a substantial subset using FiftyOne
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            max_samples=min(50000, len(needed_ids)),
            only_matching=False,
            dataset_dir=str(oi_dir / "fiftyone_cache")
        )
        
        logger.info(f"✅ FiftyOne downloaded {len(dataset)} images")
        
        # Copy images from FiftyOne cache to our format
        fo_image_dir = oi_dir / "fiftyone_cache" / "open-images-v7" / "train" / "data"
        if fo_image_dir.exists():
            logger.info("📁 Copying images from FiftyOne cache...")
            
            copied = 0
            for img_file in fo_image_dir.glob("*.jpg"):
                target_file = oi_dir / img_file.name
                if not target_file.exists():
                    shutil.copy2(img_file, target_file)
                    copied += 1
            
            logger.info(f"✅ Copied {copied} images from FiftyOne")
            
            # Count final results
            final_files = list(oi_dir.glob("**/*.jpg"))
            return len([f for f in final_files if f.stat().st_size > 0])
        
    except ImportError:
        logger.warning("⚠️ FiftyOne not installed - install with: pip install fiftyone")
    except Exception as e:
        logger.warning(f"❌ FiftyOne method failed: {e}")
    
    # Final count attempt
    final_files = list(oi_dir.glob("**/*.jpg"))
    final_count = len([f for f in final_files if f.stat().st_size > 0])
    
    logger.info(f"📊 Final image count: {final_count:,}")
    
    if final_count < 50000:
        logger.warning("⚠️ Low image count. Recommendations:")
        logger.warning("   1. Install AWS CLI: pip install awscli")
        logger.warning("   2. Check internet connection for large downloads")
        logger.warning("   3. AWS method can download 200K+ images (requires ~150GB disk space)")
        logger.warning("   4. Each training shard contains 50K-100K images")
    
    return final_count
