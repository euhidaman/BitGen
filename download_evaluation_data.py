"""
Download BabyLM Challenge Evaluation Data
Downloads evaluation data from the official OSF repository for both 2024 and 2025 evaluation pipelines
URL: https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819f54f5dc6fc2bff0a7bba/?zip=
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import logging
from tqdm import tqdm
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, filepath: str, chunk_size: int = 8192) -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"✅ Downloaded: {filepath}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract ZIP file and handle nested directory structure"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # First extract to a temporary location to inspect structure
            temp_extract = Path(extract_to) / "temp_extract"
            temp_extract.mkdir(exist_ok=True)

            zip_ref.extractall(temp_extract)

            # Check the structure and move files appropriately
            extracted_items = list(temp_extract.iterdir())

            # If there's a single directory containing everything, use that
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                source_dir = extracted_items[0]
                logger.info(f"Found nested directory: {source_dir.name}")

                # Move contents from nested directory to target location
                target_dir = Path(extract_to)
                for item in source_dir.iterdir():
                    target_path = target_dir / item.name
                    if target_path.exists():
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                        else:
                            target_path.unlink()
                    shutil.move(str(item), str(target_path))
            else:
                # Move all items directly
                target_dir = Path(extract_to)
                for item in extracted_items:
                    target_path = target_dir / item.name
                    if target_path.exists():
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                        else:
                            target_path.unlink()
                    shutil.move(str(item), str(target_path))

            # Clean up temp directory
            shutil.rmtree(temp_extract)

        logger.info(f"✅ Extracted {zip_path} to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to extract {zip_path}: {e}")
        return False


def extract_all_nested_zips(directory: Path, max_iterations: int = 5) -> int:
    """Extract all ZIP files in a directory recursively with multiple iterations"""
    total_extracted = 0

    for iteration in range(max_iterations):
        logger.info(f"🔄 ZIP Extraction Iteration {iteration + 1}/{max_iterations}: Scanning {directory}")

        # Find all ZIP files in current iteration
        zip_files = list(directory.rglob("*.zip"))

        if not zip_files:
            logger.info(f"  📂 No ZIP files found in iteration {iteration + 1}")
            break

        logger.info(f"  📦 Found {len(zip_files)} ZIP files to extract")
        iteration_extracted = 0

        for zip_file in zip_files:
            logger.info(f"    📦 Processing: {zip_file.relative_to(directory)}")

            # Determine extraction directory (same directory as ZIP file)
            extract_dir = zip_file.parent / zip_file.stem

            try:
                # Create extraction directory
                extract_dir.mkdir(exist_ok=True)

                # Extract ZIP file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                logger.info(f"      ✅ Extracted {zip_file.name} to {extract_dir.name}/")
                iteration_extracted += 1
                total_extracted += 1

                # List extracted contents
                extracted_items = list(extract_dir.iterdir())
                logger.info(f"      📁 Extracted {len(extracted_items)} items:")

                # Show first 3 items and count ZIP files
                zip_count = 0
                for i, item in enumerate(extracted_items[:3]):
                    if item.is_file():
                        size_kb = item.stat().st_size / 1024
                        if item.suffix.lower() == '.zip':
                            logger.info(f"        📦 {item.name} ({size_kb:.1f} KB) - ZIP file detected!")
                            zip_count += 1
                        else:
                            logger.info(f"        📄 {item.name} ({size_kb:.1f} KB)")
                    else:
                        logger.info(f"        📁 {item.name}/")

                # Count remaining ZIP files
                for item in extracted_items[3:]:
                    if item.suffix.lower() == '.zip':
                        zip_count += 1

                if len(extracted_items) > 3:
                    logger.info(f"        ... and {len(extracted_items) - 3} more items")

                if zip_count > 0:
                    logger.info(f"      🔍 Found {zip_count} additional ZIP files to extract in next iteration")

                # Clean up ZIP file
                zip_file.unlink()
                logger.info(f"      🗑️ Removed {zip_file.name}")

            except Exception as e:
                logger.error(f"      ❌ Failed to extract {zip_file.name}: {e}")

        logger.info(f"  ✅ Iteration {iteration + 1} completed: {iteration_extracted} files extracted")

        # If no files were extracted in this iteration, we're done
        if iteration_extracted == 0:
            break

    logger.info(f"🎉 Total ZIP extraction completed: {total_extracted} ZIP files extracted across {iteration + 1} iterations")
    return total_extracted


def verify_evaluation_data(eval_data_dir: Path) -> bool:
    """Verify that evaluation data has been properly extracted"""
    logger.info("🔍 Verifying evaluation data...")

    # Expected directories
    expected_dirs = [
        "fast_eval",
        "full_eval",
        "fast_eval/blimp_fast",
        "fast_eval/entity_tracking_fast",
        "fast_eval/reading",
        "fast_eval/supplement_fast",
        "fast_eval/wug_adj_nominalization",
        "fast_eval/wug_past_tense",
        "full_eval/blimp_filtered",
        "full_eval/cdi_childes",
        "full_eval/comps",
        "full_eval/entity_tracking",
        "full_eval/glue_filtered",
        "full_eval/reading",
        "full_eval/supplement_filtered",
        "full_eval/vqa_filtered",
        "full_eval/winoground_filtered",
        "full_eval/wug_adj_nominalization",
        "full_eval/wug_past_tense"
    ]

    missing_dirs = []
    found_dirs = []

    for expected_dir in expected_dirs:
        dir_path = eval_data_dir / expected_dir
        if dir_path.exists():
            found_dirs.append(expected_dir)
        else:
            missing_dirs.append(expected_dir)

    logger.info("📊 Verification Summary:")
    logger.info(f"  • Found directories: {len(found_dirs)}/{len(expected_dirs)}")
    logger.info(f"  • Missing directories: {len(missing_dirs)}")

    if missing_dirs:
        logger.warning("⚠️ Missing directories:")
        for missing in missing_dirs:
            logger.warning(f"    - {missing}")

    # Check if we have the minimum required structure
    if (eval_data_dir / "fast_eval").exists() and (eval_data_dir / "full_eval").exists():
        logger.info("🎉 Evaluation data verification successful!")

        # Check for both 2024 and 2025 pipeline compatibility
        has_multimodal = (eval_data_dir / "full_eval" / "vqa_filtered").exists() or (eval_data_dir / "full_eval" / "winoground_filtered").exists()
        has_text_tasks = (eval_data_dir / "full_eval" / "blimp_filtered").exists()

        if has_multimodal:
            logger.info("✅ 2024 pipeline data available (multimodal)")
        if has_text_tasks and has_multimodal:
            logger.info("✅ 2025 pipeline data available (text + multimodal)")

        return True
    else:
        logger.error("❌ Missing fast_eval or full_eval directories")
        return False


def setup_evaluation_symlinks(eval_data_dir: Path) -> bool:
    """Setup symlinks/copies for evaluation pipelines"""
    logger.info("🔗 Setting up symlinks to evaluation data...")

    # Pipeline directories
    pipeline_2024 = Path("../evaluation-pipeline-2024")
    pipeline_2025 = Path("../evaluation-pipeline-2025")

    success_count = 0

    for pipeline_name, pipeline_dir in [("2024", pipeline_2024), ("2025", pipeline_2025)]:
        if pipeline_dir.exists():
            target_eval_data = pipeline_dir / "evaluation_data"

            try:
                # Remove existing evaluation_data if it exists
                if target_eval_data.exists():
                    if target_eval_data.is_symlink():
                        target_eval_data.unlink()
                    else:
                        shutil.rmtree(target_eval_data)

                # Create symlink (or copy on Windows if symlink fails)
                try:
                    target_eval_data.symlink_to(eval_data_dir.resolve(), target_is_directory=True)
                    logger.info(f"✅ Created symlink for {pipeline_name} pipeline: {target_eval_data}")
                except OSError:
                    # Fallback to copying on Windows if symlinks don't work
                    shutil.copytree(eval_data_dir, target_eval_data)
                    logger.info(f"✅ Copied evaluation data for {pipeline_name} pipeline: {target_eval_data}")

                success_count += 1

            except Exception as e:
                logger.warning(f"⚠️ Failed to setup evaluation data for {pipeline_name} pipeline: {e}")
        else:
            logger.warning(f"⚠️ Pipeline directory not found: {pipeline_dir}")

    if success_count == 0:
        logger.error("❌ Failed to set up evaluation data for any pipeline")
        return False
    elif success_count == 2:
        logger.info("🎉 Successfully set up evaluation data for both pipelines")
        return True
    else:
        logger.warning(f"⚠️ Partial success: set up evaluation data for {success_count}/2 pipelines")
        return True


def download_evaluation_data():
    """Download BabyLM evaluation data from OSF"""
    logger.info("🚀 Starting BabyLM Evaluation Data Download")
    logger.info("=" * 60)

    # Target directory for evaluation data
    eval_data_dir = Path("../evaluation_data")
    eval_data_dir.mkdir(exist_ok=True)

    # Check if data already exists and is complete
    if verify_evaluation_data(eval_data_dir):
        logger.info("✅ Evaluation data already exists and appears complete!")

        # Still try to setup symlinks in case they're missing
        setup_success = setup_evaluation_symlinks(eval_data_dir)

        if setup_success:
            logger.info("🎉 Evaluation data setup completed!")
            return True
        else:
            logger.warning("⚠️ Evaluation data exists but symlink setup failed")
            logger.info("💡 You can manually copy evaluation_data to pipeline directories")
            return False

    # Official evaluation data URL
    eval_data_url = "https://files.osf.io/v1/resources/ryjfm/providers/osfstorage/6819f54f5dc6fc2bff0a7bba/?zip="
    zip_filename = eval_data_dir / "babylm_evaluation_data.zip"

    logger.info(f"📁 Target directory: {eval_data_dir.absolute()}")
    logger.info(f"📥 Downloading from: {eval_data_url}")

    # Download the ZIP file
    logger.info(f"\n📥 Downloading evaluation data...")
    if not download_file(eval_data_url, str(zip_filename)):
        logger.error("❌ Download failed!")
        return False

    logger.info("✅ Download completed successfully!")

    # Extract ZIP file with improved handling
    logger.info("📦 Extracting evaluation data...")
    if not extract_zip(str(zip_filename), str(eval_data_dir)):
        logger.error("❌ Extraction failed!")
        return False

    logger.info("✅ Extraction completed successfully!")

    # Clean up ZIP file
    zip_filename.unlink()
    logger.info("🧹 Cleaned up ZIP file")

    # Extract all nested ZIP files (including ewok_fast.zip)
    logger.info("📦 Extracting nested ZIP files...")
    extracted_count = extract_all_nested_zips(eval_data_dir)

    if extracted_count > 0:
        logger.info(f"✅ Extracted {extracted_count} nested ZIP files")
    else:
        logger.info("ℹ️ No nested ZIP files found")

    # Verify the extracted data
    logger.info("🔍 Verifying evaluation data...")
    if not verify_evaluation_data(eval_data_dir):
        logger.error("❌ Download failed!")
        return False

    # Setup symlinks for evaluation pipelines
    logger.info("🔗 Setting up symlinks to evaluation data...")
    setup_success = setup_evaluation_symlinks(eval_data_dir)

    if not setup_success:
        logger.error("❌ Failed to set up evaluation data for any pipeline")
        logger.warning("⚠️ Download successful but symlink setup failed")
        logger.info("💡 You can manually copy evaluation_data to pipeline directories")
        return False

    logger.info("✅ Download and extraction completed successfully!")
    return True


def main():
    """Main function"""
    try:
        success = download_evaluation_data()
        if success:
            logger.info("🎉 All evaluation data setup completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Evaluation data setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("⚠️ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
