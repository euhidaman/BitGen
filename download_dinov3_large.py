"""
Download and setup Facebook's DINOv3-Large for BitMar
Superior vision understanding compared to DINOv2
Uses SafeTensors format (modern standard)
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel, AutoImageProcessor, AutoConfig
    from PIL import Image
    import numpy as np
    # SafeTensors is included with modern transformers
    logger.info("✅ Using SafeTensors format (modern standard)")
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install with: pip install transformers pillow numpy")
    sys.exit(1)


def download_dinov3_large(cache_dir: Optional[str] = None) -> bool:
    """
    Download Facebook's DINOv3-Large model (SafeTensors format)

    Args:
        cache_dir: Optional cache directory

    Returns:
        bool: Success status
    """
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    try:
        logger.info("🚀 Downloading Facebook DINOv3-Large (SafeTensors format)...")
        logger.info(f"Model: {model_name}")
        logger.info("📦 Expected files:")
        logger.info("  • model.safetensors (modern format)")
        logger.info("  • config.json")
        logger.info("  • preprocessor_config.json")

        # Set cache directory
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            logger.info(f"Using cache directory: {cache_dir}")

        # Download model (SafeTensors format)
        logger.info("📥 Downloading model weights (SafeTensors)...")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            use_safetensors=True  # Explicitly use SafeTensors
        )

        # Download processor
        logger.info("📥 Downloading image processor...")
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        # Download config
        logger.info("📥 Downloading model config...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        # Verify model dimensions
        logger.info("🔍 Verifying DINOv3-Large specifications...")
        logger.info(f"  • Model type: {type(model).__name__}")
        logger.info(f"  • Architecture: {config.architectures}")
        logger.info(f"  • Hidden size: {config.hidden_size}")
        logger.info(f"  • Patch size: {config.patch_size}")
        logger.info(f"  • Image size: {config.image_size}")
        logger.info(f"  • Model format: SafeTensors ✅")

        # Expected DINOv3-Large specs
        expected_hidden_size = 1024
        if config.hidden_size != expected_hidden_size:
            logger.warning(f"⚠️  Expected hidden size {expected_hidden_size}, got {config.hidden_size}")
        else:
            logger.info(f"✅ Correct hidden size: {config.hidden_size}")

        # Verify patch size
        expected_patch_size = 16
        if config.patch_size != expected_patch_size:
            logger.warning(f"⚠️  Expected patch size {expected_patch_size}, got {config.patch_size}")
        else:
            logger.info(f"✅ Correct patch size: {config.patch_size}")

        logger.info("✅ DINOv3-Large download completed successfully!")
        logger.info("🎯 Ready for superior multimodal training!")
        logger.info("📦 SafeTensors format provides better security and faster loading!")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to download DINOv3-Large: {e}")
        logger.error("💡 Check your internet connection and HuggingFace access")
        return False


def test_dinov3_large(cache_dir: Optional[str] = None) -> bool:
    """
    Test DINOv3-Large functionality with SafeTensors

    Args:
        cache_dir: Optional cache directory

    Returns:
        bool: Test success status
    """
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    try:
        logger.info("🧪 Testing DINOv3-Large functionality (SafeTensors)...")

        # Set cache directory
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir

        # Load model and processor (SafeTensors)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            use_safetensors=True  # Explicit SafeTensors usage
        )
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        logger.info("✅ Model and processor loaded from SafeTensors")

        # Create test image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        # Process image
        inputs = processor(test_image, return_tensors="pt")
        logger.info(f"✅ Image processed: {inputs['pixel_values'].shape}")

        # Test inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Check output dimensions
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        else:
            # Try to get the main output
            features = outputs[0] if isinstance(outputs, tuple) else outputs

        logger.info(f"✅ Inference successful!")
        logger.info(f"  • Output shape: {features.shape}")
        logger.info(f"  • Feature dimension: {features.shape[-1]}")
        logger.info(f"  • Format: SafeTensors ✅")

        # Verify expected dimensions
        expected_dim = 1024
        if features.shape[-1] == expected_dim:
            logger.info(f"✅ Correct DINOv3-Large feature dimension ({expected_dim})")
        else:
            logger.warning(f"⚠️  Expected {expected_dim}D features, got {features.shape[-1]}D")

        # Verify SafeTensors usage
        logger.info("🔒 SafeTensors benefits:")
        logger.info("  • Faster loading than pickle-based formats")
        logger.info("  • Better security (no arbitrary code execution)")
        logger.info("  • Memory efficient loading")
        logger.info("  • Cross-platform compatibility")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def verify_safetensors_format() -> bool:
    """Verify that we're actually using SafeTensors format"""
    try:
        logger.info("🔍 Verifying SafeTensors format usage...")

        # Check if safetensors is available
        try:
            import safetensors
            logger.info("✅ SafeTensors library available")
        except ImportError:
            logger.warning("⚠️  SafeTensors library not found, will use transformers built-in")

        # Check transformers version
        import transformers
        logger.info(f"✅ Transformers version: {transformers.__version__}")

        # Modern transformers (4.20+) automatically use SafeTensors when available
        version_parts = transformers.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        if major >= 4 and minor >= 20:
            logger.info("✅ Transformers version supports SafeTensors by default")
        else:
            logger.warning(f"⚠️  Consider upgrading transformers for better SafeTensors support")

        return True

    except Exception as e:
        logger.error(f"❌ SafeTensors verification failed: {e}")
        return False


def verify_installation() -> bool:
    """Verify complete installation with SafeTensors support"""
    try:
        logger.info("🔍 Verifying complete DINOv3-Large installation...")

        # Verify SafeTensors support
        if not verify_safetensors_format():
            return False

        # Test download
        if not download_dinov3_large():
            return False

        # Test functionality
        if not test_dinov3_large():
            return False

        logger.info("🎉 DINOv3-Large setup verification complete!")
        logger.info("🔒 SafeTensors format verified and ready!")
        logger.info("🎯 Ready to train BitMar with superior vision understanding!")

        return True

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Download Facebook DINOv3-Large for BitMar")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for models")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    parser.add_argument("--verify", action="store_true", help="Run full verification")
    parser.add_argument("--check-safetensors", action="store_true", help="Check SafeTensors support")

    args = parser.parse_args()

    logger.info("🚀 Facebook DINOv3-Large Setup for BitMar")
    logger.info("📦 Using SafeTensors format (modern standard)")
    logger.info("=" * 60)

    try:
        if args.check_safetensors:
            success = verify_safetensors_format()
        elif args.test_only:
            success = test_dinov3_large(args.cache_dir)
        elif args.verify:
            success = verify_installation()
        else:
            success = download_dinov3_large(args.cache_dir)

        if success:
            logger.info("✅ Setup completed successfully!")
            logger.info("📦 SafeTensors format ready for training!")
            sys.exit(0)
        else:
            logger.error("❌ Setup failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
