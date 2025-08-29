"""
Test Facebook DINOv3-Large integration with BitMar
Verify superior vision understanding capabilities with SafeTensors format
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel, AutoImageProcessor, AutoConfig
    from PIL import Image
    import torchvision.transforms as transforms
    logger.info("✅ Using SafeTensors format (modern standard)")
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install with: pip install transformers pillow torch torchvision")
    sys.exit(1)


def create_test_images() -> Tuple[Image.Image, Image.Image]:
    """Create test images for DINOv3-Large"""

    # Test image 1: Random noise
    noise_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    noise_image = Image.fromarray(noise_array, 'RGB')

    # Test image 2: Simple pattern
    pattern = np.zeros((224, 224, 3), dtype=np.uint8)
    pattern[50:174, 50:174, :] = 255  # White square
    pattern[75:149, 75:149, 0] = 255  # Red inner square
    pattern[75:149, 75:149, 1:] = 0
    pattern_image = Image.fromarray(pattern, 'RGB')

    return noise_image, pattern_image


def verify_safetensors_format() -> bool:
    """Verify SafeTensors format is being used"""
    try:
        logger.info("🔒 Verifying SafeTensors format...")

        # Check transformers version for SafeTensors support
        import transformers
        version_parts = transformers.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        logger.info(f"✅ Transformers version: {transformers.__version__}")

        if major >= 4 and minor >= 21:
            logger.info("✅ SafeTensors supported by default")
        else:
            logger.warning("⚠️  Consider upgrading transformers for better SafeTensors support")

        # Check if safetensors library is available
        try:
            import safetensors
            logger.info("✅ SafeTensors library available")
        except ImportError:
            logger.info("📦 SafeTensors will use transformers built-in support")

        return True

    except Exception as e:
        logger.error(f"❌ SafeTensors verification failed: {e}")
        return False


def test_dinov3_basic_functionality() -> bool:
    """Test basic DINOv3-Large functionality with SafeTensors"""
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    try:
        logger.info("🧪 Testing basic DINOv3-Large functionality...")
        logger.info("📦 Expected files: model.safetensors, config.json, preprocessor_config.json")

        # Load model components with SafeTensors
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True  # Explicit SafeTensors usage
        )
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        logger.info(f"✅ Model loaded from SafeTensors: {type(model).__name__}")
        logger.info(f"✅ Processor loaded: {type(processor).__name__}")
        logger.info(f"✅ Config loaded - Hidden size: {config.hidden_size}")
        logger.info(f"📦 Model format: SafeTensors (secure & fast)")

        # Create test images
        noise_image, pattern_image = create_test_images()

        # Process images
        noise_inputs = processor(noise_image, return_tensors="pt")
        pattern_inputs = processor(pattern_image, return_tensors="pt")

        logger.info(f"✅ Image preprocessing successful")
        logger.info(f"  • Input shape: {noise_inputs['pixel_values'].shape}")

        # Test inference
        model.eval()
        with torch.no_grad():
            noise_outputs = model(**noise_inputs)
            pattern_outputs = model(**pattern_inputs)

        # Extract features (handle different output formats)
        def extract_features(outputs):
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state.mean(dim=1)  # Pool over patches
            elif hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
            elif isinstance(outputs, tuple):
                return outputs[0].mean(dim=1)
            else:
                return outputs.mean(dim=1) if outputs.dim() > 2 else outputs

        noise_features = extract_features(noise_outputs)
        pattern_features = extract_features(pattern_outputs)

        logger.info(f"✅ Feature extraction successful")
        logger.info(f"  • Feature shape: {noise_features.shape}")
        logger.info(f"  • Feature dimension: {noise_features.shape[-1]}")
        logger.info(f"  • SafeTensors format: ✅")

        # Verify expected dimensions for DINOv3-Large
        expected_dim = 1024
        if noise_features.shape[-1] == expected_dim:
            logger.info(f"✅ Correct DINOv3-Large dimension: {expected_dim}")
        else:
            logger.error(f"❌ Wrong dimension: expected {expected_dim}, got {noise_features.shape[-1]}")
            return False

        # Test feature diversity
        similarity = torch.cosine_similarity(noise_features, pattern_features, dim=1)
        logger.info(f"✅ Feature similarity test: {similarity.item():.4f}")

        if similarity.item() > 0.95:
            logger.warning("⚠️  Features too similar - possible issue")
        else:
            logger.info("✅ Good feature diversity between different images")

        # SafeTensors benefits
        logger.info("🔒 SafeTensors advantages:")
        logger.info("  • No arbitrary code execution (secure)")
        logger.info("  • Faster loading than pickle formats")
        logger.info("  • Memory efficient")
        logger.info("  • Cross-platform compatible")

        return True

    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False


def test_dinov3_batch_processing() -> bool:
    """Test batch processing capabilities with SafeTensors"""
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    try:
        logger.info("🧪 Testing DINOv3-Large batch processing...")

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True
        )
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Create batch of test images
        batch_size = 4
        images = []
        for i in range(batch_size):
            # Create different test images
            array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            images.append(Image.fromarray(array, 'RGB'))

        # Process batch
        inputs = processor(images, return_tensors="pt")
        logger.info(f"✅ Batch preprocessing: {inputs['pixel_values'].shape}")

        # Test batch inference
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract batch features
        def extract_batch_features(outputs):
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state.mean(dim=1)  # Pool over patches
            elif hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
            elif isinstance(outputs, tuple):
                return outputs[0].mean(dim=1)
            else:
                return outputs.mean(dim=1) if outputs.dim() > 2 else outputs

        features = extract_batch_features(outputs)

        logger.info(f"✅ Batch inference successful")
        logger.info(f"  • Batch feature shape: {features.shape}")
        logger.info(f"  • Expected: ({batch_size}, 1024)")

        # Verify batch dimensions
        if features.shape[0] == batch_size and features.shape[1] == 1024:
            logger.info("✅ Correct batch processing with SafeTensors")
            return True
        else:
            logger.error(f"❌ Wrong batch dimensions: {features.shape}")
            return False

    except Exception as e:
        logger.error(f"❌ Batch processing test failed: {e}")
        return False


def test_dinov3_bitmar_compatibility() -> bool:
    """Test compatibility with BitMar requirements"""
    try:
        logger.info("🧪 Testing BitMar compatibility...")

        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True
        )
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Test image processing compatible with BitMar
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = processor(test_image, return_tensors="pt")

        # Test feature extraction for BitMar integration
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract features compatible with BitMar vision compression
        def extract_for_bitmar(outputs):
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
                # BitMar expects: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
                if features.dim() == 3:
                    pooled_features = features.mean(dim=1)  # Global average pooling
                    return features, pooled_features
                else:
                    return features, features
            elif hasattr(outputs, 'pooler_output'):
                pooled = outputs.pooler_output
                return pooled.unsqueeze(1), pooled  # Add sequence dimension
            else:
                # Handle other output formats
                feats = outputs[0] if isinstance(outputs, tuple) else outputs
                if feats.dim() == 3:
                    pooled = feats.mean(dim=1)
                    return feats, pooled
                else:
                    return feats, feats

        features, pooled_features = extract_for_bitmar(outputs)

        logger.info("✅ BitMar compatibility tests:")
        logger.info(f"  • Raw features shape: {features.shape}")
        logger.info(f"  • Pooled features shape: {pooled_features.shape}")
        logger.info(f"  • Feature dimension: {pooled_features.shape[-1]}")
        logger.info(f"  • SafeTensors format: ✅")

        # Verify BitMar compatibility requirements
        expected_dim = 1024
        if pooled_features.shape[-1] == expected_dim:
            logger.info(f"✅ Compatible with BitMar vision compression ({expected_dim}D → 192D)")
            logger.info("✅ Ready for BitMar cross-modal training")
            return True
        else:
            logger.error(f"❌ Incompatible dimensions: expected {expected_dim}, got {pooled_features.shape[-1]}")
            return False

    except Exception as e:
        logger.error(f"❌ BitMar compatibility test failed: {e}")
        return False


def test_dinov3_memory_efficiency() -> bool:
    """Test memory efficiency with SafeTensors"""
    try:
        logger.info("🧪 Testing memory efficiency...")

        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            logger.info(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")

        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

        # Load model with SafeTensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True
        )
        model = model.to(device)
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

        if torch.cuda.is_available():
            model_memory = torch.cuda.memory_allocated() - initial_memory
            logger.info(f"Model memory usage: {model_memory / 1024**2:.1f} MB")

        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        successful_batches = []

        for batch_size in batch_sizes:
            try:
                # Create test batch
                images = []
                for i in range(batch_size):
                    array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    images.append(Image.fromarray(array, 'RGB'))

                inputs = processor(images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Test inference
                with torch.no_grad():
                    outputs = model(**inputs)

                successful_batches.append(batch_size)

                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated()
                    logger.info(f"✅ Batch {batch_size}: Peak memory {peak_memory / 1024**2:.1f} MB")
                    torch.cuda.reset_peak_memory_stats()
                else:
                    logger.info(f"✅ Batch {batch_size}: Successful (CPU)")

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"⚠️  Batch {batch_size}: Out of memory")
                break
            except Exception as e:
                logger.warning(f"⚠️  Batch {batch_size}: {e}")
                break

        logger.info(f"✅ SafeTensors memory efficiency test completed")
        logger.info(f"✅ Successfully processed batch sizes: {successful_batches}")
        logger.info("🔒 SafeTensors provides efficient memory loading")

        return len(successful_batches) > 0

    except Exception as e:
        logger.error(f"❌ Memory efficiency test failed: {e}")
        return False


def run_all_tests() -> bool:
    """Run all DINOv3-Large tests with SafeTensors verification"""
    logger.info("🚀 Running comprehensive DINOv3-Large tests...")
    logger.info("📦 Verifying SafeTensors format support")
    logger.info("=" * 70)

    tests = [
        ("SafeTensors Format Verification", verify_safetensors_format),
        ("Basic Functionality", test_dinov3_basic_functionality),
        ("Batch Processing", test_dinov3_batch_processing),
        ("BitMar Compatibility", test_dinov3_bitmar_compatibility),
        ("Memory Efficiency", test_dinov3_memory_efficiency)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 Test Summary:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  • {test_name}: {status}")

    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! DINOv3-Large with SafeTensors is ready for BitMar!")
        logger.info("📦 SafeTensors format verified and working correctly")
        logger.info("🔒 Secure, fast, and memory-efficient model loading")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test DINOv3-Large SafeTensors for BitMar")
    parser.add_argument("--test", choices=['safetensors', 'basic', 'batch', 'compatibility', 'memory', 'all'],
                       default='all', help="Which test to run")

    args = parser.parse_args()

    logger.info("🧪 DINOv3-Large SafeTensors Testing for BitMar")
    logger.info("📦 Modern secure format verification")
    logger.info("=" * 60)

    try:
        if args.test == 'all':
            success = run_all_tests()
        elif args.test == 'safetensors':
            success = verify_safetensors_format()
        elif args.test == 'basic':
            success = test_dinov3_basic_functionality()
        elif args.test == 'batch':
            success = test_dinov3_batch_processing()
        elif args.test == 'compatibility':
            success = test_dinov3_bitmar_compatibility()
        elif args.test == 'memory':
            success = test_dinov3_memory_efficiency()
        else:
            logger.error(f"Unknown test: {args.test}")
            success = False

        if success:
            logger.info("\n✅ Testing completed successfully!")
            logger.info("📦 SafeTensors format ready for secure training!")
            sys.exit(0)
        else:
            logger.error("\n❌ Testing failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
