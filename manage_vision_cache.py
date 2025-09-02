#!/usr/bin/env python3
"""
Vision Features Cache Management Tool
Manage cached vision embeddings for BitMar training
"""

import argparse
import json
import sys
from pathlib import Path
import logging
from typing import Dict, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.dataset import LocalizedNarrativesCOCODataModule
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load training configuration"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def show_cache_info(config: Dict):
    """Show information about vision features cache"""
    data_config = config['data']
    dataset_dir = Path(data_config['dataset_dir'])
    cache_dir = dataset_dir / "vision_features_cache"
    
    logger.info("📊 Vision Features Cache Information:")
    logger.info(f"  • Dataset directory: {dataset_dir}")
    logger.info(f"  • Cache directory: {cache_dir}")
    
    if not cache_dir.exists():
        logger.info("  • Status: No cache directory found")
        return
        
    # List cache files
    cache_files = list(cache_dir.glob("features_*.npy"))
    metadata_files = list(cache_dir.glob("metadata_*.pkl"))
    
    logger.info(f"  • Cache files found: {len(cache_files)}")
    logger.info(f"  • Metadata files found: {len(metadata_files)}")
    
    if cache_files:
        logger.info("  • Cache files:")
        for cache_file in cache_files:
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            logger.info(f"    - {cache_file.name} ({size_mb:.1f} MB)")
            
    if metadata_files:
        logger.info("  • Metadata files:")
        for metadata_file in metadata_files:
            logger.info(f"    - {metadata_file.name}")
            
            # Try to load and show metadata
            try:
                import pickle
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"      Samples: {metadata.get('num_samples', 'unknown')}")
                logger.info(f"      Dummy vision: {metadata.get('use_dummy_vision', 'unknown')}")
                logger.info(f"      Real features: {metadata.get('extract_vision_features', 'unknown')}")
            except Exception as e:
                logger.warning(f"      Failed to read metadata: {e}")


def clear_cache(config: Dict, confirm: bool = False):
    """Clear vision features cache"""
    data_config = config['data']
    
    if not confirm:
        response = input("⚠️  This will delete all cached vision features. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Cache clearing cancelled.")
            return
    
    logger.info("🗑️  Clearing vision features cache...")
    
    # Create data module and clear cache
    data_module = LocalizedNarrativesCOCODataModule(data_config)
    data_module.clear_vision_cache()
    
    logger.info("✅ Vision features cache cleared successfully")


def rebuild_cache(config: Dict, use_real_vision: bool = False):
    """Rebuild vision features cache"""
    data_config = config['data'].copy()
    
    if use_real_vision:
        logger.info("🎨 Rebuilding cache with REAL vision features (slower but higher quality)")
        data_config['extract_vision_features'] = True
        data_config['use_dummy_vision'] = False
    else:
        logger.info("⚡ Rebuilding cache with DUMMY vision features (faster)")
        data_config['extract_vision_features'] = False
        data_config['use_dummy_vision'] = True
    
    data_config['cache_vision_features'] = True
    data_config['force_rebuild_cache'] = True
    
    # Create data module and setup (this will rebuild cache)
    data_module = LocalizedNarrativesCOCODataModule(data_config)
    data_module.setup(rebuild_cache=True)
    
    logger.info("✅ Vision features cache rebuilt successfully")


def main():
    parser = argparse.ArgumentParser(description="Manage vision features cache")
    parser.add_argument("--config", type=str, default="configs/bitmar_with_memory.yaml",
                       help="Path to training configuration file")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show cache information')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear vision features cache')
    clear_parser.add_argument('--yes', action='store_true', help='Skip confirmation prompt')
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild vision features cache')
    rebuild_parser.add_argument('--real-vision', action='store_true', 
                               help='Use real vision features (slower but better quality)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute command
    if args.command == 'info':
        show_cache_info(config)
    elif args.command == 'clear':
        clear_cache(config, confirm=args.yes)
    elif args.command == 'rebuild':
        rebuild_cache(config, use_real_vision=args.real_vision)


if __name__ == "__main__":
    main()
