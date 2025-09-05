"""
Enhanced Dataset processing for BitGen
Handles OpenImages and robot selection datasets with on-the-fly image processing
Supports dynamic robot selection grounding and FIBER cross-modal alignment
Processes images on-the-fly during training for modern VLM architecture
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from typing import Dict, List, Tuple, Optional, Union
import logging
import random
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import pickle
import hashlib

# Setup logger first
logger = logging.getLogger(__name__)

# HuggingFace datasets import
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
    logger.info("✅ HuggingFace datasets available")
except ImportError:
    HF_DATASETS_AVAILABLE = False
    logger.warning(
        "⚠️ HuggingFace datasets not available - install with: pip install datasets")


class EnhancedBitGenDataset(Dataset):
    """Enhanced dataset for BitGen with on-the-fly image processing and robot reasoning grounding"""

    def __init__(
        self,
        dataset_dir: str,
        robot_data_dir: str = "robot_selection_data/data",  # Robot selection dataset
        tokenizer_name: str = "gpt2",
        max_seq_length: int = 256,
        vision_model: str = "facebook/dinov2-base",  # DiNOv2 for on-the-fly processing
        process_images_on_the_fly: bool = True,  # Process images during training
        enable_robot_grounding: bool = True,  # Enable robot selection grounding
        max_robots_per_sample: int = 3,  # Maximum robots to select per task
        robot_selection_probability: float = 0.3,  # Probability of including robot selection
        max_samples_per_split: int = 50000,  # Maximum samples to load
        image_size: Tuple[int, int] = (224, 224),  # Image processing size
    ):
        self.dataset_dir = Path(dataset_dir)
        self.robot_data_dir = Path(robot_data_dir)
        self.max_seq_length = max_seq_length
        self.process_images_on_the_fly = process_images_on_the_fly
        self.enable_robot_grounding = enable_robot_grounding
        self.max_robots_per_sample = max_robots_per_sample
        self.robot_selection_probability = robot_selection_probability
        self.max_samples_per_split = max_samples_per_split
        self.image_size = image_size

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize vision model for on-the-fly processing
        if self.process_images_on_the_fly:
            try:
                logger.info(f"🔄 Loading vision model for on-the-fly processing: {vision_model}")
                self.vision_processor = AutoImageProcessor.from_pretrained(vision_model)
                self.vision_model = AutoModel.from_pretrained(vision_model)
                self.vision_model.eval()
                logger.info("✅ Vision model loaded for on-the-fly processing")
            except Exception as e:
                logger.error(f"❌ Failed to load vision model: {e}")
                raise RuntimeError(f"Vision model required for on-the-fly processing: {e}")
        else:
            self.vision_processor = None
            self.vision_model = None

        # Load robot selection data for grounding
        if self.enable_robot_grounding:
            self._load_robot_selection_data()

        # Load multimodal datasets
        self._load_multimodal_datasets()

        # Create training indices
        self.indices = list(range(len(self.all_captions)))

        # FINAL DATASET SUMMARY
        logger.info("🎯 ENHANCED BITGEN DATASET SUMMARY:")
        logger.info(f"   📝 Total image-caption pairs: {len(self.all_captions):,}")
        logger.info(f"   🤖 Robot selection examples: {len(self.robot_examples):,}" if hasattr(self, 'robot_examples') else "   🤖 No robot data loaded")
        logger.info(f"   📊 Training indices: {len(self.indices):,}")
        logger.info(f"   🔄 On-the-fly image processing: {self.process_images_on_the_fly}")
        logger.info(f"   🤖 Robot grounding enabled: {self.enable_robot_grounding}")
        logger.info(f"   📈 Robot selection probability: {self.robot_selection_probability}")
        logger.info(f"✅ Enhanced dataset ready for BitGen training")

    def _load_robot_selection_data(self):
        """Load robot selection datasets for grounding"""
        try:
            logger.info("🤖 Loading robot selection data for grounding...")
            
            # Load single robot selection
            single_path = self.robot_data_dir / "Single-Robot-Selection" / "single_robot_selection_dataset.json"
            multi_path = self.robot_data_dir / "Multi-Robot-Selection" / "multi_robot_selection_dataset.json"
            
            self.robot_examples = []
            
            if single_path.exists():
                with open(single_path, 'r') as f:
                    single_data = json.load(f)
                    self.robot_examples.extend(single_data)
                logger.info(f"   📄 Loaded {len(single_data):,} single robot examples")
            
            if multi_path.exists():
                with open(multi_path, 'r') as f:
                    multi_data = json.load(f)
                    self.robot_examples.extend(multi_data)
                logger.info(f"   📄 Loaded {len(multi_data):,} multi robot examples")
            
            logger.info(f"✅ Total robot examples: {len(self.robot_examples):,}")
            
            # Extract robot capabilities for dynamic selection
            self._extract_robot_capabilities()
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load robot selection data: {e}")
            self.robot_examples = []
            self.enable_robot_grounding = False

    def _extract_robot_capabilities(self):
        """Extract robot capabilities from examples"""
        if not self.robot_examples:
            return
            
        # Parse robot types and capabilities from instruction
        self.robot_types = ['Drone', 'Underwater Robot', 'Humanoid', 'Robot with Wheels', 'Robot with Legs']
        self.robot_capabilities = {}
        
        # Extract from first example's instruction
        instruction = self.robot_examples[0]['instruction']
        lines = instruction.split('\n')
        
        current_robot = None
        for line in lines:
            line = line.strip()
            if line.endswith(':') and any(robot in line for robot in self.robot_types):
                for robot in self.robot_types:
                    if robot in line:
                        current_robot = robot
                        self.robot_capabilities[robot] = {
                            'capabilities': [],
                            'limitations': [],
                            'environments': []
                        }
                        break
            elif current_robot and line.startswith('capabilities:'):
                caps = line.replace('capabilities:', '').strip().rstrip(',')
                self.robot_capabilities[current_robot]['capabilities'] = [c.strip() for c in caps.split(',') if c.strip()]
            elif current_robot and line.startswith('limitations:'):
                lims = line.replace('limitations:', '').strip().rstrip(',')
                self.robot_capabilities[current_robot]['limitations'] = [l.strip() for l in lims.split(',') if l.strip()]
            elif current_robot and line.startswith('environments:'):
                envs = line.replace('environments:', '').strip().rstrip(',')
                self.robot_capabilities[current_robot]['environments'] = [e.strip() for e in envs.split(',') if e.strip()]

    def _load_multimodal_datasets(self):
        """Load OpenImages and caption data for multimodal training"""
        logger.info("📥 Loading multimodal datasets...")

        self.all_captions = []
        self.all_image_paths = []
        self.all_image_ids = []
        self.dataset_sources = []

        # Try to load from simple download format (aligned_pairs.json)
        simple_pairs_file = self.dataset_dir / "aligned_pairs.json"
        
        if simple_pairs_file.exists():
            try:
                logger.info(f"🚀 Loading aligned pairs from: {simple_pairs_file}")
                
                with open(simple_pairs_file, 'r') as f:
                    aligned_pairs = json.load(f)
                
                # Extract data for on-the-fly processing
                for pair in aligned_pairs:
                    self.all_captions.append(pair['caption'])
                    self.all_image_paths.append(pair['image_path'])
                    self.all_image_ids.append(pair['image_id'])
                    self.dataset_sources.append('openimages')
                
                logger.info(f"✅ Loaded {len(aligned_pairs):,} aligned image-caption pairs")
                
            except Exception as e:
                logger.warning(f"Failed to load aligned pairs: {e}")
        
        # Fallback: Load from unified captions if no aligned pairs
        if not self.all_captions:
            unified_file = self.dataset_dir / "all_captions.json"
            if unified_file.exists():
                logger.info("📄 Loading unified captions file...")
                with open(unified_file, 'r', encoding='utf-8') as f:
                    self.all_captions = json.load(f)

                # Create dummy image data
                self.all_image_paths = [None] * len(self.all_captions)
                self.all_image_ids = list(range(len(self.all_captions)))
                self.dataset_sources = ['unified'] * len(self.all_captions)

                logger.info(f"✅ Loaded {len(self.all_captions):,} captions from unified file")

        if not self.all_captions:
            raise RuntimeError("❌ No multimodal data found! Expected aligned_pairs.json or all_captions.json")

    def _process_image_on_the_fly(self, image_path: str) -> torch.Tensor:
        """Process image on-the-fly during training"""
        try:
            # Load image
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                # Create dummy image if file not found
                image = Image.new('RGB', self.image_size, color=(128, 128, 128))
            
            # Process with vision model
            if self.vision_processor and self.vision_model:
                inputs = self.vision_processor(image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.vision_model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
                return features.squeeze(0)
            else:
                # Return dummy features if no vision model
                return torch.randn(768)  # DiNOv2-base feature size
                
        except Exception as e:
            logger.warning(f"Failed to process image {image_path}: {e}")
            return torch.randn(768)  # Return dummy features on error

    def _select_robots_for_task(self, task_description: str, image_features: Optional[torch.Tensor] = None) -> Dict:
        """Select top-N robots for a given task based on description and optional image features"""
        if not self.enable_robot_grounding or not hasattr(self, 'robot_examples'):
            return {}
        
        # For now, randomly select from robot examples (can be enhanced with semantic matching)
        robot_example = random.choice(self.robot_examples)
        
        # Extract robot selection from example
        if 'output' in robot_example:
            # Single robot selection
            selected_robots = robot_example['output'].split(', ')
        elif 'subtasks' in robot_example:
            # Multi robot selection - extract unique robots
            selected_robots = list(set([subtask['assigned_robot'] for subtask in robot_example['subtasks']]))
        else:
            selected_robots = [random.choice(self.robot_types)]
        
        # Limit to max_robots_per_sample
        selected_robots = selected_robots[:self.max_robots_per_sample]
        
        return {
            'task': task_description,
            'selected_robots': selected_robots,
            'reasoning': f"Based on task requirements, selected: {', '.join(selected_robots)}",
            'robot_capabilities': {robot: self.robot_capabilities.get(robot, {}) for robot in selected_robots}
        }

    def _load_datasets(self):
        """Load HuggingFace LocalizedNarratives and COCO datasets"""
        logger.info("📥 Loading datasets from HuggingFace and local files...")

        self.all_captions = []
        self.all_image_ids = []
        self.all_image_urls = []
        self.dataset_sources = []

        # First, try to load unified captions if available
        unified_file = self.dataset_dir / "all_captions.json"
        if unified_file.exists():
            logger.info("📄 Loading unified captions file...")
            with open(unified_file, 'r', encoding='utf-8') as f:
                self.all_captions = json.load(f)

            # Create dummy image data
            self.all_image_ids = list(range(len(self.all_captions)))
            self.all_image_urls = [None] * len(self.all_captions)
            self.dataset_sources = ['unified'] * len(self.all_captions)

            logger.info(
                f"✅ Loaded {len(self.all_captions):,} captions from unified file")
            return

        # Load HuggingFace LocalizedNarratives if enabled
        if self.use_hf_localized_narratives:
            logger.warning(
                "⚠️ HuggingFace LocalizedNarratives disabled due to deprecated scripts")
            self._load_localized_narratives()
        else:
            # Use local Localized Narratives
            self._load_localized_narratives()
            # Load COCO captions with image URLs (only when not using HuggingFace)
            self._load_coco_captions()

        logger.info(
            f"✅ Total loaded: {len(self.all_captions):,} image-caption pairs")

    def _load_or_create_vision_features(self):
        """Load cached vision features - prioritize simple download format, then legacy caches"""

        # Priority 1: Look for simple download script format (features.npy + aligned_pairs.json)
        simple_features_file = self.dataset_dir / "features.npy"
        simple_pairs_file = self.dataset_dir / "aligned_pairs.json"
        
        if simple_features_file.exists() and simple_pairs_file.exists():
            try:
                logger.info(
                    f"🚀 Loading features from simple download format: {simple_features_file}")
                logger.info(f"   🔍 Checking file sizes:")
                logger.info(f"     Features file size: {simple_features_file.stat().st_size / (1024**2):.1f} MB")
                logger.info(f"     Pairs file size: {simple_pairs_file.stat().st_size / (1024**2):.1f} MB")

                # Load features directly
                self.all_features = np.load(simple_features_file)
                
                # Load aligned pairs to get the mapping
                with open(simple_pairs_file, 'r') as f:
                    aligned_pairs = json.load(f)
                
                logger.info(
                    f"✅ Loaded {len(self.all_features):,} features from simple download")
                logger.info(f"   🧠 Feature shape: {self.all_features.shape}")
                logger.info(f"   📝 Aligned pairs: {len(aligned_pairs):,}")
                logger.info(f"   🔢 Features per sample: {self.all_features.shape[1] if len(self.all_features.shape) > 1 else 'unknown'}")
                
                # Verify alignment between features and pairs
                if len(self.all_features) != len(aligned_pairs):
                    logger.warning(f"⚠️  MISMATCH: {len(self.all_features):,} features but {len(aligned_pairs):,} pairs!")
                else:
                    logger.info(f"   ✅ Perfect alignment: {len(self.all_features):,} features = {len(aligned_pairs):,} pairs")
                
                # Update captions and image data to match aligned pairs
                if len(aligned_pairs) > 0:
                    self.all_captions = [pair['caption'] for pair in aligned_pairs]
                    self.all_image_ids = [pair['image_id'] for pair in aligned_pairs]
                    self.all_image_urls = [pair.get('image_path', '') for pair in aligned_pairs]
                    self.dataset_sources = ['simple_download'] * len(aligned_pairs)
                    
                    logger.info(f"   📝 Updated dataset to use {len(self.all_captions):,} aligned samples")
                
                return

            except Exception as e:
                logger.warning(f"Failed to load simple download format: {e}")

        # Priority 2: Look for download cache (created by download_multimodal_data.py)
        if self.download_cache_file.exists() and self.download_metadata_file.exists():
            try:
                logger.info(
                    f"🚀 Loading pre-cached vision features from download step: {self.download_cache_file}")

                # Load metadata to verify compatibility
                with open(self.download_metadata_file, 'rb') as f:
                    metadata = pickle.load(f)

                # Load cached features
                self.all_features = np.load(self.download_cache_file)
                logger.info(
                    f"✅ Loaded {len(self.all_features):,} pre-cached vision features from download")
                logger.info(
                    f"   Feature type: {metadata.get('feature_type', 'unknown')}")
                logger.info(
                    f"   Feature shape per sample: {metadata.get('feature_shape', 'unknown')}")
                return

            except Exception as e:
                logger.warning(f"Failed to load download cache: {e}")

        # Priority 3: Look for training cache (legacy)
        if (self.cache_vision_features and not self.force_rebuild_cache and
                self.training_cache_file.exists() and self.training_metadata_file.exists()):
            try:
                logger.info(
                    f"🚀 Loading cached vision features from training cache: {self.training_cache_file}")

                # Load metadata to verify compatibility
                with open(self.training_metadata_file, 'rb') as f:
                    metadata = pickle.load(f)

                # Verify cache is compatible
                if (metadata['num_samples'] == len(self.all_captions) and
                    metadata['use_dummy_vision'] == self.use_dummy_vision and
                        metadata['extract_vision_features'] == self.extract_vision_features):

                    # Load cached features
                    self.all_features = np.load(self.training_cache_file)
                    logger.info(
                        f"✅ Loaded {len(self.all_features):,} cached vision features from training cache")
                    return
                else:
                    logger.info("⚠️  Training cache metadata mismatch")
            except Exception as e:
                logger.warning(f"Failed to load training cache: {e}")

        # Priority 4: Error - no cache available and we don't create features during training
        logger.error("❌ No pre-cached vision features found!")
        logger.error("   Expected one of:")
        logger.error(f"   1. Simple format: {simple_features_file} + {simple_pairs_file}")
        logger.error(f"   2. Download cache: {self.download_cache_file}")
        logger.error(f"   3. Training cache: {self.training_cache_file}")
        logger.error("")
        logger.error("   To create features, run:")
        logger.error("   python simple_download_two_shards.py")
        logger.error("   OR")
        logger.error(
            "   python download_multimodal_data.py --dataset both --data_dir ./data --cache_vision_features")
        logger.error(
            "   Or use the manage_vision_cache.py script to create cache")

        # FAIL HARD - no dummy features allowed
        raise RuntimeError("❌ No pre-cached vision features found! Training cannot proceed without real features.")

    def _save_vision_features_cache(self):
        """Save vision features to disk cache"""
        try:
            logger.info(f"💾 Saving vision features cache to {self.cache_file}")

            # Save features
            np.save(self.cache_file, self.all_features)

            # Save metadata
            metadata = {
                'num_samples': len(self.all_captions),
                'use_dummy_vision': self.use_dummy_vision,
                'extract_vision_features': self.extract_vision_features,
                'cache_key': self.cache_key,
                # File modification time
                'created_at': str(Path(__file__).stat().st_mtime)
            }

            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info("✅ Vision features cached successfully")

        except Exception as e:
            logger.warning(f"Failed to save vision features cache: {e}")

        # Pre-compute vision features if requested
        if self.extract_vision_features and not self.use_dummy_vision:
            self._precompute_vision_features()
        elif self.use_dummy_vision:
            self.all_features = np.random.randn(
                len(self.all_captions), 768).astype(np.float32)
            logger.info(
                "🎨 Created dummy vision features (768-dim) for all samples")

    def _load_localized_narratives(self):
        """Load Localized Narratives from JSONL files with image URLs"""
        ln_dir = self.dataset_dir / "localized_narratives"
        if not ln_dir.exists():
            logger.warning("⚠️  Localized Narratives directory not found")
            return

        ln_count = 0
        for dataset_dir in ln_dir.iterdir():
            if dataset_dir.is_dir():
                for jsonl_file in dataset_dir.glob("*.jsonl"):
                    logger.info(f"📄 Loading {jsonl_file}")
                    try:
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f):
                                if line.strip():
                                    try:
                                        data = json.loads(line)
                                        # Extract caption/narrative
                                        caption = data.get('caption') or data.get(
                                            'narrative', '')
                                        if caption and len(caption.strip()) > 0:
                                            self.all_captions.append(
                                                caption.strip())
                                            self.all_image_ids.append(
                                                data.get('image_id', f"ln_{line_num}"))
                                            self.all_image_urls.append(
                                                data.get('image_url'))
                                            self.dataset_sources.append(
                                                f"localized_narratives_{dataset_dir.name}")
                                            ln_count += 1
                                    except json.JSONDecodeError as e:
                                        logger.debug(
                                            f"Skipping malformed JSON on line {line_num}: {e}")
                                        continue
                    except Exception as e:
                        logger.warning(f"Failed to load {jsonl_file}: {e}")

        logger.info(f"✅ Localized Narratives: {ln_count:,} samples loaded")

    def _load_hf_localized_narratives(self):
        """Load LocalizedNarratives from HuggingFace datasets"""
        if not HF_DATASETS_AVAILABLE:
            logger.warning(
                "⚠️ HuggingFace datasets not available, falling back to local loading")
            self._load_localized_narratives()
            return

        try:
            logger.info(
                f"🤗 Loading LocalizedNarratives from HuggingFace (config: {self.hf_dataset_config})...")

            # Load the dataset from HuggingFace
            dataset = load_dataset(
                "HuggingFaceM4/LocalizedNarratives", self.hf_dataset_config)

            ln_count = 0

            # Process train split
            if 'train' in dataset:
                train_data = dataset['train']
                max_train_samples = min(
                    len(train_data), self.max_samples_per_split)

                logger.info(
                    f"📖 Processing {max_train_samples:,} train samples...")

                for i in tqdm(range(max_train_samples), desc="Loading train samples"):
                    try:
                        sample = train_data[i]

                        # Extract caption
                        caption = sample.get('caption', '')
                        if caption and len(caption.strip()) > 0:
                            self.all_captions.append(caption.strip())

                            # Extract image information
                            image_id = sample.get('image_id', f"hf_train_{i}")
                            self.all_image_ids.append(image_id)

                            # Handle image - it's a PIL Image object from HuggingFace
                            image = sample.get('image')
                            if image is not None:
                                # Store the PIL image directly for later processing
                                self.all_image_urls.append(image)
                            else:
                                self.all_image_urls.append(None)

                            self.dataset_sources.append(
                                f"hf_localized_narratives_{self.hf_dataset_config}_train")
                            ln_count += 1

                    except Exception as e:
                        logger.debug(f"Error processing train sample {i}: {e}")
                        continue

            # Process validation split if available
            if 'validation' in dataset:
                val_data = dataset['validation']
                max_val_samples = min(
                    len(val_data), self.max_samples_per_split // 2)

                logger.info(
                    f"📖 Processing {max_val_samples:,} validation samples...")

                for i in tqdm(range(max_val_samples), desc="Loading validation samples"):
                    try:
                        sample = val_data[i]

                        # Extract caption
                        caption = sample.get('caption', '')
                        if caption and len(caption.strip()) > 0:
                            self.all_captions.append(caption.strip())

                            # Extract image information
                            image_id = sample.get('image_id', f"hf_val_{i}")
                            self.all_image_ids.append(image_id)

                            # Handle image - it's a PIL Image object from HuggingFace
                            image = sample.get('image')
                            if image is not None:
                                # Store the PIL image directly for later processing
                                self.all_image_urls.append(image)
                            else:
                                self.all_image_urls.append(None)

                            self.dataset_sources.append(
                                f"hf_localized_narratives_{self.hf_dataset_config}_val")
                            ln_count += 1

                    except Exception as e:
                        logger.debug(
                            f"Error processing validation sample {i}: {e}")
                        continue

            logger.info(
                f"✅ HuggingFace LocalizedNarratives: {ln_count:,} samples loaded")

        except Exception as e:
            logger.error(
                f"❌ Failed to load HuggingFace LocalizedNarratives: {e}")
            logger.info("🔄 Falling back to local loading...")
            self._load_localized_narratives()

    def _load_coco_captions(self):
        """Load COCO captions from annotation files with image URLs"""
        coco_dir = self.dataset_dir / "coco" / "annotations"
        if not coco_dir.exists():
            logger.warning("⚠️  COCO annotations directory not found")
            return

        coco_count = 0

        # Load image info first to get URLs
        image_info = {}
        for split in ['train2017', 'val2017']:
            instances_file = coco_dir / f"instances_{split}.json"
            if instances_file.exists():
                try:
                    with open(instances_file, 'r') as f:
                        data = json.load(f)
                    for img in data['images']:
                        # COCO images follow pattern: http://images.cocodataset.org/{split}/{id:012d}.jpg
                        image_url = f"http://images.cocodataset.org/zips/{split}/{img['file_name']}"
                        image_info[img['id']] = image_url
                except Exception as e:
                    logger.warning(
                        f"Failed to load image info from {instances_file}: {e}")

        # Load captions with image URLs
        for split in ['train2017', 'val2017']:
            captions_file = coco_dir / f"captions_{split}.json"
            if captions_file.exists():
                logger.info(f"📄 Loading {captions_file}")
                try:
                    with open(captions_file, 'r') as f:
                        data = json.load(f)

                    for annotation in data['annotations']:
                        caption = annotation['caption'].strip()
                        if len(caption) > 0:
                            self.all_captions.append(caption)
                            image_id = annotation['image_id']
                            self.all_image_ids.append(image_id)
                            self.all_image_urls.append(
                                image_info.get(image_id))
                            self.dataset_sources.append(f"coco_{split}")
                            coco_count += 1

                except Exception as e:
                    logger.warning(f"Failed to load {captions_file}: {e}")

        logger.info(f"✅ COCO: {coco_count:,} samples loaded")

    def _precompute_vision_features(self):
        """Pre-compute vision features for all images (supports both URLs and PIL images)"""
        logger.info("🔄 Pre-computing vision features from images...")

        self.all_features = []

        for idx, image_source in enumerate(tqdm(self.all_image_urls, desc="Processing images")):
            try:
                if image_source:
                    # Handle PIL Image objects from HuggingFace
                    if isinstance(image_source, Image.Image):
                        image = image_source.convert('RGB')
                    elif isinstance(image_source, str):
                        # Download and process image from URL
                        response = requests.get(image_source, timeout=10)
                        image = Image.open(
                            BytesIO(response.content)).convert('RGB')
                    else:
                        # Unknown image source type
                        logger.debug(
                            f"Unknown image source type for idx {idx}: {type(image_source)}")
                        self.all_features.append(
                            np.random.randn(768).astype(np.float32))
                        continue

                    # Process with vision model
                    inputs = self.vision_processor(image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.vision_model(**inputs)
                        # Get pooled features (768-dim)
                        features = outputs.pooler_output.squeeze().numpy()
                        self.all_features.append(features)
                else:
                    # No image source, use dummy features
                    self.all_features.append(
                        np.random.randn(768).astype(np.float32))

            except Exception as e:
                logger.debug(f"Failed to process image {idx}: {e}")
                # Use dummy features for failed images
                self.all_features.append(
                    np.random.randn(768).astype(np.float32))

        self.all_features = np.array(self.all_features)
        logger.info(f"✅ Pre-computed {len(self.all_features)} vision features")

    def _get_vision_features(self, idx: int) -> np.ndarray:
        """Get vision features for a sample (always use cached/pre-computed now)"""
        # Always use pre-computed/cached features now
        if hasattr(self, 'all_features'):
            return self.all_features[idx]

        # Fallback to dummy if no features available (shouldn't happen)
        logger.warning(
            f"No pre-computed features available for sample {idx}, using dummy")
        return np.random.randn(768).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample with on-the-fly vision processing and optional robot grounding"""
        real_idx = self.indices[idx]
        caption = self.all_captions[real_idx]
        
        # Determine if this sample should include robot selection
        include_robot_selection = (
            self.enable_robot_grounding and 
            random.random() < self.robot_selection_probability
        )

        # Get vision features via on-the-fly processing
        vision_features = self._get_vision_features_on_the_fly(real_idx)

        # Tokenize caption
        encoded = self.tokenizer(
            caption,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Convert attention mask to boolean for PyTorch compatibility
        attention_mask = attention_mask.bool()

        # Create labels for text generation
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Prepare return data
        sample_data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'vision_features': vision_features,  # Fresh features from on-the-fly processing
            'has_vision': True,
            'vision_index': real_idx,
            'sample_type': 'multimodal_caption',
            'sample_index': idx,
            'text': caption,
            'image_path': self.all_image_paths[real_idx] if hasattr(self, 'all_image_paths') else None,
            'dataset_source': self.dataset_sources[real_idx] if hasattr(self, 'dataset_sources') else 'unknown'
        }

        # Add robot grounding if enabled
        if include_robot_selection:
            robot_selection = self._select_robots_for_task(caption, vision_features)
            sample_data.update({
                'robot_selection': robot_selection,
                'has_robot_grounding': True,
                'selected_robots': robot_selection.get('selected_robots', []),
                'robot_reasoning': robot_selection.get('reasoning', ''),
                'robot_capabilities': robot_selection.get('robot_capabilities', {})
            })
        else:
            sample_data.update({
                'has_robot_grounding': False,
                'selected_robots': [],
                'robot_reasoning': '',
                'robot_capabilities': {}
            })

        return sample_data

    def _get_vision_features_on_the_fly(self, idx: int) -> torch.Tensor:
        """Get vision features via on-the-fly image processing"""
        if not self.process_images_on_the_fly:
            # Fallback to dummy features if not processing on-the-fly
            return torch.randn(768)
        
        # Get image path
        image_path = None
        if hasattr(self, 'all_image_paths') and idx < len(self.all_image_paths):
            image_path = self.all_image_paths[idx]
        
        if image_path and os.path.exists(image_path):
            # Process real image
            vision_features = self._process_image_on_the_fly(image_path)
        else:
            # Generate dummy features for missing images
            vision_features = torch.randn(768)
        
        # Ensure proper tensor format
        if not isinstance(vision_features, torch.Tensor):
            vision_features = torch.tensor(vision_features, dtype=torch.float32)
        
        # Ensure exactly 768 dimensions
        if vision_features.dim() > 1:
            vision_features = vision_features.flatten()
        
        if vision_features.size(0) != 768:
            if vision_features.size(0) > 768:
                vision_features = vision_features[:768]
            else:
                pad_size = 768 - vision_features.size(0)
                vision_features = torch.cat([vision_features, torch.zeros(pad_size)])
        
        return vision_features

    def __len__(self) -> int:
        """Return dataset length"""
        return len(self.indices)

    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a specific sample"""
        real_idx = self.indices[idx]
        return {
            'index': idx,
            'real_index': real_idx,
            'caption': self.all_captions[real_idx],
            'image_id': self.all_image_ids[real_idx] if hasattr(self, 'all_image_ids') else real_idx,
            'image_path': self.all_image_paths[real_idx] if hasattr(self, 'all_image_paths') else None,
            'source': self.dataset_sources[real_idx] if hasattr(self, 'dataset_sources') else 'unknown',
            'has_robot_grounding': self.enable_robot_grounding
        }


def create_data_module(config: Dict) -> 'EnhancedBitGenDataModule':
    """Create enhanced data module for BitGen"""
    return EnhancedBitGenDataModule(config)


class EnhancedBitGenDataModule:
    """Enhanced data module for BitGen with on-the-fly image processing and robot grounding"""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset = None
        self.train_loader = None

    def setup(self, force_rebuild: bool = False):
        """Setup the enhanced BitGen dataset"""
        logger.info("🔧 Setting up Enhanced BitGen data module...")

        # Configuration for enhanced features
        robot_data_dir = self.config.get('robot_data_dir', 'robot_selection_data/data')
        process_images_on_the_fly = self.config.get('process_images_on_the_fly', True)
        enable_robot_grounding = self.config.get('enable_robot_grounding', True)
        max_robots_per_sample = self.config.get('max_robots_per_sample', 3)
        robot_selection_probability = self.config.get('robot_selection_probability', 0.3)

        logger.info(f"   🔄 On-the-fly image processing: {process_images_on_the_fly}")
        logger.info(f"   🤖 Robot grounding: {enable_robot_grounding}")
        logger.info(f"   📊 Robot selection probability: {robot_selection_probability}")

        self.dataset = EnhancedBitGenDataset(
            dataset_dir=self.config['dataset_dir'],
            robot_data_dir=robot_data_dir,
            tokenizer_name=self.config['text_encoder_name'],
            max_seq_length=self.config['max_seq_length'],
            vision_model=self.config.get('vision_encoder_name', 'facebook/dinov2-base'),
            process_images_on_the_fly=process_images_on_the_fly,
            enable_robot_grounding=enable_robot_grounding,
            max_robots_per_sample=max_robots_per_sample,
            robot_selection_probability=robot_selection_probability,
            max_samples_per_split=self.config.get('max_samples_per_split', 50000),
            image_size=self.config.get('image_size', (224, 224))
        )

        logger.info(f"📊 Enhanced dataset ready with {len(self.dataset):,} samples")

    def get_train_dataloader(self, batch_size: int = 16, num_workers: int = 4, shuffle: bool = True) -> DataLoader:
        """Create training data loader"""
        if self.dataset is None:
            raise RuntimeError("Dataset not setup. Call setup() first.")

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        if self.dataset is None:
            return {}

        stats = {
            'total_samples': len(self.dataset),
            'has_robot_grounding': getattr(self.dataset, 'enable_robot_grounding', False),
            'robot_examples': len(getattr(self.dataset, 'robot_examples', [])),
            'vision_processing': getattr(self.dataset, 'process_images_on_the_fly', False),
            'robot_types': getattr(self.dataset, 'robot_types', []),
        }

        return stats


def create_data_module(config: Dict) -> EnhancedBitGenDataModule:
    """Create enhanced data module from config"""
    return EnhancedBitGenDataModule(config)

    def clear_vision_cache(self):
        """Clear cached vision features"""
        if hasattr(self, 'dataset') and self.dataset:
            cache_dir = self.dataset.cache_dir
            if cache_dir.exists():
                logger.info(
                    f"🗑️  Clearing vision features cache from {cache_dir}")
                for cache_file in cache_dir.glob("*.npy"):
                    cache_file.unlink()
                    logger.info(f"   Deleted: {cache_file}")
                for metadata_file in cache_dir.glob("*.pkl"):
                    metadata_file.unlink()
                    logger.info(f"   Deleted: {metadata_file}")
                logger.info("✅ Vision features cache cleared")
            else:
                logger.info("ℹ️  No vision features cache to clear")
        else:
            logger.warning("⚠️  No dataset loaded, cannot clear cache")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        if self.train_loader is None:
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=self.config.get('pin_memory', True),
                persistent_workers=self.config.get('persistent_workers', True),
                drop_last=True
            )
            
            # CRITICAL: Log exact training details
            total_samples = len(self.dataset)
            batch_size = self.config['batch_size']
            batches_per_epoch = total_samples // batch_size  # drop_last=True means we lose the remainder
            
            logger.info("🚀 TRAINING DATALOADER CREATED:")
            logger.info(f"   📊 Total samples in dataset: {total_samples:,}")
            logger.info(f"   📦 Batch size: {batch_size}")
            logger.info(f"   🔄 Batches per epoch: {batches_per_epoch:,}")
            logger.info(f"   📉 Samples dropped (remainder): {total_samples % batch_size}")
            logger.info(f"   🎯 Effective samples per epoch: {batches_per_epoch * batch_size:,}")
            
        return self.train_loader

    def val_dataloader(self) -> List[DataLoader]:
        """Return empty list since we use entire dataset for training"""
        return []

    def get_dataset_info(self) -> Dict[str, any]:
        """Get dataset information"""
        if not self.dataset:
            return {}

        return {
            'total_samples': len(self.dataset),
            'tokenizer': self.config['text_encoder_name'],
            'max_seq_length': self.config['max_seq_length'],
            'datasets': 'Localized Narratives + COCO',
            'vision_features': '768-dimensional (model compatible)',
            'extract_real_vision': not self.dataset.use_dummy_vision
        }


def test_dataset(config: Dict):
    """Test dataset loading and processing"""
    logger.info("🧪 Testing Localized Narratives + COCO dataset...")

    # Create data module
    data_module = create_data_module(config)
    data_module.setup()

    # Test sample
    sample = data_module.dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
    logger.info(f"Vision features shape: {sample['vision_features'].shape}")
    logger.info(f"Vision features type: {sample['vision_features'].dtype}")
    logger.info(f"Caption: {sample['text'][:100]}...")
    logger.info(f"Source: {sample['source']}")


def test_enhanced_dataset(config: Dict):
    """Test enhanced BitGen dataset loading and processing"""
    logger.info("🧪 Testing Enhanced BitGen dataset...")

    # Create data module
    data_module = create_data_module(config)
    data_module.setup()

    # Test sample
    sample = data_module.dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
    logger.info(f"Vision features shape: {sample['vision_features'].shape}")
    logger.info(f"Vision features type: {sample['vision_features'].dtype}")
    logger.info(f"Caption: {sample['text'][:100]}...")
    logger.info(f"Has robot grounding: {sample['has_robot_grounding']}")
    if sample['has_robot_grounding']:
        logger.info(f"Selected robots: {sample['selected_robots']}")
        logger.info(f"Robot reasoning: {sample['robot_reasoning'][:100]}...")

    # Test dataset stats
    stats = data_module.get_dataset_stats()
    logger.info(f"Dataset stats: {stats}")

    # Verify vision features are exactly 768-dim
    assert sample['vision_features'].shape == torch.Size([768]), f"Expected [768], got {sample['vision_features'].shape}"
    logger.info("✅ Vision features are correctly formatted for model")

    logger.info("✅ Enhanced dataset test completed successfully!")
    return data_module


if __name__ == "__main__":
    # Test configuration
    test_config = {
        'dataset_dir': "./data",
        'robot_data_dir': "../robot_selection_data/data",
        'text_encoder_name': "gpt2",
        'vision_encoder_name': 'facebook/dinov2-base',
        'max_seq_length': 256,
        'batch_size': 16,
        'num_workers': 4,
        'process_images_on_the_fly': True,
        'enable_robot_grounding': True,
        'robot_selection_probability': 0.3,
        'max_robots_per_sample': 3,
        'image_size': (224, 224)
    }

    # Test enhanced dataset
    test_enhanced_dataset(test_config)
