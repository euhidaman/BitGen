"""
Data Loading Components for BitGen
Handles COCO dataset and robot selection data for multi-modal training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torchvision.transforms as transforms
from collections import defaultdict
import random

class BitGenTokenizer:
    """Simple tokenizer optimized for embedded systems"""

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size

        # Basic vocabulary (reduced for embedded)
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<reasoning>': 4,
            '</reasoning>': 5,
            '<answer>': 6,
            '</answer>': 7,
            '<robot>': 8,
            '</robot>': 9
        }

        # Build vocabulary from common words
        self.vocab = self.build_vocabulary()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def build_vocabulary(self) -> List[str]:
        """Build compact vocabulary for embedded deployment"""
        vocab = list(self.special_tokens.keys())

        # Common English words (frequency-based)
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'can', 'may', 'must', 'shall',
            'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'how', 'why',
            'what', 'who', 'which', 'one', 'two', 'three', 'four', 'five',
            'robot', 'task', 'image', 'picture', 'move', 'pick', 'place', 'go', 'stop',
            'left', 'right', 'up', 'down', 'forward', 'backward', 'turn', 'rotate'
        ]

        vocab.extend(common_words)

        # Add single characters and basic punctuation
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789.,!?;:-()[]{}"\''
        vocab.extend(list(chars))

        # Pad vocabulary to desired size with numbered tokens
        while len(vocab) < self.vocab_size:
            vocab.append(f'<unk_{len(vocab)}>')

        return vocab[:self.vocab_size]

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization"""
        text = text.lower().strip()
        tokens = text.split()

        # Further split on punctuation
        result = []
        for token in tokens:
            if any(punct in token for punct in '.,!?;:'):
                # Simple split on punctuation
                result.extend([c for c in token if c.isalnum() or c in ' '])
            else:
                result.append(token)

        return [t for t in result if t.strip()]

    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """Encode text to token IDs with proper bounds checking"""
        tokens = self.tokenize(text)

        # Add special tokens
        token_ids = [self.special_tokens['<bos>']]

        for token in tokens[:max_length-2]:  # Reserve space for BOS and EOS
            token_id = self.token_to_id.get(token, self.special_tokens['<unk>'])
            # Ensure token ID is within vocabulary bounds
            token_id = min(token_id, self.vocab_size - 1)
            token_ids.append(token_id)

        token_ids.append(self.special_tokens['<eos>'])

        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(self.special_tokens['<pad>'])

        # Final safety check - clamp all token IDs to vocabulary bounds
        token_ids = [min(max(tid, 0), self.vocab_size - 1) for tid in token_ids]

        return token_ids[:max_length]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id == self.special_tokens['<pad>']:
                break
            if token_id == self.special_tokens['<eos>']:
                break
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])

        return ' '.join(tokens)

class COCODataset(Dataset):
    """COCO dataset optimized for BitGen training - REAL IMAGES ONLY"""

    def __init__(self,
                 data_file: str,
                 image_size: int = 224,
                 max_seq_len: int = 128,
                 vocab_size: int = 8192,
                 max_samples: Optional[int] = None):

        print(f"📂 Loading dataset from: {data_file}")
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        original_size = len(self.data)
        print(f"📊 Original dataset size: {original_size:,} samples")

        # CRITICAL: Limit dataset size to prevent extremely long training
        if max_samples is not None and max_samples < len(self.data):
            print(f"⚠️ WARNING: Limiting dataset from {len(self.data):,} to {max_samples:,} samples")
            print(f"   This will reduce training time from ~{len(self.data)//6000:.1f} hours to ~{max_samples//6000:.1f} hours per epoch")
            random.seed(42)  # Reproducible sampling
            self.data = random.sample(self.data, max_samples)

        # Warn if dataset is extremely large
        if len(self.data) > 50000:
            print(f"⚠️⚠️⚠️ WARNING: Very large dataset detected: {len(self.data):,} samples")
            print(f"   This will take approximately {len(self.data)//6000:.1f} hours per epoch!")
            print(f"   Consider using --max_samples parameter to limit dataset size")
            print(f"   Recommended: --max_samples 10000 for faster training")

        print(f"✅ Final dataset size: {len(self.data):,} samples")
        print(f"   Expected iterations per epoch (batch_size=64): {len(self.data)//64:,}")

        self.tokenizer = BitGenTokenizer(vocab_size)
        self.max_seq_len = max_seq_len

        # Image transforms optimized for embedded - MANDATORY for real images
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load and transform image - MUST succeed or crash
        try:
            image_path = item['image_path']
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            # Log successful image loading (debug level)
            # print(f"✅ Successfully loaded image: {image_path}")

        except FileNotFoundError:
            raise FileNotFoundError(f"❌ CRITICAL: Image file not found: {item.get('image_path', 'unknown path')}")
        except Exception as e:
            raise RuntimeError(f"❌ CRITICAL: Failed to load/process image {item.get('image_path', 'unknown path')}. Real images are mandatory. Error: {e}")

        # Process caption
        caption = item['caption']
        input_ids = self.tokenizer.encode(caption, self.max_seq_len)

        # Validate token IDs are within bounds - CRITICAL for CUDA safety
        max_valid_id = self.tokenizer.vocab_size - 1
        input_ids = [min(max(tid, 0), max_valid_id) for tid in input_ids]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token_id != self.tokenizer.special_tokens['<pad>'] else 0 for token_id in input_ids]

        # Create labels (shifted input_ids for language modeling)
        labels = input_ids[1:] + [self.tokenizer.special_tokens['<pad>']]

        # Validate labels are also within bounds
        labels = [min(max(tid, 0), max_valid_id) for tid in labels]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'images': image,
            'caption': caption,
            'image_id': item.get('image_id', idx)
        }

class RobotSelectionDataset(Dataset):
    """Dataset for multi-label robot selection training with top-K robot outputs"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        # CRITICAL: Validate required dataset file exists
        required_file = self.data_dir / "Multi-Robot-Selection" / "multi_robot_selection_dataset.json"
        if not required_file.exists():
            # Try alternate path
            required_file = self.data_dir / "multi_robot_selection_dataset.json"
            if not required_file.exists():
                raise FileNotFoundError(
                    f"❌ CRITICAL ERROR: Required dataset file not found!\n"
                    f"Expected path: {self.data_dir / 'Multi-Robot-Selection' / 'multi_robot_selection_dataset.json'}\n"
                    f"Or: {self.data_dir / 'multi_robot_selection_dataset.json'}\n"
                    f"Robot selection training CANNOT proceed without multi_robot_selection_dataset.json\n"
                    f"Please ensure the dataset is downloaded and placed in the correct directory."
                )
        
        self.dataset_file = required_file
        self.data = self.load_robot_data()
        self.tokenizer = BitGenTokenizer()

        # Robot type mapping from dataset
        self.robot_types = self.extract_robot_types()
        self.robot_to_id = {robot: idx for idx, robot in enumerate(self.robot_types)}
        
        print(f"✅ Loaded robot selection dataset: {len(self.data)} samples")
        print(f"✅ Robot types found: {self.robot_types}")

    def load_robot_data(self) -> List[Dict]:
        """Load robot selection data from multi_robot_selection_dataset.json"""
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(
                f"❌ Dataset format error: Expected list, got {type(data)}\n"
                f"File: {self.dataset_file}"
            )
        
        if len(data) == 0:
            raise ValueError(
                f"❌ Dataset is empty!\n"
                f"File: {self.dataset_file}"
            )
        
        return data

    def extract_robot_types(self) -> List[str]:
        """Extract unique robot types from multi-robot output strings"""
        robot_types = set()

        for item in self.data:
            # Parse "original_single_robot_output": "Drone, Robot with Legs"
            robot_output = item.get('original_single_robot_output', '')
            if robot_output:
                # Split by comma and extract individual robot names
                robots = [r.strip() for r in robot_output.split(',')]
                robot_types.update(robots)
        
        if not robot_types:
            raise ValueError(
                f"❌ No robot types found in dataset!\n"
                f"Check 'original_single_robot_output' field in {self.dataset_file}"
            )

        return sorted(list(robot_types))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract task description
        task_desc = item.get('task_description', item.get('description', item.get('task', '')))

        # Parse multi-robot output: "Drone, Robot with Legs, Humanoid"
        robot_output = item.get('original_single_robot_output', '')
        selected_robots = [r.strip() for r in robot_output.split(',')] if robot_output else []
        
        # Create multi-hot label vector for multi-label classification
        robot_labels = torch.zeros(len(self.robot_types), dtype=torch.float32)
        valid_robots = []
        for robot_name in selected_robots:
            if robot_name in self.robot_to_id:
                robot_labels[self.robot_to_id[robot_name]] = 1.0
                valid_robots.append(robot_name)
        
        # Tokenize task description
        input_ids = self.tokenizer.encode(task_desc, max_length=128)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token_id != self.tokenizer.special_tokens['<pad>'] else 0 for token_id in input_ids]

        labels = input_ids[1:] + [self.tokenizer.special_tokens['<pad>']]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'robot_labels': robot_labels,  # Multi-hot vector [num_robots]
            'selected_robots': valid_robots,  # List of robot names for logging
            'task_description': task_desc,
            'num_robots_selected': len(valid_robots)
        }

class BitGenDataLoader:
    """Optimized data loader for embedded training"""

    def __init__(self,
                 coco_dataset: COCODataset,
                 robot_dataset: Optional[RobotSelectionDataset] = None,
                 batch_size: int = 4,
                 shuffle: bool = True,
                 num_workers: int = 0):  # Reduced for embedded

        self.coco_dataset = coco_dataset
        self.robot_dataset = robot_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Create data loaders
        self.coco_loader = DataLoader(
            coco_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # Disabled for embedded
            drop_last=True
        )

        self.robot_loader = None
        if robot_dataset:
            self.robot_loader = DataLoader(
                robot_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=True
            )

    def __iter__(self):
        """Iterator that combines COCO and robot data"""
        coco_iter = iter(self.coco_loader)

        if self.robot_loader:
            robot_iter = iter(self.robot_loader)

            # Alternate between datasets
            for coco_batch in coco_iter:
                yield coco_batch, 'coco'

                try:
                    robot_batch = next(robot_iter)
                    yield robot_batch, 'robot'
                except StopIteration:
                    robot_iter = iter(self.robot_loader)
                    robot_batch = next(robot_iter)
                    yield robot_batch, 'robot'
        else:
            for coco_batch in coco_iter:
                yield coco_batch, 'coco'

class EmbeddedDataPreprocessor:
    """Preprocessing utilities for embedded deployment"""

    @staticmethod
    def compress_images(image_dir: str, output_dir: str, quality: int = 85):
        """Compress images for embedded storage"""
        from PIL import Image
        import os

        input_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        compressed_count = 0
        total_size_before = 0
        total_size_after = 0

        for img_file in input_path.rglob("*.jpg"):
            try:
                with Image.open(img_file) as img:
                    # Original size
                    original_size = img_file.stat().st_size
                    total_size_before += original_size

                    # Compress and resize
                    img = img.convert('RGB')
                    img = img.resize((224, 224), Image.LANCZOS)

                    # Save compressed
                    output_file = output_path / img_file.name
                    img.save(output_file, 'JPEG', quality=quality, optimize=True)

                    # New size
                    new_size = output_file.stat().st_size
                    total_size_after += new_size

                    compressed_count += 1

            except Exception as e:
                print(f"Error compressing {img_file}: {e}")

        compression_ratio = total_size_before / total_size_after if total_size_after > 0 else 1
        print(f"Compressed {compressed_count} images")
        print(f"Size reduction: {total_size_before / 1024 / 1024:.1f} MB -> {total_size_after / 1024 / 1024:.1f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")

    @staticmethod
    def create_embedded_vocabulary(text_data: List[str], vocab_size: int = 4096) -> Dict:
        """Create minimal vocabulary for embedded deployment"""
        from collections import Counter

        # Tokenize all text
        all_tokens = []
        for text in text_data:
            tokens = text.lower().split()
            all_tokens.extend(tokens)

        # Count frequency
        token_counts = Counter(all_tokens)

        # Select most frequent tokens
        most_common = token_counts.most_common(vocab_size - 10)  # Reserve space for special tokens

        # Build vocabulary
        vocab = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
            '<reasoning>': 4, '</reasoning>': 5, '<answer>': 6, '</answer>': 7,
            '<robot>': 8, '</robot>': 9
        }

        for i, (token, count) in enumerate(most_common):
            vocab[token] = i + 10

        return vocab

    @staticmethod
    def validate_dataset_for_embedded(dataset_path: str) -> Dict:
        """Validate dataset for embedded constraints"""
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        stats = {
            'total_samples': len(data),
            'avg_caption_length': 0,
            'max_caption_length': 0,
            'missing_images': 0,
            'invalid_images': 0
        }

        caption_lengths = []

        for item in data:
            # Check caption
            caption = item.get('caption', '')
            caption_length = len(caption.split())
            caption_lengths.append(caption_length)

            # Check image
            image_path = item.get('image_path', '')
            if not image_path or not Path(image_path).exists():
                stats['missing_images'] += 1
            else:
                try:
                    with Image.open(image_path) as img:
                        if img.size[0] < 32 or img.size[1] < 32:  # Too small
                            stats['invalid_images'] += 1
                except:
                    stats['invalid_images'] += 1

        stats['avg_caption_length'] = np.mean(caption_lengths)
        stats['max_caption_length'] = max(caption_lengths)

        # Recommendations for embedded deployment
        recommendations = []

        if stats['avg_caption_length'] > 20:
            recommendations.append("Consider truncating captions to <20 words for embedded efficiency")

        if stats['missing_images'] > len(data) * 0.1:
            recommendations.append("High number of missing images - consider data cleaning")

        if stats['max_caption_length'] > 50:
            recommendations.append("Very long captions detected - may exceed embedded memory limits")

        stats['recommendations'] = recommendations

        return stats

# Utility functions for data preparation
def prepare_coco_for_embedded(input_file: str, output_file: str, max_samples: int = 10000):
    """Prepare COCO dataset for embedded training"""
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Sample data for embedded constraints
    if len(data) > max_samples:
        data = random.sample(data, max_samples)

    # Filter and clean data
    cleaned_data = []

    for item in data:
        # Check if image exists
        if 'image_path' in item and Path(item['image_path']).exists():
            # Truncate long captions
            caption = item['caption']
            if len(caption.split()) > 25:
                words = caption.split()[:25]
                item['caption'] = ' '.join(words)

            cleaned_data.append(item)

    # Save prepared data
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"Prepared {len(cleaned_data)} samples for embedded training")
    print(f"Saved to {output_file}")

    return cleaned_data

def create_robot_selection_dataset(output_file: str, num_samples: int = 1000):
    """Create synthetic robot selection dataset for training"""

    robot_types = [
        'manipulator', 'mobile_base', 'quadruped', 'humanoid',
        'aerial_drone', 'ground_vehicle', 'gripper_robot', 'inspection_robot'
    ]

    task_templates = [
        "Pick up the {object} from the {location}",
        "Move to the {location} and inspect {object}",
        "Navigate through the {environment} to reach {target}",
        "Grasp the {object} and place it on {surface}",
        "Survey the {area} for {target_object}",
        "Transport {object} from {start} to {end}",
        "Manipulate the {tool} to perform {action}",
        "Explore the {environment} and map {features}"
    ]

    objects = ['box', 'bottle', 'tool', 'component', 'package', 'part', 'item']
    locations = ['table', 'shelf', 'floor', 'container', 'rack', 'platform']
    environments = ['warehouse', 'factory', 'outdoor area', 'room', 'corridor']

    data = []

    for i in range(num_samples):
        # Random task generation
        template = random.choice(task_templates)

        task_desc = template.format(
            object=random.choice(objects),
            location=random.choice(locations),
            environment=random.choice(environments),
            target=random.choice(locations),
            surface=random.choice(locations),
            area=random.choice(environments),
            target_object=random.choice(objects),
            start=random.choice(locations),
            end=random.choice(locations),
            tool=random.choice(objects),
            action="assembly task",
            features="obstacles"
        )

        # Select appropriate robot based on task keywords
        robot_type = 'manipulator'  # default

        if 'navigate' in task_desc.lower() or 'move' in task_desc.lower():
            robot_type = random.choice(['mobile_base', 'quadruped'])
        elif 'aerial' in task_desc.lower() or 'survey' in task_desc.lower():
            robot_type = 'aerial_drone'
        elif 'pick' in task_desc.lower() or 'grasp' in task_desc.lower():
            robot_type = random.choice(['manipulator', 'gripper_robot'])
        elif 'inspect' in task_desc.lower():
            robot_type = 'inspection_robot'

        data.append({
            'task_description': task_desc,
            'robot_type': robot_type,
            'difficulty': random.choice(['easy', 'medium', 'hard']),
            'environment': random.choice(environments)
        })

    # Save dataset
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Created {len(data)} robot selection samples")
    print(f"Saved to {output_file}")

    return data
