"""
RefCOCO/+/g Dataset Loader for FIBER Stage 2 (Fine-Grained Training)

Datasets:
- RefCOCO: Referring expression comprehension with bounding boxes
- RefCOCO+: Harder version (no location words)
- RefCOCOg: More complex expressions

Each sample contains:
- Image with bounding boxes
- Referring expressions (phrases pointing to specific regions)
- Annotations for phrase grounding

FIBER Stage 2: Fine-Grained
- Resolution: 640x640 (higher than Stage 1's 224x224)
- Task: Ground phrases to image regions
- Loss: Phrase grounding + spatial reasoning
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torchvision.transforms as transforms


class RefCOCODataset(Dataset):
    """
    RefCOCO/+/g Dataset for Phrase Grounding (FIBER Stage 2)
    
    Structure:
    - Images from COCO train2014
    - Annotations with referring expressions + bounding boxes
    - Higher resolution (640x640) for fine-grained understanding
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        dataset_type: str = "refcoco",  # refcoco, refcoco+, refcocog
        max_seq_len: int = 256,
        vocab_size: int = 50257,
        image_size: int = 640,  # FIBER Stage 2: Higher resolution
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            data_root: Root directory (e.g., "data")
            split: train/val/test
            dataset_type: refcoco, refcoco+, or refcocog
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size (GPT-2)
            image_size: Target image size (640 for Stage 2)
            transform: Optional image transforms
        """
        self.data_root = Path(data_root)
        self.split = split
        self.dataset_type = dataset_type
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.image_size = image_size
        
        # Paths
        self.image_dir = self.data_root / "coco" / "train2014"
        self.annotation_file = self.data_root / "mdetr_annotations" / f"final_{dataset_type}_{split}.json"
        
        # Check if files exist
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        # Load annotations
        print(f"📦 Loading {dataset_type} {split} annotations...")
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        print(f"✓ Loaded {len(self.annotations)} samples from {dataset_type} {split}")
        
        # Image transforms (FIBER Stage 2: Higher resolution)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Tokenizer (simple whitespace for now, can upgrade to GPT-2 tokenizer)
        self.tokenizer = self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
        """Create simple whitespace tokenizer"""
        # In production, use: from transformers import GPT2Tokenizer
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return lambda text: text.lower().split()
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get one sample with image, caption, and bounding boxes
        
        Returns:
            {
                'image': Tensor [3, H, W] - Resized image (640x640)
                'input_ids': Tensor [seq_len] - Tokenized referring expression
                'boxes': List[Dict] - Bounding boxes with token spans
                    Each box: {
                        'bbox': [x, y, w, h] - normalized coordinates
                        'token_span': [[start, end]] - which tokens refer to this box
                    }
                'caption': str - Original referring expression
                'image_id': int - COCO image ID
            }
        """
        annotation = self.annotations[idx]
        
        # Load image
        image_id = annotation.get('image_id', annotation.get('img_id'))
        image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        image_path = self.image_dir / image_filename
        
        if not image_path.exists():
            # Fallback: try without leading zeros
            image_filename = f"COCO_train2014_{image_id}.jpg"
            image_path = self.image_dir / image_filename
        
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        # Apply transforms (resize to 640x640)
        image_tensor = self.transform(image)
        
        # Get caption (referring expression)
        caption = annotation.get('caption', annotation.get('sent', ''))
        
        # Tokenize caption
        tokens = self.tokenizer(caption)
        
        # Convert to token IDs (simple: use hash % vocab_size)
        # In production: use GPT-2 tokenizer.encode()
        input_ids = [hash(token) % self.vocab_size for token in tokens]
        
        # Pad or truncate to max_seq_len
        if len(input_ids) < self.max_seq_len:
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
        else:
            input_ids = input_ids[:self.max_seq_len]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Parse bounding boxes
        boxes = []
        if 'bbox' in annotation:
            # Single box (RefCOCO format)
            bbox = annotation['bbox']  # [x, y, w, h] in original image coords
            
            # Normalize bbox to [0, 1]
            normalized_bbox = [
                bbox[0] / original_width,
                bbox[1] / original_height,
                bbox[2] / original_width,
                bbox[3] / original_height
            ]
            
            # Token span (entire caption refers to this box)
            token_span = [[0, len(tokens)]]
            
            boxes.append({
                'bbox': normalized_bbox,
                'token_span': token_span
            })
        
        elif 'tokens_positive' in annotation:
            # Multiple boxes with token alignments (MDETR format)
            raw_boxes = annotation.get('boxes', [])
            tokens_positive = annotation.get('tokens_positive', [])
            
            for box, token_positive in zip(raw_boxes, tokens_positive):
                # Normalize bbox
                normalized_bbox = [
                    box[0] / original_width,
                    box[1] / original_height,
                    box[2] / original_width,
                    box[3] / original_height
                ]
                
                # Token spans that refer to this box
                token_spans = token_positive if isinstance(token_positive[0], list) else [token_positive]
                
                boxes.append({
                    'bbox': normalized_bbox,
                    'token_span': token_spans
                })
        
        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'boxes': boxes,
            'caption': caption,
            'image_id': image_id
        }


class MixedGroundingDataset(Dataset):
    """
    Mixed Grounding Dataset (MDETR annotations)
    Combines multiple grounding datasets for diverse training
    
    Used in FIBER Stage 2 for phrase grounding pre-training
    """
    
    def __init__(
        self,
        data_root: str,
        max_seq_len: int = 256,
        vocab_size: int = 50257,
        image_size: int = 640,
        use_coco: bool = False  # Whether to include COCO images
    ):
        """
        Args:
            data_root: Root directory
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
            image_size: Target image size (640 for Stage 2)
            use_coco: If True, use full mixed dataset. If False, use no-coco version
        """
        self.data_root = Path(data_root)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.image_size = image_size
        
        # Annotation file
        if use_coco:
            annotation_file = self.data_root / "mdetr_annotations" / "final_mixed_train.json"
        else:
            annotation_file = self.data_root / "mdetr_annotations" / "final_mixed_train_no_coco.json"
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        print(f"📦 Loading Mixed Grounding annotations...")
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        print(f"✓ Loaded {len(self.annotations)} samples from Mixed Grounding")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Simple tokenizer
        self.tokenizer = lambda text: text.lower().split()
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get one sample with image and grounding annotations"""
        annotation = self.annotations[idx]
        
        # Determine image path based on dataset_name
        dataset_name = annotation.get('dataset_name', 'coco')
        image_id = annotation.get('image_id', annotation.get('img_id'))
        
        if 'coco' in dataset_name.lower():
            image_path = self.data_root / "coco" / "train2014" / f"COCO_train2014_{image_id:012d}.jpg"
        elif 'gqa' in dataset_name.lower():
            image_path = self.data_root / "gqa" / "images" / f"{image_id}.jpg"
        else:
            # Fallback: try to construct path from annotation
            image_path = self.data_root / dataset_name / annotation.get('file_name', f"{image_id}.jpg")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        image_tensor = self.transform(image)
        
        # Get caption
        caption = annotation.get('caption', '')
        
        # Tokenize
        tokens = self.tokenizer(caption)
        input_ids = [hash(token) % self.vocab_size for token in tokens]
        
        # Pad/truncate
        if len(input_ids) < self.max_seq_len:
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
        else:
            input_ids = input_ids[:self.max_seq_len]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Parse boxes (MDETR format)
        boxes = []
        if 'boxes' in annotation and 'tokens_positive' in annotation:
            raw_boxes = annotation['boxes']
            tokens_positive = annotation['tokens_positive']
            
            for box, token_positive in zip(raw_boxes, tokens_positive):
                normalized_bbox = [
                    box[0] / original_width,
                    box[1] / original_height,
                    box[2] / original_width,
                    box[3] / original_height
                ]
                
                token_spans = token_positive if isinstance(token_positive[0], list) else [token_positive]
                
                boxes.append({
                    'bbox': normalized_bbox,
                    'token_span': token_spans
                })
        
        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'boxes': boxes,
            'caption': caption,
            'image_id': image_id
        }


def collate_fn_with_boxes(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching samples with variable number of boxes
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary with images, input_ids, and list of boxes per sample
    """
    images = torch.stack([sample['image'] for sample in batch])
    input_ids = torch.stack([sample['input_ids'] for sample in batch])
    
    # Keep boxes as list (variable length per sample)
    boxes = [sample['boxes'] for sample in batch]
    captions = [sample['caption'] for sample in batch]
    image_ids = [sample['image_id'] for sample in batch]
    
    return {
        'images': images,
        'input_ids': input_ids,
        'boxes': boxes,
        'captions': captions,
        'image_ids': image_ids
    }


# Example usage
if __name__ == "__main__":
    # Test RefCOCO dataset
    dataset = RefCOCODataset(
        data_root="data",
        split="train",
        dataset_type="refcoco",
        image_size=640
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Caption: {sample['caption']}")
    print(f"Number of boxes: {len(sample['boxes'])}")
    if sample['boxes']:
        print(f"First box: {sample['boxes'][0]}")
