"""
Enhanced Multi-Dataset Loader for BitGen Stage 1
Supports FIBER-style coarse-grained AND fine-grained training

Coarse-Grained: Image-text pairs (COCO, SBU, VG, CC3M) → ITC, ITM losses
Fine-Grained: Image-text-box data (RefCOCO, VG regions) → Phrase grounding, spatial reasoning
"""

import os
import json
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from transformers import GPT2Tokenizer
import torchvision.transforms as transforms


class BaseVisionLanguageDataset(Dataset):
    """Base class for vision-language datasets"""
    
    def __init__(
        self,
        data_root: str,
        max_seq_len: int = 256,
        vocab_size: int = 50257,
        image_size: int = 224,
        is_fine_grained: bool = False
    ):
        self.data_root = Path(data_root)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.is_fine_grained = is_fine_grained  # Whether dataset has region annotations
        
        # GPT-2 tokenizer for text
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text with GPT-2 tokenizer"""
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return tokens.squeeze(0)
    
    def load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load and transform image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return blank image on error
            return torch.zeros(3, self.image_size, self.image_size)


class COCOCaptionDataset(BaseVisionLanguageDataset):
    """COCO Captions (coarse-grained)"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        **kwargs
    ):
        super().__init__(data_root, **kwargs)
        self.split = split
        
        # Load COCO annotations
        if split == "train":
            ann_file = self.data_root / "coco" / "annotations" / "captions_train2017.json"
            image_dir = self.data_root / "coco" / "train2017"
        elif split == "val":
            ann_file = self.data_root / "coco" / "annotations" / "captions_val2017.json"
            image_dir = self.data_root / "coco" / "val2017"
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if not ann_file.exists():
            print(f"Warning: COCO annotations not found at {ann_file}")
            self.data = []
            return
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image_id to file_name mapping
        id2filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Create data samples
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            image_path = image_dir / id2filename[image_id]
            
            if image_path.exists():
                self.data.append({
                    'image_path': str(image_path),
                    'caption': caption,
                    'image_id': image_id,
                    'dataset': 'coco'
                })
        
        print(f"Loaded {len(self.data)} COCO {split} samples")
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = self.load_image(item['image_path'])
        input_ids = self.tokenize_text(item['caption'])
        
        return {
            'images': image,
            'input_ids': input_ids,
            'captions': item['caption'],
            'dataset': item['dataset'],
            'is_fine_grained': False
        }


class VisualGenomeCaptionDataset(BaseVisionLanguageDataset):
    """Visual Genome Region Descriptions (coarse-grained - using full image)"""
    
    def __init__(
        self,
        data_root: str,
        max_samples: Optional[int] = None,
        **kwargs
    ):
        super().__init__(data_root, **kwargs)
        
        # Load VG region descriptions
        ann_file = self.data_root / "visual_genome" / "region_descriptions.json"
        
        if not ann_file.exists():
            print(f"Warning: VG annotations not found at {ann_file}")
            self.data = []
            return
        
        with open(ann_file, 'r') as f:
            vg_data = json.load(f)
        
        # VG images are in two folders
        image_dirs = [
            self.data_root / "visual_genome" / "VG_100K",
            self.data_root / "visual_genome" / "VG_100K_2"
        ]
        
        # Create samples (use first region description as image caption)
        for item in vg_data:
            image_id = item['id']
            regions = item['regions']
            
            if not regions:
                continue
            
            # Try to find image in either folder
            image_path = None
            for image_dir in image_dirs:
                candidate = image_dir / f"{image_id}.jpg"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path:
                # Use first region as caption (coarse-grained)
                caption = regions[0]['phrase']
                
                self.data.append({
                    'image_path': str(image_path),
                    'caption': caption,
                    'image_id': image_id,
                    'dataset': 'visual_genome'
                })
            
            if max_samples and len(self.data) >= max_samples:
                break
        
        print(f"Loaded {len(self.data)} Visual Genome samples")
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = self.load_image(item['image_path'])
        input_ids = self.tokenize_text(item['caption'])
        
        return {
            'images': image,
            'input_ids': input_ids,
            'captions': item['caption'],
            'dataset': item['dataset'],
            'is_fine_grained': False
        }


class RefCOCODataset(BaseVisionLanguageDataset):
    """RefCOCO/+/g for fine-grained phrase grounding"""
    
    def __init__(
        self,
        data_root: str,
        dataset_name: str = "refcoco",  # refcoco, refcoco+, refcocog
        **kwargs
    ):
        super().__init__(data_root, is_fine_grained=True, **kwargs)
        self.dataset_name = dataset_name
        
        # Load MDETR annotations (use finetune_ prefix, not final_)
        ann_file = self.data_root / "mdetr_annotations" / f"finetune_{dataset_name}_train.json"
        image_dir = self.data_root / "coco" / "train2014"
        
        if not ann_file.exists():
            print(f"Warning: {dataset_name} annotations not found at {ann_file}")
            self.data = []
            return
        
        with open(ann_file, 'r') as f:
            refcoco_data = json.load(f)
        
        # Create samples
        for item in refcoco_data:
            image_id = item['image_id']
            # COCO 2014 format: COCO_train2014_000000123456.jpg
            image_path = image_dir / f"COCO_train2014_{image_id:012d}.jpg"
            
            if not image_path.exists():
                continue
            
            # Referring expression
            caption = item['caption']
            
            # Bounding boxes for grounding
            boxes = []
            tokens_positive = item.get('tokens_positive', [])
            
            if 'bbox' in item and item['bbox']:
                # Format: [x, y, width, height] → normalize to [0, 1]
                for bbox, token_span in zip(item['bbox'], tokens_positive):
                    x, y, w, h = bbox
                    boxes.append({
                        'bbox': [x, y, w, h],
                        'token_span': token_span  # Which tokens refer to this box
                    })
            
            self.data.append({
                'image_path': str(image_path),
                'caption': caption,
                'boxes': boxes,
                'image_id': image_id,
                'dataset': dataset_name
            })
        
        print(f"Loaded {len(self.data)} {dataset_name} samples with bounding boxes")
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = self.load_image(item['image_path'])
        input_ids = self.tokenize_text(item['caption'])
        
        # Normalize bounding boxes to [0, 1]
        boxes = []
        for box_item in item['boxes']:
            x, y, w, h = box_item['bbox']
            # Assuming original image size (will need actual size for proper normalization)
            # For now, keep as-is
            boxes.append({
                'bbox': torch.tensor([x, y, w, h], dtype=torch.float32),
                'token_span': box_item['token_span']
            })
        
        return {
            'images': image,
            'input_ids': input_ids,
            'captions': item['caption'],
            'boxes': boxes,
            'dataset': item['dataset'],
            'is_fine_grained': True
        }


class VisualGenomeRegionDataset(BaseVisionLanguageDataset):
    """Visual Genome regions with bounding boxes (fine-grained)"""
    
    def __init__(
        self,
        data_root: str,
        max_samples: Optional[int] = None,
        **kwargs
    ):
        super().__init__(data_root, is_fine_grained=True, **kwargs)
        
        # Load VG objects (has bbox info)
        ann_file = self.data_root / "visual_genome" / "objects.json"
        
        if not ann_file.exists():
            print(f"Warning: VG objects not found at {ann_file}")
            self.data = []
            return
        
        with open(ann_file, 'r') as f:
            vg_data = json.load(f)
        
        # VG images
        image_dirs = [
            self.data_root / "visual_genome" / "VG_100K",
            self.data_root / "visual_genome" / "VG_100K_2"
        ]
        
        # Create samples (each region = one sample)
        for item in vg_data:
            image_id = item['image_id']
            objects = item['objects']
            
            if not objects:
                continue
            
            # Find image
            image_path = None
            for image_dir in image_dirs:
                candidate = image_dir / f"{image_id}.jpg"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path:
                # Create one sample per object
                for obj in objects:
                    if 'names' not in obj or not obj['names']:
                        continue
                    
                    # Object name as caption
                    caption = obj['names'][0]
                    
                    # Bounding box
                    bbox = [obj['x'], obj['y'], obj['w'], obj['h']]
                    
                    self.data.append({
                        'image_path': str(image_path),
                        'caption': caption,
                        'boxes': [{'bbox': bbox, 'token_span': [[0, len(caption)]]}],
                        'image_id': image_id,
                        'dataset': 'vg_regions'
                    })
            
            if max_samples and len(self.data) >= max_samples:
                break
        
        print(f"Loaded {len(self.data)} VG region samples with bounding boxes")
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = self.load_image(item['image_path'])
        input_ids = self.tokenize_text(item['caption'])
        
        # Convert boxes
        boxes = []
        for box_item in item['boxes']:
            x, y, w, h = box_item['bbox']
            boxes.append({
                'bbox': torch.tensor([x, y, w, h], dtype=torch.float32),
                'token_span': box_item['token_span']
            })
        
        return {
            'images': image,
            'input_ids': input_ids,
            'captions': item['caption'],
            'boxes': boxes,
            'dataset': item['dataset'],
            'is_fine_grained': True
        }


def create_multidataset_loader(
    data_root: str,
    stage: str = "coarse",  # "coarse" or "fine" or "both"
    batch_size: int = 128,
    max_seq_len: int = 256,
    vocab_size: int = 50257,
    num_workers: int = 4,
    shuffle: bool = True,
    max_vg_samples: Optional[int] = 100000  # Limit VG to avoid memory issues
) -> DataLoader:
    """
    Create multi-dataset dataloader for BitGen Stage 1
    
    Args:
        data_root: Root directory containing all datasets
        stage: "coarse" (image-text), "fine" (region-level), or "both"
        batch_size: Batch size
        max_seq_len: Max sequence length for text
        vocab_size: Vocabulary size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        max_vg_samples: Max samples from VG (it's huge)
    
    Returns:
        DataLoader combining requested datasets
    """
    datasets = []
    
    if stage in ["coarse", "both"]:
        print("\n📦 Loading Coarse-Grained Datasets...")
        
        # COCO Captions
        try:
            coco_train = COCOCaptionDataset(
                data_root=data_root,
                split="train",
                max_seq_len=max_seq_len,
                vocab_size=vocab_size
            )
            if len(coco_train) > 0:
                datasets.append(coco_train)
        except Exception as e:
            print(f"Failed to load COCO: {e}")
        
        # Visual Genome Captions
        try:
            vg_caption = VisualGenomeCaptionDataset(
                data_root=data_root,
                max_seq_len=max_seq_len,
                vocab_size=vocab_size,
                max_samples=max_vg_samples
            )
            if len(vg_caption) > 0:
                datasets.append(vg_caption)
        except Exception as e:
            print(f"Failed to load VG captions: {e}")
    
    if stage in ["fine", "both"]:
        print("\n📦 Loading Fine-Grained Datasets...")
        
        # RefCOCO
        for refcoco_name in ["refcoco", "refcoco+", "refcocog"]:
            try:
                refcoco = RefCOCODataset(
                    data_root=data_root,
                    dataset_name=refcoco_name,
                    max_seq_len=max_seq_len,
                    vocab_size=vocab_size
                )
                if len(refcoco) > 0:
                    datasets.append(refcoco)
            except Exception as e:
                print(f"Failed to load {refcoco_name}: {e}")
        
        # Visual Genome Regions
        try:
            vg_regions = VisualGenomeRegionDataset(
                data_root=data_root,
                max_seq_len=max_seq_len,
                vocab_size=vocab_size,
                max_samples=max_vg_samples
            )
            if len(vg_regions) > 0:
                datasets.append(vg_regions)
        except Exception as e:
            print(f"Failed to load VG regions: {e}")
    
    if not datasets:
        raise ValueError(f"No datasets loaded! Check data_root: {data_root}")
    
    # Combine datasets
    combined_dataset = ConcatDataset(datasets)
    
    print(f"\n✓ Total samples: {len(combined_dataset):,}")
    print(f"  Datasets: {len(datasets)}")
    for i, ds in enumerate(datasets):
        dataset_name = ds.data[0]['dataset'] if len(ds) > 0 else "unknown"
        is_fine = "fine-grained" if ds.is_fine_grained else "coarse-grained"
        print(f"    {i+1}. {dataset_name}: {len(ds):,} samples ({is_fine})")
    
    # Create dataloader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multi_dataset_collate_fn
    )
    
    return dataloader


def multi_dataset_collate_fn(batch):
    """
    Custom collate function to handle both coarse and fine-grained data
    
    Coarse-grained: image + text
    Fine-grained: image + text + bounding boxes
    """
    images = torch.stack([item['images'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    captions = [item['captions'] for item in batch]
    datasets = [item['dataset'] for item in batch]
    is_fine_grained = [item['is_fine_grained'] for item in batch]
    
    # Handle bounding boxes (only for fine-grained samples)
    boxes = []
    for item in batch:
        if item['is_fine_grained'] and 'boxes' in item:
            boxes.append(item['boxes'])
        else:
            boxes.append([])  # Empty list for coarse-grained
    
    return {
        'images': images,
        'input_ids': input_ids,
        'captions': captions,
        'datasets': datasets,
        'is_fine_grained': is_fine_grained,
        'boxes': boxes
    }


if __name__ == "__main__":
    # Test the dataloader
    print("Testing Multi-Dataset Loader...")
    
    try:
        # Test coarse-grained only
        loader = create_multidataset_loader(
            data_root="./data",
            stage="coarse",
            batch_size=8,
            num_workers=0,
            shuffle=True
        )
        
        print(f"\nLoading first batch...")
        batch = next(iter(loader))
        
        print(f"Batch keys: {batch.keys()}")
        print(f"Images shape: {batch['images'].shape}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Captions: {batch['captions'][:2]}")
        print(f"Datasets: {batch['datasets'][:5]}")
        print(f"Fine-grained flags: {batch['is_fine_grained'][:5]}")
        
        print("\n✓ Multi-dataset loader test passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
