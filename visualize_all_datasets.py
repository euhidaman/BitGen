#!/usr/bin/env python3
"""
Visualize Samples from All Datasets

Randomly picks and displays 9 images from different datasets with their annotations:
- COCO: Image + caption
- Visual Genome: Image + region descriptions + bounding boxes
- RefCOCO/+/g: Image + referring expressions + bounding boxes

Shows what each dataset provides for training.
"""

import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import textwrap


def load_coco_sample(data_root: Path) -> Dict:
    """Load random COCO image with caption"""
    ann_file = data_root / "coco" / "annotations" / "captions_train2017.json"
    image_dir = data_root / "coco" / "train2017"
    
    if not ann_file.exists():
        return None
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Pick random annotation
    ann = random.choice(coco_data['annotations'])
    
    # Find corresponding image
    image_info = next((img for img in coco_data['images'] if img['id'] == ann['image_id']), None)
    if not image_info:
        return None
    
    image_path = image_dir / image_info['file_name']
    if not image_path.exists():
        return None
    
    return {
        'dataset': 'COCO Captions',
        'image_path': image_path,
        'caption': ann['caption'],
        'boxes': [],
        'type': 'caption'
    }


def load_visual_genome_sample(data_root: Path) -> Dict:
    """Load random Visual Genome image with regions"""
    ann_file = data_root / "visual_genome" / "region_descriptions.json"
    image_dirs = [
        data_root / "visual_genome" / "VG_100K",
        data_root / "visual_genome" / "VG_100K_2"
    ]
    
    if not ann_file.exists():
        return None
    
    with open(ann_file, 'r') as f:
        vg_data = json.load(f)
    
    # Pick random image with regions
    attempts = 0
    while attempts < 50:
        item = random.choice(vg_data)
        image_id = item['id']
        regions = item.get('regions', [])
        
        if not regions:
            attempts += 1
            continue
        
        # Find image file
        image_path = None
        for image_dir in image_dirs:
            candidate = image_dir / f"{image_id}.jpg"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path:
            # Get first few regions with bboxes
            boxes = []
            for region in regions[:5]:  # Limit to 5 regions
                if 'x' in region and 'y' in region and 'width' in region and 'height' in region:
                    boxes.append({
                        'bbox': [region['x'], region['y'], region['width'], region['height']],
                        'text': region.get('phrase', '')
                    })
            
            if boxes:
                return {
                    'dataset': 'Visual Genome',
                    'image_path': image_path,
                    'caption': f"{len(regions)} regions total",
                    'boxes': boxes,
                    'type': 'regions'
                }
        
        attempts += 1
    
    return None


def load_refcoco_sample(data_root: Path, dataset_name: str = "refcoco") -> Dict:
    """Load random RefCOCO image with referring expression"""
    ann_file = data_root / "mdetr_annotations" / f"finetune_{dataset_name}_train.json"
    image_dir = data_root / "coco" / "train2014"
    
    if not ann_file.exists():
        return None
    
    with open(ann_file, 'r') as f:
        mdetr_data = json.load(f)
    
    # Pick random image
    attempts = 0
    while attempts < 50:
        img_info = random.choice(mdetr_data['images'])
        img_id = img_info['id']
        caption = img_info.get('caption', '')
        
        if not caption:
            attempts += 1
            continue
        
        # Find annotations for this image
        anns = [ann for ann in mdetr_data['annotations'] if ann['image_id'] == img_id]
        
        if not anns:
            attempts += 1
            continue
        
        image_path = image_dir / img_info['file_name']
        if not image_path.exists():
            attempts += 1
            continue
        
        # Get bounding boxes
        boxes = []
        for ann in anns:
            if 'bbox' in ann and ann['bbox']:
                bbox = ann['bbox']
                tokens_positive = ann.get('tokens_positive', [])
                
                # Extract text from tokens_positive (char spans)
                box_text = ""
                if tokens_positive:
                    for start, end in tokens_positive:
                        box_text += caption[start:end] + " "
                
                boxes.append({
                    'bbox': bbox,
                    'text': box_text.strip() or "object"
                })
        
        if boxes:
            return {
                'dataset': f'RefCOCO{"+" if "plus" in dataset_name else ("g" if "g" in dataset_name else "")}',
                'image_path': image_path,
                'caption': caption,
                'boxes': boxes,
                'type': 'grounding'
            }
        
        attempts += 1
    
    return None


def visualize_sample(ax: plt.Axes, sample: Dict):
    """Visualize a single sample with annotations"""
    if sample is None:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        return
    
    # Load and display image
    try:
        img = Image.open(sample['image_path']).convert('RGB')
        ax.imshow(img)
        
        # Draw bounding boxes if present
        if sample['boxes']:
            for i, box_data in enumerate(sample['boxes']):
                bbox = box_data['bbox']
                x, y, w, h = bbox
                
                # Different colors for different boxes
                colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
                color = colors[i % len(colors)]
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Draw text label
                text = box_data.get('text', '')
                if text:
                    # Wrap text if too long
                    wrapped_text = textwrap.fill(text, width=20)
                    ax.text(
                        x, y - 5,
                        wrapped_text,
                        fontsize=8,
                        color='white',
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
                    )
        
        # Title: dataset name
        ax.set_title(sample['dataset'], fontsize=10, fontweight='bold')
        
        # Caption below image
        caption = sample['caption']
        if len(caption) > 100:
            caption = caption[:100] + "..."
        wrapped_caption = textwrap.fill(caption, width=40)
        ax.set_xlabel(wrapped_caption, fontsize=8, wrap=True)
        
        ax.axis('off')
        
    except Exception as e:
        ax.axis('off')
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', fontsize=8)


def visualize_all_datasets(data_root: str = "data", save_path: str = "dataset_samples.png"):
    """Visualize 9 random samples from all datasets"""
    data_root = Path(data_root)
    
    print("🎨 Visualizing samples from all datasets...")
    print("="*60)
    
    # Collect samples from different datasets
    samples = []
    
    # 3 COCO samples
    print("📷 Loading COCO samples...")
    for _ in range(3):
        sample = load_coco_sample(data_root)
        if sample:
            samples.append(sample)
    
    # 2 Visual Genome samples
    print("📷 Loading Visual Genome samples...")
    for _ in range(2):
        sample = load_visual_genome_sample(data_root)
        if sample:
            samples.append(sample)
    
    # 2 RefCOCO samples
    print("📷 Loading RefCOCO samples...")
    for dataset_name in ['refcoco', 'refcoco+']:
        sample = load_refcoco_sample(data_root, dataset_name)
        if sample:
            samples.append(sample)
    
    # 2 RefCOCOg samples
    print("📷 Loading RefCOCOg samples...")
    for _ in range(2):
        sample = load_refcoco_sample(data_root, 'refcocog')
        if sample:
            samples.append(sample)
    
    # Fill remaining slots with any available samples
    while len(samples) < 9:
        sample = random.choice([
            load_coco_sample(data_root),
            load_visual_genome_sample(data_root),
            load_refcoco_sample(data_root, random.choice(['refcoco', 'refcoco+', 'refcocog']))
        ])
        if sample:
            samples.append(sample)
    
    # Shuffle and take 9
    random.shuffle(samples)
    samples = samples[:9]
    
    print(f"✓ Loaded {len(samples)} samples")
    print("="*60)
    
    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('BitGen Dataset Samples (Image Captions + Bounding Boxes)', fontsize=16, fontweight='bold')
    
    for idx, (ax, sample) in enumerate(zip(axes.flat, samples)):
        print(f"  [{idx+1}/9] Visualizing {sample['dataset'] if sample else 'empty'}...")
        visualize_sample(ax, sample)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization: {save_path}")
    
    # Show
    try:
        plt.show()
    except:
        print("(Display not available, but image saved)")
    
    print("\n✅ Visualization complete!")
    print("\nDataset Types:")
    print("  📝 COCO: Image-level captions (coarse-grained)")
    print("  🎯 Visual Genome: Region descriptions + bounding boxes (fine-grained)")
    print("  🔍 RefCOCO/+/g: Referring expressions + bounding boxes (phrase grounding)")


if __name__ == "__main__":
    import sys
    
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    save_path = sys.argv[2] if len(sys.argv) > 2 else "dataset_samples.png"
    
    visualize_all_datasets(data_root, save_path)
