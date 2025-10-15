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
        ax.set_title(sample['dataset'], fontsize=11, fontweight='bold', pad=10)
        
        # Caption below image - make it more visible
        caption = sample['caption']
        if len(caption) > 150:
            caption = caption[:150] + "..."
        wrapped_caption = textwrap.fill(caption, width=50)
        
        # Use text box for better visibility
        ax.text(
            0.5, -0.15,
            wrapped_caption,
            fontsize=9,
            ha='center',
            va='top',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8, edgecolor='black')
        )
        
        ax.axis('off')
        
    except Exception as e:
        ax.axis('off')
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', fontsize=8)


def create_dataset_grid(data_root: Path, dataset_name: str, loader_func, loader_args=None):
    """Create a 3x3 grid for a specific dataset"""
    print(f"📷 Loading 9 samples from {dataset_name}...")
    
    samples = []
    attempts = 0
    max_attempts = 100
    
    while len(samples) < 9 and attempts < max_attempts:
        if loader_args:
            sample = loader_func(data_root, *loader_args)
        else:
            sample = loader_func(data_root)
        
        if sample:
            samples.append(sample)
        attempts += 1
    
    if len(samples) == 0:
        print(f"  ⚠️  No samples found for {dataset_name}")
        return None
    
    # Fill remaining slots if needed
    while len(samples) < 9:
        samples.append(None)
    
    print(f"  ✓ Loaded {len([s for s in samples if s])} samples")
    
    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f'{dataset_name} - Sample Images', fontsize=20, fontweight='bold', y=0.995)
    
    for idx, (ax, sample) in enumerate(zip(axes.flat, samples)):
        visualize_sample(ax, sample)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig


def visualize_all_datasets(data_root: str = "data", output_dir: str = "."):
    """Visualize 9 random samples from EACH dataset (separate grids)"""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("🎨 BitGen Dataset Visualization")
    print("Creating separate 3x3 grids for each dataset...")
    print("="*80 + "\n")
    
    saved_files = []
    
    # 1. COCO Captions
    fig = create_dataset_grid(data_root, "COCO Captions (Image-level)", load_coco_sample)
    if fig:
        save_path = output_dir / "coco_samples.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {save_path}\n")
        saved_files.append(save_path)
        plt.close(fig)
    
    # 2. Visual Genome
    fig = create_dataset_grid(data_root, "Visual Genome (Region Descriptions)", load_visual_genome_sample)
    if fig:
        save_path = output_dir / "visual_genome_samples.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {save_path}\n")
        saved_files.append(save_path)
        plt.close(fig)
    
    # 3. RefCOCO
    fig = create_dataset_grid(data_root, "RefCOCO (Referring Expressions)", load_refcoco_sample, ['refcoco'])
    if fig:
        save_path = output_dir / "refcoco_samples.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {save_path}\n")
        saved_files.append(save_path)
        plt.close(fig)
    
    # 4. RefCOCO+
    fig = create_dataset_grid(data_root, "RefCOCO+ (Referring Expressions - No Location)", load_refcoco_sample, ['refcoco+'])
    if fig:
        save_path = output_dir / "refcoco_plus_samples.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {save_path}\n")
        saved_files.append(save_path)
        plt.close(fig)
    
    # 5. RefCOCOg
    fig = create_dataset_grid(data_root, "RefCOCOg (Referring Expressions - Long)", load_refcoco_sample, ['refcocog'])
    if fig:
        save_path = output_dir / "refcocog_samples.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved: {save_path}\n")
        saved_files.append(save_path)
        plt.close(fig)
    
    print("="*80)
    print(f"✅ Visualization complete! Generated {len(saved_files)} grids")
    print("\nSaved files:")
    for f in saved_files:
        print(f"  📄 {f}")
    print("\nDataset Types:")
    print("  📝 COCO: Image-level captions (coarse-grained)")
    print("  🎯 Visual Genome: Region descriptions + bounding boxes (fine-grained)")
    print("  🔍 RefCOCO/+/g: Referring expressions + bounding boxes (phrase grounding)")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset-visualization"
    
    visualize_all_datasets(data_root, output_dir)
