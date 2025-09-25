#!/usr/bin/env python3
"""
COCO Dataset Visualization - 3x3 Grid
Shows 9 random image-caption pairs to verify correct pairing
"""

import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import random
from PIL import Image
import numpy as np

def visualize_coco_grid(dataset_file: str = "data/coco/validated_coco.json", output_file: str = "coco_grid_visualization.png"):
    """Create a 3x3 grid of images with their captions"""

    print("🖼️ Loading COCO dataset for visualization...")

    # Load the dataset
    with open(dataset_file, 'r') as f:
        data = json.load(f)

    print(f"📊 Dataset contains {len(data)} image-caption pairs")

    if len(data) < 9:
        print("⚠️ Need at least 9 pairs for 3x3 grid")
        return

    # Select 9 random pairs
    random.seed(42)  # For reproducible results
    selected_pairs = random.sample(data, 9)

    # Create the figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('COCO Dataset: Image-Caption Pairs Verification', fontsize=16, fontweight='bold')

    for idx, pair in enumerate(selected_pairs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        image_path = pair['image_path']
        caption = pair['caption']
        image_id = pair['image_id']

        print(f"📷 Processing pair {idx+1}: Image ID {image_id}")
        print(f"   📁 Path: {image_path}")
        print(f"   💬 Caption: {caption}")

        try:
            # Load and display image
            if Path(image_path).exists():
                img = Image.open(image_path)
                img = img.convert('RGB')  # Ensure RGB format
                ax.imshow(img)

                # Add caption as title (truncate if too long)
                caption_short = caption if len(caption) <= 60 else caption[:57] + "..."
                ax.set_title(f"ID: {image_id}\n{caption_short}", fontsize=10, wrap=True, pad=10)

            else:
                # Image doesn't exist - show placeholder
                ax.text(0.5, 0.5, f"Image Missing\nID: {image_id}\n{caption[:30]}...",
                       ha='center', va='center', fontsize=8, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            # Show error placeholder
            ax.text(0.5, 0.5, f"Error Loading\nID: {image_id}\n{str(e)[:30]}...",
                   ha='center', va='center', fontsize=8, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="salmon"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        print(f"   ✅ Processed pair {idx+1}")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"💾 Visualization saved to: {output_file}")

    # Also try to display if possible
    try:
        plt.show()
    except:
        print("📺 Display not available in current environment")

    return True

def analyze_dataset_stats(dataset_file: str = "data/coco/validated_coco.json"):
    """Analyze dataset statistics"""

    print("\n📊 DATASET ANALYSIS")
    print("=" * 50)

    with open(dataset_file, 'r') as f:
        data = json.load(f)

    print(f"📈 Total pairs: {len(data)}")

    if data:
        # Caption length stats
        caption_lengths = [len(item['caption']) for item in data]
        print(f"📝 Caption length - Min: {min(caption_lengths)}, Max: {max(caption_lengths)}, Avg: {sum(caption_lengths)/len(caption_lengths):.1f}")

        # Check image existence
        existing_images = sum(1 for item in data if Path(item['image_path']).exists())
        print(f"🖼️ Images found: {existing_images}/{len(data)} ({existing_images/len(data)*100:.1f}%)")

        # Source file distribution
        if 'source_file' in data[0]:
            sources = {}
            for item in data:
                source = item.get('source_file', 'unknown')
                sources[source] = sources.get(source, 0) + 1

            print(f"📂 Source files:")
            for source, count in sources.items():
                print(f"   • {source}: {count} pairs")

        # Show first 3 examples
        print(f"\n🔍 FIRST 3 EXAMPLES:")
        for i, item in enumerate(data[:3]):
            print(f"   {i+1}. ID {item['image_id']}: {item['caption']}")
            print(f"      📁 {item['image_path']}")

    return data

if __name__ == "__main__":
    dataset_file = "data/coco/validated_coco.json"

    if not Path(dataset_file).exists():
        print(f"❌ Dataset file not found: {dataset_file}")
        print("💡 Run: python bitgen_cli.py download --output_dir data/coco")
        exit(1)

    # Analyze the dataset
    data = analyze_dataset_stats(dataset_file)

    # Create visualization
    if data:
        visualize_coco_grid(dataset_file)
    else:
        print("❌ No data to visualize")
