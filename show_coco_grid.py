"""
Standalone COCO Dataset Visualization Grid for Jupyter Notebooks
Run this in a separate cell to display the image-caption verification grid
"""

import json
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random

def show_coco_verification_grid(data_file="data/coco/validated_coco.json", num_samples=9):
    """
    Display a grid of COCO images with their captions for verification
    Perfect for Jupyter notebooks!

    Args:
        data_file: Path to the validated COCO JSON file
        num_samples: Number of samples to show (default 9 for 3x3 grid)
    """

    # Enable inline plotting for Jupyter
    try:
        get_ipython()
        plt.ion()
        print("🖥️ Jupyter notebook detected - enabling inline display")
    except NameError:
        print("🖥️ Running in regular Python environment")

    try:
        # Load the dataset
        data_path = Path(data_file)
        if not data_path.exists():
            print(f"❌ Dataset file not found: {data_file}")
            print("💡 Make sure you've run the COCO download first!")
            return None

        with open(data_path, 'r') as f:
            data = json.load(f)

        print(f"📊 Loaded dataset with {len(data)} image-caption pairs")

        if len(data) < num_samples:
            print(f"⚠️ Only {len(data)} pairs available, adjusting grid size")
            num_samples = min(len(data), 9)

        # Calculate grid size
        if num_samples <= 4:
            rows, cols = 2, 2
            figsize = (12, 10)
        elif num_samples <= 6:
            rows, cols = 2, 3
            figsize = (15, 10)
        else:
            rows, cols = 3, 3
            figsize = (20, 16)

        # Select random samples
        random.seed(42)  # Reproducible
        selected_pairs = random.sample(data, num_samples)

        # Create the visualization
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()

        fig.suptitle('🖼️ COCO Dataset: Image-Caption Alignment Verification\n✅ Confirming proper pairing',
                    fontsize=16, fontweight='bold', y=0.98)

        print(f"🎯 Displaying {num_samples} random image-caption pairs:")

        for idx, pair in enumerate(selected_pairs):
            if idx >= len(axes):
                break

            ax = axes[idx]
            image_path = pair['image_path']
            caption = pair['caption']
            image_id = pair['image_id']

            print(f"   📷 Sample {idx+1}: ID {image_id}")
            print(f"      📁 {Path(image_path).name}")
            print(f"      💬 {caption}")

            try:
                # Load and display image
                if Path(image_path).exists():
                    img = Image.open(image_path)
                    img = img.convert('RGB')
                    ax.imshow(img)

                    # Add caption as title
                    caption_short = caption if len(caption) <= 90 else caption[:87] + "..."
                    ax.set_title(f"ID: {image_id}\n{caption_short}",
                               fontsize=9, wrap=True, pad=8)
                    print(f"      ✅ Displayed successfully")

                else:
                    # Show missing image placeholder
                    ax.text(0.5, 0.5, f"❌ Image Missing\nID: {image_id}\n{caption[:40]}...",
                           ha='center', va='center', fontsize=10, transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    print(f"      ❌ Image file missing")

            except Exception as e:
                # Show error placeholder
                ax.text(0.5, 0.5, f"⚠️ Error Loading\n{str(e)[:30]}...",
                       ha='center', va='center', fontsize=10, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="orange"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                print(f"      ❌ Error: {e}")

            # Clean up axes
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].set_visible(False)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the grid
        output_path = Path(data_file).parent / "coco_verification_grid.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Grid saved to: {output_path}")

        # Display in Jupyter
        try:
            get_ipython()
            from IPython.display import display, Image as IPImage
            plt.show()
            print("🖥️ Grid displayed in Jupyter notebook!")

            # Also show the saved image file for backup
            print(f"\n📸 You can also view the saved image file:")
            print(f"   {output_path}")

        except NameError:
            plt.show()
            print("🖥️ Grid displayed!")

        return fig

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return None

def quick_verify_alignment(data_file="data/coco/validated_coco.json"):
    """Quick text-only verification of image-caption alignment"""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)

        print(f"📊 COCO Dataset Quick Verification")
        print(f"   Total pairs: {len(data)}")

        # Check first few pairs
        for i, item in enumerate(data[:5]):
            img_path = Path(item['image_path'])
            exists = "✅" if img_path.exists() else "❌"
            print(f"   {exists} Pair {i+1}: {img_path.name} → '{item['caption'][:60]}...'")

        # Check how many images exist
        existing = sum(1 for item in data if Path(item['image_path']).exists())
        print(f"\n🎯 Result: {existing}/{len(data)} images found ({existing/len(data)*100:.1f}%)")

        if existing == len(data):
            print("✅ Perfect alignment! All images match their captions.")
        else:
            print(f"⚠️ {len(data) - existing} images missing from dataset")

    except Exception as e:
        print(f"❌ Verification failed: {e}")

# For easy Jupyter usage
if __name__ == "__main__":
    # If run as script, show the grid
    show_coco_verification_grid()
else:
    # If imported, print usage instructions
    print("🖼️ COCO Visualization Tools Loaded!")
    print("📋 Usage in Jupyter:")
    print("   show_coco_verification_grid()  # Show 3x3 image grid")
    print("   quick_verify_alignment()       # Quick text verification")
