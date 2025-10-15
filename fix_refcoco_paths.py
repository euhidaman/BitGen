#!/usr/bin/env python3
"""
Quick fix for RefCOCO annotations path
Moves JSON files from OpenSource/ subdirectory to mdetr_annotations/ root
"""

from pathlib import Path
import shutil

def fix_refcoco_paths(data_root: str = "data"):
    """Move RefCOCO annotations to correct location"""
    data_root = Path(data_root)
    mdetr_dir = data_root / "mdetr_annotations"
    opensource_dir = mdetr_dir / "OpenSource"
    
    if not opensource_dir.exists():
        print(f"❌ OpenSource directory not found at {opensource_dir}")
        print(f"   RefCOCO annotations may already be in the correct location")
        return False
    
    print(f"🔧 Moving RefCOCO annotations from OpenSource/ to mdetr_annotations/...")
    
    moved_count = 0
    for json_file in opensource_dir.glob("*.json"):
        dest_file = mdetr_dir / json_file.name
        if not dest_file.exists():
            shutil.move(str(json_file), str(dest_file))
            print(f"   ✓ Moved {json_file.name}")
            moved_count += 1
        else:
            print(f"   ⚠️  {json_file.name} already exists, skipping")
    
    # Remove empty OpenSource directory
    try:
        if not any(opensource_dir.iterdir()):
            opensource_dir.rmdir()
            print(f"   ✓ Removed empty OpenSource directory")
    except Exception as e:
        print(f"   ⚠️  Could not remove OpenSource directory: {e}")
    
    print(f"\n✅ Moved {moved_count} files")
    
    # Verify files are now in correct location
    print("\n📋 Verifying RefCOCO annotations:")
    required_files = [
        "final_refcoco_train.json",
        "final_refcoco+_train.json",
        "final_refcocog_train.json"
    ]
    
    all_ok = True
    for filename in required_files:
        file_path = mdetr_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {filename} - NOT FOUND")
            all_ok = False
    
    if all_ok:
        print("\n✅ All RefCOCO annotations are now in the correct location!")
        print("   You can now run training with: python src/train_stage1_vision_language.py")
    else:
        print("\n❌ Some files are still missing")
        print("   Try re-running: python download_fiber_datasets.py")
    
    return all_ok


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    fix_refcoco_paths(data_root)
