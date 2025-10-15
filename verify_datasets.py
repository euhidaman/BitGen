"""
Dataset Verification Script for BitGen
Checks if all required datasets are downloaded and in the correct locations
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def check_directory(path: Path, name: str, min_files: int = 0) -> Tuple[bool, int]:
    """Check if directory exists and has minimum number of files"""
    if not path.exists():
        return False, 0
    
    files = list(path.glob("*.jpg"))
    count = len(files)
    return count >= min_files, count


def check_file(path: Path, name: str) -> bool:
    """Check if file exists"""
    return path.exists()


def verify_datasets(data_root: str = "data") -> Dict:
    """Verify all datasets required for training"""
    
    data_root = Path(data_root)
    results = {
        'all_ok': True,
        'critical_missing': [],
        'warnings': [],
        'details': {}
    }
    
    print("="*70)
    print("BitGen Dataset Verification")
    print("="*70)
    print()
    
    # COCO 2014 (CRITICAL - needed for RefCOCO Phase 2)
    print("📦 COCO 2014 (Critical for Phase 2 - RefCOCO)")
    print("-" * 70)
    
    train2014_ok, train2014_count = check_directory(
        data_root / "coco" / "train2014",
        "COCO train2014",
        min_files=80000
    )
    print(f"   {'✅' if train2014_ok else '❌'} train2014/: {train2014_count:,} images (expected ~82,783)")
    results['details']['coco_train2014'] = {'count': train2014_count, 'ok': train2014_ok}
    
    val2014_ok, val2014_count = check_directory(
        data_root / "coco" / "val2014",
        "COCO val2014",
        min_files=40000
    )
    print(f"   {'✅' if val2014_ok else '❌'} val2014/: {val2014_count:,} images (expected ~40,504)")
    results['details']['coco_val2014'] = {'count': val2014_count, 'ok': val2014_ok}
    
    if not train2014_ok:
        results['all_ok'] = False
        results['critical_missing'].append("COCO train2014 (needed for RefCOCO Phase 2)")
    
    print()
    
    # COCO 2017 (Main training)
    print("📦 COCO 2017 (Main training)")
    print("-" * 70)
    
    train2017_ok, train2017_count = check_directory(
        data_root / "coco" / "train2017",
        "COCO train2017",
        min_files=110000
    )
    print(f"   {'✅' if train2017_ok else '❌'} train2017/: {train2017_count:,} images (expected ~118,287)")
    results['details']['coco_train2017'] = {'count': train2017_count, 'ok': train2017_ok}
    
    val2017_ok, val2017_count = check_directory(
        data_root / "coco" / "val2017",
        "COCO val2017",
        min_files=4000
    )
    print(f"   {'✅' if val2017_ok else '❌'} val2017/: {val2017_count:,} images (expected ~5,000)")
    results['details']['coco_val2017'] = {'count': val2017_count, 'ok': val2017_ok}
    
    if not train2017_ok:
        results['all_ok'] = False
        results['critical_missing'].append("COCO train2017 (needed for Phase 1)")
    
    print()
    
    # COCO Annotations
    print("📦 COCO Annotations")
    print("-" * 70)
    
    ann_2014_train = check_file(
        data_root / "coco" / "annotations" / "instances_train2014.json",
        "instances_train2014.json"
    )
    print(f"   {'✅' if ann_2014_train else '❌'} annotations/instances_train2014.json")
    
    ann_2017_captions = check_file(
        data_root / "coco" / "annotations" / "captions_train2017.json",
        "captions_train2017.json"
    )
    print(f"   {'✅' if ann_2017_captions else '❌'} annotations/captions_train2017.json")
    results['details']['coco_annotations'] = {
        'train2014': ann_2014_train,
        'captions2017': ann_2017_captions
    }
    
    if not ann_2017_captions:
        results['all_ok'] = False
        results['critical_missing'].append("COCO captions annotations")
    
    print()
    
    # Visual Genome
    print("📦 Visual Genome")
    print("-" * 70)
    
    vg_100k_ok, vg_100k_count = check_directory(
        data_root / "visual_genome" / "VG_100K",
        "VG_100K",
        min_files=60000  # Updated: dataset now has ~64K images (was 108K in older version)
    )
    print(f"   {'✅' if vg_100k_ok else '❌'} VG_100K/: {vg_100k_count:,} images (expected ~64,346)")
    results['details']['vg_100k'] = {'count': vg_100k_count, 'ok': vg_100k_ok}
    
    vg_100k2_ok, vg_100k2_count = check_directory(
        data_root / "visual_genome" / "VG_100K_2",
        "VG_100K_2",
        min_files=40000  # Updated: dataset now has ~44K images (was 108K in older version)
    )
    print(f"   {'✅' if vg_100k2_ok else '❌'} VG_100K_2/: {vg_100k2_count:,} images (expected ~43,903)")
    results['details']['vg_100k_2'] = {'count': vg_100k2_count, 'ok': vg_100k2_ok}
    
    vg_regions = check_file(
        data_root / "visual_genome" / "region_descriptions.json",
        "region_descriptions.json"
    )
    print(f"   {'✅' if vg_regions else '❌'} region_descriptions.json")
    
    vg_objects = check_file(
        data_root / "visual_genome" / "objects.json",
        "objects.json"
    )
    print(f"   {'✅' if vg_objects else '❌'} objects.json")
    
    results['details']['vg_annotations'] = {
        'regions': vg_regions,
        'objects': vg_objects
    }
    
    if not (vg_100k_ok or vg_100k2_ok):
        results['warnings'].append("Visual Genome images missing (Phase 1 will have less data)")
    if not vg_regions:
        results['warnings'].append("Visual Genome region descriptions missing")
    if not vg_objects:
        results['warnings'].append("Visual Genome objects missing (Phase 2 fine-grained will have less data)")
    
    print()
    
    # RefCOCO Annotations
    print("📦 RefCOCO/+/g Annotations (MDETR)")
    print("-" * 70)
    
    # Debug: Show what files actually exist
    mdetr_dir = data_root / "mdetr_annotations"
    if mdetr_dir.exists():
        print(f"   📁 Looking in: {mdetr_dir}")
        json_files = list(mdetr_dir.glob("*.json"))
        if json_files:
            print(f"   📄 Found {len(json_files)} JSON files:")
            for f in json_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"      - {f.name} ({size_mb:.1f} MB)")
        else:
            print(f"   ⚠️  No JSON files found in {mdetr_dir}")
            # Check OpenSource subdirectory
            opensource_dir = mdetr_dir / "OpenSource"
            if opensource_dir.exists():
                print(f"   ⚠️  Found OpenSource/ subdirectory - files may need to be moved")
                print(f"      Run: python fix_refcoco_paths.py")
    else:
        print(f"   ❌ Directory does not exist: {mdetr_dir}")
    print()
    
    # Check for finetune_ prefix (actual MDETR format)
    refcoco = check_file(
        data_root / "mdetr_annotations" / "finetune_refcoco_train.json",
        "finetune_refcoco_train.json"
    )
    print(f"   {'✅' if refcoco else '❌'} finetune_refcoco_train.json")
    
    refcoco_plus = check_file(
        data_root / "mdetr_annotations" / "finetune_refcoco+_train.json",
        "finetune_refcoco+_train.json"
    )
    print(f"   {'✅' if refcoco_plus else '❌'} finetune_refcoco+_train.json")
    
    refcocog = check_file(
        data_root / "mdetr_annotations" / "finetune_refcocog_train.json",
        "finetune_refcocog_train.json"
    )
    print(f"   {'✅' if refcocog else '❌'} finetune_refcocog_train.json")
    
    results['details']['refcoco_annotations'] = {
        'refcoco': refcoco,
        'refcoco+': refcoco_plus,
        'refcocog': refcocog
    }
    
    if not (refcoco or refcoco_plus or refcocog):
        results['all_ok'] = False
        results['critical_missing'].append("RefCOCO annotations (needed for Phase 2)")
    
    print()
    print("="*70)
    print("Summary")
    print("="*70)
    
    if results['all_ok'] and not results['warnings']:
        print("✅ ALL DATASETS VERIFIED - READY FOR TRAINING!")
        print()
        print("Total images:")
        total_images = (
            train2014_count + val2014_count +
            train2017_count + val2017_count +
            vg_100k_count + vg_100k2_count
        )
        print(f"   📊 {total_images:,} images available for training")
        print()
        print("Next step:")
        print("   python src/train_stage1_vision_language.py")
    else:
        print("❌ SOME DATASETS MISSING OR INCOMPLETE")
        print()
        
        if results['critical_missing']:
            print("🔴 Critical issues (training will fail):")
            for issue in results['critical_missing']:
                print(f"   - {issue}")
            print()
        
        if results['warnings']:
            print("⚠️  Warnings (training will work but with reduced data):")
            for warning in results['warnings']:
                print(f"   - {warning}")
            print()
        
        print("To fix:")
        print("   python download_fiber_datasets.py")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    import sys
    
    # Allow custom data root
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    results = verify_datasets(data_root)
    
    # Exit with error code if critical issues
    if not results['all_ok']:
        sys.exit(1)
    else:
        sys.exit(0)
