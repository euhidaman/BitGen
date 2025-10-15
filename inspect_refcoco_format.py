#!/usr/bin/env python3
"""
Inspect RefCOCO annotation format to understand the structure
"""

import json
from pathlib import Path

def inspect_refcoco_format(data_root: str = "data"):
    """Inspect the structure of RefCOCO annotations"""
    data_root = Path(data_root)
    
    ann_file = data_root / "mdetr_annotations" / "finetune_refcoco_train.json"
    
    if not ann_file.exists():
        print(f"❌ File not found: {ann_file}")
        return
    
    print(f"📂 Reading: {ann_file}")
    print(f"📊 File size: {ann_file.stat().st_size / (1024*1024):.1f} MB\n")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"🔍 Top-level type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"📋 Keys: {list(data.keys())}")
        print()
        
        for key in data.keys():
            val = data[key]
            print(f"  {key}:")
            print(f"    Type: {type(val)}")
            if isinstance(val, list):
                print(f"    Length: {len(val)}")
                if len(val) > 0:
                    print(f"    First item type: {type(val[0])}")
                    print(f"    First item: {val[0]}")
            elif isinstance(val, dict):
                print(f"    Keys: {list(val.keys())[:10]}")
            print()
    
    elif isinstance(data, list):
        print(f"📊 List length: {len(data)}")
        if len(data) > 0:
            print(f"\n🔍 First item:")
            print(f"  Type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"  Keys: {list(data[0].keys())}")
                print(f"\n  Full first item:")
                for k, v in data[0].items():
                    if isinstance(v, (str, int, float)):
                        print(f"    {k}: {v}")
                    elif isinstance(v, list):
                        print(f"    {k}: [list of {len(v)} items]")
                        if len(v) > 0:
                            print(f"      First: {v[0]}")
                    else:
                        print(f"    {k}: {type(v)}")
            else:
                print(f"  Value: {data[0]}")
            
            if len(data) > 1:
                print(f"\n🔍 Second item keys: {list(data[1].keys()) if isinstance(data[1], dict) else type(data[1])}")
    
    else:
        print(f"⚠️  Unexpected type: {type(data)}")
        print(f"Value: {data}")


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    inspect_refcoco_format(data_root)
