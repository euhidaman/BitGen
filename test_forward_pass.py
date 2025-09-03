#!/usr/bin/env python3
"""
Minimal test script to debug tensor dimension issues
"""
import torch
import yaml
from pathlib import Path

# Simple minimal imports to test just the model forward pass
from src.model import create_bitmar_model
from src.dataset import LocalizedNarrativesDataset

def test_model_forward():
    """Test model forward pass with minimal setup"""
    print("🔧 Testing model forward pass...")
    
    # Load minimal config
    config_path = Path("configs/test_with_memory.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model_config = config['model']
    model = create_bitmar_model(model_config)
    model.eval()
    
    print(f"✅ Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create minimal dataset
    data_config = config['data']
    dataset = LocalizedNarrativesDataset(**data_config)
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Test single forward pass
    try:
        sample = dataset[0]
        print(f"✅ Got sample with keys: {sample.keys()}")
        
        # Convert to batch format
        batch = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)  # Add batch dimension
            else:
                batch[key] = [value]
        
        print(f"🔄 Testing forward pass...")
        print(f"   Input shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**batch)
        
        print(f"✅ Forward pass successful!")
        print(f"   Output keys: {outputs.keys()}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_forward()
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
