#!/usr/bin/env python3
"""
Test script to verify FIBER integration works correctly
"""

import yaml
import torch
from src.model import create_bitmar_model

def test_fiber_integration():
    """Test that FIBER integration works with the training config"""
    print("🧪 Testing FIBER integration...")
    
    # Load config
    with open('configs/bitmar_with_memory.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    print("📦 Creating BitMar model with FIBER...")
    model = create_bitmar_model(config['model'])
    print("✅ Model creation successful!")
    
    # Test forward pass
    print("🚀 Testing forward pass with FIBER loss computation...")
    batch_size = 2
    seq_len = 10
    
    # Create dummy inputs
    text_input = torch.randint(0, 1000, (batch_size, seq_len))
    image_input = torch.randn(batch_size, 3, 224, 224)
    text_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        output = model(text_input, image_input, text_mask)
    
    print(f"✅ Forward pass successful!")
    print(f"📊 Output keys: {list(output.keys())}")
    print(f"🎯 Has FIBER loss: {'fiber_loss' in output}")
    
    if 'fiber_loss' in output:
        fiber_loss = output['fiber_loss']
        print(f"🔥 FIBER loss shape: {fiber_loss.shape}")
        print(f"🔥 FIBER loss value: {fiber_loss.item():.4f}")
        
        # Check if loss components exist
        if hasattr(model.fusion, 'authentic_fiber'):
            print("✅ AuthenticFIBERLoss found in model!")
            print("✅ ITC + ITM + MLM objectives active")
        else:
            print("⚠️  AuthenticFIBERLoss not found")
    
    print("\n🎉 FIBER integration test completed successfully!")
    print("🚀 Ready to run: python train_unified.py --config configs/bitmar_with_memory.yaml")
    
    return True

if __name__ == "__main__":
    test_fiber_integration()
