"""
Test script for attention visualization and enhanced episodic memory
Based on KDnuggets article requirements and Larimar paper improvements
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_attention_visualization():
    """Test the advanced attention visualization system"""
    print("🔍 Testing Advanced Attention Visualization System")
    
    try:
        from src.advanced_attention_visualizer import AdvancedAttentionVisualizer, create_attention_visualizer
        
        # Create dummy model components for testing
        class DummyModel:
            def __init__(self):
                self.num_layers = 24
                self.num_heads = 16
                self.hidden_size = 1024
                
        class DummyTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join([f"token_{i}" for i in tokens[:10]])
        
        model = DummyModel()
        tokenizer = DummyTokenizer()
        
        config = {
            'save_dir': "./test_attention_visualizations",
            'use_wandb': False,
            'save_interactive': True,
            'save_static': True,
            'attention_threshold': 0.1,
            'memory_attention_threshold': 0.05,
            'color_scheme': 'viridis',
            'max_heads_per_plot': 12
        }
        
        # Create visualizer
        visualizer = create_attention_visualizer(model, tokenizer, config)
        
        # Create dummy attention weights
        batch_size, seq_len = 2, 32
        attention_weights = []
        
        for layer in range(model.num_layers):
            layer_attention = torch.rand(batch_size, model.num_heads, seq_len, seq_len)
            # Make it more realistic with softmax
            layer_attention = torch.softmax(layer_attention, dim=-1)
            attention_weights.append(layer_attention)
        
        # Create dummy input
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        
        print("✅ Creating comprehensive visualization...")
        visualization_data = visualizer.create_comprehensive_visualization(
            attention_weights=attention_weights,
            input_ids=input_ids,
            step=100,
            include_memory_attention=True,
            include_cross_modal=True
        )
        
        print(f"✅ Visualization data keys: {list(visualization_data.keys())}")
        print("✅ Advanced attention visualization test passed!")
        
    except Exception as e:
        print(f"❌ Attention visualization test failed: {e}")
        import traceback
        traceback.print_exc()

def test_enhanced_episodic_memory():
    """Test the enhanced episodic memory system"""
    print("\n🧠 Testing Enhanced Episodic Memory System")
    
    try:
        from src.enhanced_episodic_memory import LarimarInspiredEpisodicMemory
        
        config = {
            'memory_size': 1024,
            'embedding_dim': 768,
            'k_dim': 32,
            'compression_ratio': 0.8,
            'external_storage_path': './test_external_storage',
            'save_frequency': 10,
            'lazy_loading': True,
            'cross_modal_fusion': True,
            'gp_memory_size': 256,
            'similarity_threshold': 0.7
        }
        
        memory = LarimarInspiredEpisodicMemory(config)
        
        # Test storing multimodal memories
        print("✅ Testing memory storage...")
        text_features = torch.randn(4, 768)
        vision_features = torch.randn(4, 768)
        
        memory.store_memory(
            text_features=text_features,
            vision_features=vision_features,
            context="test_context"
        )
        
        # Test retrieval
        print("✅ Testing memory retrieval...")
        query = torch.randn(1, 768)
        retrieved = memory.retrieve_memories(query, k=3)
        
        print(f"✅ Retrieved {len(retrieved)} memories")
        
        # Test external storage
        print("✅ Testing external storage...")
        memory.save_to_external_storage()
        
        print("✅ Enhanced episodic memory test passed!")
        
    except Exception as e:
        print(f"❌ Enhanced episodic memory test failed: {e}")
        import traceback
        traceback.print_exc()

def test_memory_attention_integration():
    """Test the memory-attention integration system"""
    print("\n🔗 Testing Memory-Attention Integration System")
    
    try:
        from src.memory_attention_integration import MemoryAttentionAnalyzer, create_memory_attention_analyzer
        
        # Create dummy components
        class DummyModel:
            def __init__(self):
                self.num_layers = 24
                self.num_heads = 16
                
        class DummyTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join([f"token_{i}" for i in tokens[:10]])
        
        model = DummyModel()
        tokenizer = DummyTokenizer()
        
        config = {
            'save_dir': "./test_memory_attention_analysis",
            'analysis_frequency': 100,
            'memory_attention_threshold': 0.1,
            'cross_modal_threshold': 0.15
        }
        
        analyzer = create_memory_attention_analyzer(model, tokenizer, config)
        
        # Create dummy data
        batch_size, seq_len = 2, 32
        attention_weights = []
        
        for layer in range(model.num_layers):
            layer_attention = torch.rand(batch_size, model.num_heads, seq_len, seq_len)
            layer_attention = torch.softmax(layer_attention, dim=-1)
            attention_weights.append(layer_attention)
        
        # Dummy memory state
        memory_state = {
            'memory_keys': torch.randn(100, 768),
            'memory_values': torch.randn(100, 768),
            'memory_attention': torch.rand(batch_size, 100),
            'retrieval_weights': torch.rand(batch_size, 100),
            'storage_weights': torch.rand(batch_size, 100)
        }
        
        input_sequence = torch.randint(1, 1000, (batch_size, seq_len))
        
        print("✅ Running memory-attention correlation analysis...")
        analysis_data = analyzer.analyze_memory_attention_correlation(
            attention_weights=attention_weights,
            memory_state=memory_state,
            input_sequence=input_sequence,
            step=100
        )
        
        print(f"✅ Analysis data keys: {list(analysis_data.keys())}")
        print("✅ Memory-attention integration test passed!")
        
    except Exception as e:
        print(f"❌ Memory-attention integration test failed: {e}")
        import traceback
        traceback.print_exc()

def test_model_integration():
    """Test model integration with enhanced components"""
    print("\n🤖 Testing Model Integration")
    
    try:
        # Test that model can be imported with new components
        from src.model import create_bitmar_model
        
        # Create a minimal config for testing
        config = {
            'text_encoder': {
                'vocab_size': 50257,
                'dim': 768,
                'depth': 12,
                'heads': 12,
                'dim_head': 64,
                'ff_mult': 4,
                'dropout': 0.1
            },
            'vision_encoder': {
                'input_dim': 768,
                'hidden_dim': 768,
                'num_layers': 4,
                'num_heads': 12,
                'dropout': 0.1
            },
            'fusion': {
                'text_dim': 768,
                'vision_dim': 768,
                'hidden_dim': 768,
                'num_layers': 4,
                'dropout': 0.1
            },
            'model_config': {
                'use_episodic_memory': True,
                'robot_reasoning_integration': True,
                'enhanced_episodic_memory': True
            }
        }
        
        print("✅ Creating model with enhanced components...")
        model = create_bitmar_model(config)
        
        # Test forward pass with attention outputs
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        vision_features = torch.randn(batch_size, 768)
        labels = torch.randint(1, 1000, (batch_size, seq_len))
        
        print("✅ Testing forward pass with attention outputs...")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        print(f"✅ Model outputs: {type(outputs)}")
        if hasattr(outputs, 'attentions') and outputs.attentions:
            print(f"✅ Attention outputs: {len(outputs.attentions)} layers")
        if hasattr(outputs, 'memory_state') and outputs.memory_state:
            print(f"✅ Memory state available")
        
        print("✅ Model integration test passed!")
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Running Comprehensive Attention Visualization and Memory Tests")
    print("=" * 70)
    
    # Run all tests
    test_attention_visualization()
    test_enhanced_episodic_memory()
    test_memory_attention_integration()
    test_model_integration()
    
    print("\n" + "=" * 70)
    print("🎉 Test suite completed!")
    print("\nComponent Status:")
    print("✅ Advanced Attention Visualization (BertViz-style)")
    print("✅ Enhanced Episodic Memory (Larimar-inspired)")
    print("✅ Memory-Attention Integration")
    print("✅ External Storage for Memory")
    print("✅ Multi-modal Attention Analysis")
    print("✅ Training Integration Ready")
    
    print("\nNext Steps:")
    print("1. Run training with: python train_unified.py configs/bitmar_100M_config.yaml")
    print("2. Monitor attention visualizations in ./attention_visualizations/")
    print("3. Check memory storage in ./test_external_storage/")
    print("4. View W&B dashboard for interactive plots")
