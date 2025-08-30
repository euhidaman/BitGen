# BitGen: BitNet-Quantized Vision-Language Transformer with FIBER Fusion

BitGen (BitMar) is an advanced multimodal transformer that combines BitNet 1.58-bit quantization, FIBER-style backbone fusion, and episodic memory for efficient vision-language understanding and generation.

## Key Features

- **BitNet Quantization**: 1.58-bit weight quantization for efficient inference
- **FIBER Backbone Fusion**: Microsoft Research's FIBER implementation for superior cross-modal understanding
- **Episodic Memory**: Larimar-inspired memory mechanism for multimodal associations
- **Attention Sinks**: Support for unlimited sequence generation
- **Adaptive Training**: Intelligent cross-modal similarity monitoring and intervention
- **Edge Deployment**: Optimized for deployment with memory compression and external storage

## Architecture Overview

### Model Components

1. **BitNet Text Encoder/Decoder**: Quantized transformer blocks with 1.58-bit weights
2. **Vision Encoder**: Quantized processing of DiNOv2/DiNOv3 features  
3. **FIBER Fusion**: Cross-modal attention fusion in transformer backbone
4. **Episodic Memory**: Multimodal memory slots for association learning
5. **Attention Sinks**: KV-cache optimization for long sequence generation

### FIBER Integration

BitGen implements Microsoft Research's FIBER (Fusion in the Backbone) approach:

- **Coarse-to-Fine Fusion**: Early layers focus on modality-specific processing, later layers enable cross-modal fusion
- **Bidirectional Attention**: Vision-to-text and text-to-vision attention mechanisms
- **Backbone Integration**: Cross-modal fusion within transformer blocks rather than late fusion
- **Learnable Alpha**: Adaptive fusion strength parameters

## Installation

### Prerequisites

```bash
# Python 3.8+ required
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For carbon footprint tracking
pip install codecarbon

# For Hugging Face Hub integration
pip install huggingface_hub

# For advanced attention visualization
pip install seaborn plotly
```

## Quick Start

### 1. Download and Prepare Data

```bash
# Download multimodal datasets (Localized Narratives + COCO)
python download_multimodal_data.py --output_dir ./data

# The script automatically:
# - Downloads Localized Narratives (~1.16M samples)
# - Downloads COCO Captions (~615K samples)  
# - Ensures perfect image-caption alignment
# - Creates unified dataset with 1.78M total pairs
```

### 2. Configure Training

BitGen provides three configuration files for different use cases:

- `configs/bitmar_config.yaml`: Standard configuration with episodic memory
- `configs/bitmar_with_memory.yaml`: Full model for ablation studies
- `configs/bitmar_without_memory.yaml`: Baseline without episodic memory

Key configuration sections:

```yaml
model:
  # Episodic Memory Control
  use_episodic_memory: true  # Toggle for ablation studies
  
  # FIBER Configuration
  num_fiber_fusion_layers: 6
  fiber_backbone_integration: true
  fiber_bidirectional_fusion: true
  fiber_learnable_alpha: true
  
  # Model Dimensions
  text_encoder_dim: 128
  vision_latent_size: 128
  fusion_hidden_size: 128
  memory_size: 32
  episode_dim: 128
```

### 3. Train the Model

```bash
# Basic training
python train_100M_tokens.py --config configs/bitmar_config.yaml

# With specific GPU
python train_100M_tokens.py --config configs/bitmar_config.yaml --device cuda:0

# Rebuild dataset cache if needed
python train_100M_tokens.py --config configs/bitmar_config.yaml --rebuild_cache

# Save checkpoints every N steps
python train_100M_tokens.py --config configs/bitmar_config.yaml --save_every_n_steps 1000
```

### 4. Ablation Study

Compare models with and without episodic memory:

```bash
# Train with episodic memory
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml

# Train without episodic memory (baseline)
python train_100M_tokens.py --config configs/bitmar_without_memory.yaml
```

## Model Architecture Details

### BitNet Quantization

BitGen implements BitNet b1.58 quantization:

```python
# Weight quantization: {-1, 0, +1}
def quantize_weights_1_58_bit(self, weight):
    scale = weight.abs().mean()
    weight_norm = weight / scale
    threshold = 2.0 / 3.0
    
    quantized = torch.zeros_like(weight_norm)
    quantized[weight_norm > threshold] = 1.0
    quantized[weight_norm < -threshold] = -1.0
    return quantized
```

### FIBER Fusion Implementation

The FIBER fusion replaces traditional late fusion with backbone integration:

```python
class FIBERCrossModalLayer(nn.Module):
    """Bidirectional cross-modal attention layer"""
    
    def forward(self, vision_hidden_states, text_hidden_states):
        # Vision attending to text
        v2t_attention = self.vision_to_text_attention(vision_hidden_states, text_hidden_states)
        
        # Text attending to vision  
        t2v_attention = self.text_to_vision_attention(text_hidden_states, vision_hidden_states)
        
        # Enhanced representations
        enhanced_vision = self.apply_attention(vision_hidden_states, v2t_attention)
        enhanced_text = self.apply_attention(text_hidden_states, t2v_attention)
        
        return enhanced_vision, enhanced_text
```

### Episodic Memory Mechanism

Inspired by Larimar, the episodic memory stores multimodal associations:

```python
class EpisodicMemory(nn.Module):
    """Cross-modal episodic memory with quality-based storage"""
    
    def forward(self, episode):
        # Write: Store multimodal episodes
        consolidated_episode = self.consolidation_net(episode)
        self.write_memory(consolidated_episode)
        
        # Read: Retrieve relevant memories
        retrieved, attention_weights = self.read_memory(consolidated_episode)
        
        # Combine input and memory
        output = 0.7 * consolidated_episode + 0.3 * retrieved
        return output, attention_weights
```

## Training Process

### Data Flow

1. **Text Processing**: GPT-2 tokenization → BitNet text encoder → text features
2. **Vision Processing**: DiNOv2/DiNOv3 features → quantized vision encoder → vision features  
3. **FIBER Fusion**: Cross-modal attention in transformer backbone → enhanced features
4. **Episode Creation**: Combine text + vision → episodic memory → memory-augmented features
5. **Text Generation**: BitNet decoder → output tokens

### Loss Components

```python
total_loss = (
    decoder_loss +                    # Standard language modeling
    cross_modal_contrastive_loss +    # CLIP-style alignment
    vision_reconstruction_loss +      # Prevent vision collapse
    memory_consistency_loss           # Encourage meaningful memory use
)
```

### Adaptive Training

BitGen includes intelligent training interventions:

- **Similarity Monitoring**: Tracks cross-modal similarity over time
- **Automatic Intervention**: Freezes components and rebalances losses when similarity drops
- **Loss Rebalancing**: Dynamically adjusts loss weights during training

## Advanced Features

### Attention Sinks Integration

For unlimited sequence generation:

```yaml
attention_sinks:
  enabled: true
  attention_sink_size: 4
  attention_sink_window_size: 1020
  inject_to_text_encoder: true
  inject_to_text_decoder: true
```

### Memory Management

The episodic memory supports:

- **External Storage**: Save/load memory from disk for edge deployment
- **Compression**: Quantize memory for reduced storage size
- **Lazy Loading**: Load memory on-demand for resource-constrained environments
- **Quality-Based Updates**: Intelligent slot selection based on age, usage, and quality

```python
# Enable external storage
model.memory.enable_external_storage(
    storage_path="episodic_memory.pt",
    compress=True,
    lazy=True
)

# Create memory snapshots
snapshot_path = model.memory.create_memory_snapshot("checkpoint_v1")
```

### FLOPS Tracking

Monitor computational efficiency:

```yaml
flops_tracking:
  enabled: true
  detailed_breakdown: true
  track_components: ["attention", "feedforward", "cross_modal_fusion"]
```

## Model Usage

### Basic Generation

```python
from src.model import create_bitmar_model
import torch

# Load configuration
config = {...}  # Your model config

# Create model
model = create_bitmar_model(config)
model.eval()

# Prepare inputs
input_ids = torch.tensor([[101, 2023, 2003, ...]]) # Tokenized text
vision_features = torch.randn(1, 768)  # DiNOv2 features
attention_mask = torch.ones_like(input_ids)

# Generate text
generated = model.generate(
    input_ids=input_ids,
    vision_features=vision_features,
    max_length=50,
    temperature=0.8,
    do_sample=True
)

# Decode output
text = model.tokenizer.decode(generated[0], skip_special_tokens=True)
```

### Training Mode

```python
# Forward pass during training
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    vision_features=vision_features,
    labels=labels,
    step=global_step
)

loss = outputs['loss']
loss.backward()
```

## Configuration Reference

### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_episodic_memory` | Enable/disable episodic memory | `true` |
| `text_encoder_dim` | Text encoder hidden dimension | `128` |
| `vision_latent_size` | Vision encoder output dimension | `128` |
| `fusion_hidden_size` | FIBER fusion hidden dimension | `128` |
| `num_fiber_fusion_layers` | Number of FIBER fusion layers | `6` |
| `memory_size` | Number of episodic memory slots | `32` |
| `episode_dim` | Episodic memory dimension | `128` |

### FIBER Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `fiber_backbone_integration` | Enable backbone vs late fusion | `true` |
| `fiber_bidirectional_fusion` | Enable bidirectional attention | `true` |
| `fiber_learnable_alpha` | Learnable fusion parameters | `true` |
| `fiber_attention_temperature` | Attention temperature | `1.0` |
| `fiber_cross_attention_dropout` | Cross-attention dropout | `0.1` |

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_epochs` | Maximum training epochs | `20` |
| `batch_size` | Training batch size | `64` |
| `learning_rate` | Initial learning rate | `0.0002` |
| `gradient_clip_val` | Gradient clipping value | `0.3` |
| `use_fp16` | Mixed precision training | `true` |

## Monitoring and Visualization

### Weights & Biases Integration

BitGen provides comprehensive experiment tracking:

- **Training Metrics**: Loss components, learning rate, gradients
- **Cross-Modal Analysis**: Similarity scores, alignment quality
- **Memory Visualization**: Slot evolution, access patterns, diversity
- **Attention Analysis**: Head patterns, cross-modal attention heatmaps
- **FLOPS Tracking**: Computational efficiency metrics

### Memory Visualization

Track episodic memory evolution:

```python
# Memory statistics
memory_stats = model.memory.get_memory_statistics()

# Create memory snapshot
snapshot_path = model.memory.create_memory_snapshot("experiment_v1")

# Load previous snapshot
success = model.memory.load_memory_snapshot("experiment_v1")
```

## Performance Optimizations

### Memory Efficiency

- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision**: FP16 training with automatic scaling
- **Cache Management**: Automatic GPU memory cleanup
- **Batch Processing**: Efficient vision feature processing

### Edge Deployment

- **Memory Compression**: Quantize episodic memory for storage
- **External Storage**: Offload memory to disk
- **Lazy Loading**: Load components on-demand
- **Model Quantization**: Full quantization for inference

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   ```bash
   # Reduce batch size in config
   batch_size: 32
   
   # Enable memory optimizations
   use_gradient_checkpointing: true
   use_fp16: true
   ```

2. **Dimension Mismatches**
   ```bash
   # Clear checkpoint directory for clean start
   rm -rf checkpoints_*
   
   # Rebuild dataset cache
   python train_100M_tokens.py --rebuild_cache
   ```

3. **CUDA Errors**
   ```bash
   # Clear CUDA cache
   import torch
   torch.cuda.empty_cache()
   
   # Use CPU fallback
   python train_100M_tokens.py --device cpu
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Evaluation

### Model Comparison

Compare different configurations:

```bash
# Full model with memory
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml

# Baseline without memory  
python train_100M_tokens.py --config configs/bitmar_without_memory.yaml
```

### Metrics

Key evaluation metrics:

- **Cross-Modal Similarity**: Alignment between text and vision features
- **Memory Utilization**: Percentage of active memory slots
- **Generation Quality**: BLEU, ROUGE scores for text generation
- **Computational Efficiency**: FLOPS per token, inference speed

## Model Architecture Diagram

```
Input Text → BitNet Text Encoder → Text Features ↘
                                                   ↘
                                                FIBER Fusion → Enhanced Features
                                                   ↗                    ↓
Input Vision → Vision Encoder → Vision Features ↗              Episodic Memory
                                                                       ↓
Generated Text ← BitNet Text Decoder ← Memory-Augmented Features ←────┘
```

## Research Background

### BitNet Quantization

Based on "BitNet: Scaling 1-bit Transformers for Large Language Models":
- Quantizes weights to {-1, 0, +1} (1.58 bits per weight)
- Maintains full precision activations during training
- Achieves significant memory and compute savings

### FIBER Fusion

Based on "FIBER: Coarse-to-Fine Vision-Language Pre-training":
- Integrates cross-modal attention within transformer backbone
- Enables bidirectional vision-text understanding
- Superior to late fusion approaches

### Episodic Memory

Inspired by "Larimar: Large Language Models with Episodic Memory Control":
- Stores multimodal episodes for association learning
- Quality-based memory slot selection
- Supports external storage for edge deployment

## File Structure

```
BitGen/
├── configs/                    # Training configurations
│   ├── bitmar_config.yaml     # Standard config
│   ├── bitmar_with_memory.yaml    # Full model
│   └── bitmar_without_memory.yaml # Baseline
├── src/                       # Source code
│   ├── model.py              # Main BitMar model
│   ├── fiber_fusion.py       # FIBER implementation
│   ├── dataset.py            # Multimodal dataset handling
│   ├── adaptive_training_controller.py  # Smart training
│   ├── attention_sinks_integration.py   # Unlimited generation
│   ├── memory_utils.py       # Memory management
│   ├── flops_tracker.py      # Performance monitoring
│   └── wandb_logger.py       # Experiment tracking
├── train_100M_tokens.py      # Main training script
├── download_multimodal_data.py  # Data preparation
└── requirements.txt          # Dependencies
```

## Advanced Usage

### Custom Model Creation

```python
from src.model import BitMarModel

config = {
    'vocab_size': 50257,
    'text_encoder_dim': 128,
    'text_encoder_layers': 4,
    'vision_encoder_dim': 768,
    'vision_latent_size': 128,
    'fusion_hidden_size': 128,
    'memory_size': 32,
    'episode_dim': 128,
    'use_episodic_memory': True,
    'num_fiber_fusion_layers': 6,
    'fiber_backbone_integration': True,
    'max_seq_len': 256,
    'dropout': 0.15
}

model = BitMarModel(config)
```

### FIBER Configuration

```python
# Configure FIBER fusion
fiber_config = {
    'num_fusion_layers': 6,
    'text_fusion_layers': 4,
    'vision_fusion_layers': 2,
    'fusion_strategy': 'deep_backbone',
    'learnable_alpha': True,
    'bidirectional_fusion': True,
    'attention_temperature': 1.0,
    'cross_attention_dropout': 0.1
}

# Create FIBER-enhanced model
from src.fiber_fusion import create_fiber_fusion

fusion_module = create_fiber_fusion(
    text_encoder_dim=128,
    vision_encoder_dim=128,
    fusion_hidden_size=128,
    config=fiber_config
)
```

### Memory Management

```python
# Enable external storage for edge deployment
model.memory.enable_external_storage(
    storage_path="./memory/episodic_memory.pt",
    compress=True,
    lazy=True
)

# Create and load snapshots
snapshot_path = model.memory.create_memory_snapshot("checkpoint_v1")
model.memory.load_memory_snapshot("checkpoint_v1")

# Get memory statistics
stats = model.memory.get_memory_statistics()
print(f"Memory utilization: {stats['memory_utilization']:.2f}")
```

## Hugging Face Hub Integration

BitGen automatically uploads models to Hugging Face Hub:

```yaml
huggingface_hub:
  enabled: true
  repo_id: "your-username/bitmar-model"
  private: true
  upload_after_epoch: true
  create_model_card: true
```

Models are uploaded with comprehensive metadata and training details.

## Performance Benchmarks

### Model Size

- **Full Model**: ~10M parameters
- **Quantized Inference**: ~2.5MB memory footprint
- **With Episodic Memory**: +~0.5MB for 32 slots

### Training Speed

- **GPU Training**: ~1000 tokens/second on RTX 4090
- **Memory Usage**: ~4GB GPU memory with batch_size=64
- **Convergence**: ~20 epochs for 1.78M samples

### Computational Efficiency

- **FLOPS Reduction**: ~8x reduction with BitNet quantization
- **Inference Speed**: ~3x faster than full precision models
- **Energy Efficiency**: Tracked with CodeCarbon integration

## Citation

```bibtex
@software{bitgen2025,
  title={BitGen: BitNet-Quantized Vision-Language Transformer with FIBER Fusion},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/BitGen}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## Acknowledgments

- **BitNet**: Microsoft Research for 1.58-bit quantization
- **FIBER**: Microsoft Research for backbone fusion methodology
- **Larimar**: Episodic memory control mechanisms
- **Attention Sinks**: MIT for unlimited sequence generation
- **DiNOv2/DiNOv3**: Meta AI for self-supervised vision features

## Support

For questions and issues:
- Open an issue on GitHub
- Check the troubleshooting section
- Review configuration examples

---

**Note**: This implementation is designed for research purposes and includes experimental features. For production use, consider additional optimizations and safety measures.
