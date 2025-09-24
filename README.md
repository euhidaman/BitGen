# BitGen: Advanced Tiny Language Model for Embedded Microcontrollers

An advanced tiny language model designed for embedded microcontrollers that integrates:
- **Larimar Episodic Memory**: Core memory architecture for storing and retrieving experiences
- **BitNet 1.58-bit Quantization**: Ultra-efficient quantization for microcontroller deployment
- **FIBER Cross-Modal Fusion**: Vision-language understanding with image-text association
- **Attention Sinks**: Memory-efficient attention mechanism for embedded systems
- **Tiny-R1 Reasoning**: DeepSeek-R1 inspired reasoning capabilities
- **Robot Selection**: Intelligent robot selection based on task requirements

## System Architecture

The BitGen system is specifically designed for resource-constrained embedded microcontrollers with:
- Limited RAM (typically 512KB - 2MB)
- Minimal storage (4-16MB flash)
- Low computational power (ARM Cortex-M series)
- Real-time inference requirements

### Key Features

1. **Embedded-Optimized Components**: All modules are optimized for microcontroller constraints
2. **Adaptive Loss System**: Dynamic loss reweighting based on performance across modalities
3. **COCO Dataset Integration**: Vision-language training on image-caption pairs
4. **Online Learning**: Episodic memory enables continuous adaptation on embedded devices
5. **Integer Arithmetic**: Minimal floating-point operations for embedded efficiency

## Installation

```bash
pip install -r requirements.txt
python download_coco_dataset.py
```

## Usage

```python
from bitgen import BitGenModel

# Initialize model for embedded deployment
model = BitGenModel(
    memory_size=64,  # Episodic memory slots
    quantization_bits=1.58,  # BitNet quantization
    embed_dim=128,  # Compact embeddings
    num_layers=4,  # Shallow architecture for speed
    attention_sinks=4  # Attention sink tokens
)

# Train on COCO dataset
model.train_on_coco("path/to/coco/dataset")

# Deploy to microcontroller
model.export_for_embedded("bitgen_embedded.bin")
```

## Directory Structure

- `src/`: Core BitGen implementation
- `models/`: Model architectures and components
- `data/`: Dataset handling and preprocessing
- `embedded/`: Microcontroller deployment utilities
- `configs/`: Configuration files
- `scripts/`: Training and evaluation scripts
