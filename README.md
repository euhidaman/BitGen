# BitGen: BitNet-Quantized Vision-Language Transformer with FIBER Fusion, Robot Reasoning, and Security

BitGen (BitMar) is an advanced multimodal transformer that combines BitNet 1.58-bit quantization, FIBER-style backbone fusion, episodic memory, **structured robot reasoning capabilities**, and **comprehensive security** for efficient vision-language understanding, generation, and robot task planning.

## Key Features

- **BitNet Quantization**: 1.58-bit weight quantization for efficient inference
- **FIBER Backbone Fusion**: Microsoft Research's FIBER implementation for superior cross-modal understanding
- **Episodic Memory**: Larimar-inspired memory mechanism for multimodal associations
- **Robot Reasoning**: deepseek-r1 inspired structured reasoning for robot selection and task planning
- **Security Guard**: LLM Guard integration with input/output validation and memory anomaly detection
- **Attention Sinks**: Support for unlimited sequence generation
- **Adaptive Training**: Intelligent cross-modal similarity monitoring and intervention
- **Unified Training**: Single script for multimodal data + robot reasoning + security monitoring
- **Edge Deployment**: Optimized for deployment with memory compression and external storage

## Architecture Overview

### Model Components

1. **BitNet Text Encoder/Decoder**: Quantized transformer blocks with 1.58-bit weights
2. **Vision Encoder**: Quantized processing of DiNOv2/DiNOv3 features  
3. **FIBER Fusion**: Cross-modal attention fusion in transformer backbone
4. **Episodic Memory**: Multimodal memory slots for association learning
5. **Robot Reasoning Head**: Structured reasoning for robot selection tasks
6. **Security Guard**: Input/output validation and memory anomaly detection
7. **Attention Sinks**: KV-cache optimization for long sequence generation

### FIBER Integration

BitGen implements Microsoft Research's FIBER (Fusion in the Backbone) approach:

- **Coarse-to-Fine Fusion**: Early layers focus on modality-specific processing, later layers enable cross-modal fusion
- **Bidirectional Attention**: Vision-to-text and text-to-vision attention mechanisms
- **Backbone Integration**: Cross-modal fusion within transformer blocks rather than late fusion
- **Learnable Alpha**: Adaptive fusion strength parameters

### Robot Reasoning (deepseek-r1 Inspired)

BitGen integrates structured reasoning capabilities for robot selection tasks:

- **XML Format**: Uses `<reasoning>` and `<answer>` tags following deepseek-r1's approach
- **Multi-Robot Coordination**: Supports complex task decomposition across multiple robots
- **Structured Analysis**: Task analysis, environment assessment, and capability matching
- **Reward-Based Training**: Multiple reward functions for reasoning quality
- **5 Robot Types**: Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs

### Security Features

BitGen includes comprehensive security using LLM Guard:

- **Input Validation**: Prompt injection detection, toxicity filtering, secrets detection
- **Output Validation**: Bias detection, malicious URL detection, relevance checking
- **Memory Anomaly Detection**: Statistical anomaly detection in episodic memory patterns
- **Rate Limiting**: Request and token-based rate limiting
- **Security Logging**: Comprehensive security event logging and reporting

## Installation

### Prerequisites

```bash
# Python 3.8+ required
pip install torch torchvision transformers
pip install huggingface_hub wandb pyyaml tqdm
pip install llm-guard pyod  # Security and anomaly detection
pip install codecarbon matplotlib seaborn pandas psutil
```

### Quick Install

```bash
git clone https://github.com/euhidaman/BitGen.git
cd BitGen
pip install -r requirements.txt
```

## Dataset Setup

### Option 1: Download COCO Dataset Only

```bash
python download_multimodal_data.py --dataset coco --data_dir ./data
```

### Option 2: Download Localized Narratives Only

```bash
python download_multimodal_data.py --dataset localized_narratives --data_dir ./data
```

### Option 3: Download Both Datasets (Recommended)

```bash
python download_multimodal_data.py --dataset both --data_dir ./data
```

This will download:
- **COCO Captions**: ~615K high-quality image-caption pairs
- **Localized Narratives**: ~1.16M dense paragraph descriptions
- **Total**: 1.78M perfectly aligned image-caption pairs

## Training

### Unified Training Script

BitGen now uses a **single unified training script** that handles:
- Multimodal training (Localized Narratives + COCO)
- Robot reasoning grounding
- Security monitoring
- Memory anomaly detection

### Step-by-Step Training Instructions

#### Step 1: Prepare Data
```bash
# Download datasets (choose one of the options above)
python download_multimodal_data.py --dataset both --data_dir ./data

# Ensure robot selection data is available
# Data should be in: D:/BabyLM/robot_selection_data/data
```

#### Step 2: Choose Configuration

**For Ablation Study WITH Memory:**
```bash
python train_unified.py --config configs/bitmar_with_memory.yaml
```

**For Ablation Study WITHOUT Memory:**
```bash
python train_unified.py --config configs/bitmar_without_memory.yaml
```

#### Step 3: Advanced Training Options

**With GPU Selection:**
```bash
python train_unified.py --config configs/bitmar_with_memory.yaml --device cuda:0
```

**With Cache Rebuilding:**
```bash
python train_unified.py --config configs/bitmar_with_memory.yaml --rebuild_cache
```

**Legacy Training (Separate Scripts):**
```bash
# Multimodal training only
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml

# Robot reasoning only
python train_robot_reasoning.py --config configs/bitmar_robot_reasoning.yaml
```

### Training Phases

The unified training follows this approach:

1. **Phase 1 (Epochs 1-15)**: Multimodal Foundation
   - Trains on Localized Narratives + COCO
   - Establishes cross-modal understanding
   - Builds episodic memory patterns (if enabled)

2. **Phase 2 (Epochs 16-20)**: Robot Reasoning Grounding
   - Grounds reasoning on robot selection tasks
   - Integrates structured XML reasoning format
   - Validates robot selection capabilities

3. **Security Monitoring**: Throughout All Phases
   - Input/output validation using LLM Guard
   - Memory anomaly detection (if episodic memory enabled)
   - Rate limiting and security event logging

### Robot Reasoning Integration

After multimodal training, the model learns structured reasoning for robot selection:

```python
# Example robot reasoning output
task = "Inspect underwater pipelines for leaks"

# Model generates:
"""
<reasoning>
Task Analysis: Inspect underwater pipelines for leaks
Environment: Underwater - requires waterproof robot with underwater navigation
Required capabilities: Inspection sensors, underwater mobility, leak detection
The Underwater Robot is specialized for marine environments and has waterproof design
suitable for underwater pipeline inspection tasks.
</reasoning>
<answer>
Selected robot(s): Underwater Robot
</answer>
"""
```

## Security Features

### Input Validation

- **Prompt Injection Detection**: Detects attempts to manipulate model behavior
- **Toxicity Filtering**: Blocks toxic or harmful content
- **Secrets Detection**: Prevents exposure of credentials or sensitive information
- **Length Limits**: Enforces maximum input lengths for safety

### Output Validation

- **Bias Detection**: Identifies potentially biased model outputs
- **Malicious URL Detection**: Blocks generation of harmful links
- **Relevance Checking**: Ensures outputs are relevant to inputs
- **Robot Selection Validation**: Validates robot selections are from valid set

### Memory Anomaly Detection

- **Statistical Analysis**: Uses Isolation Forest, Local Outlier Factor, and One-Class SVM
- **Pattern Monitoring**: Tracks episodic memory patterns for anomalies
- **Baseline Learning**: Establishes normal memory behavior during early training
- **Real-time Detection**: Monitors memory during training and inference

### Security Monitoring

```python
# Secure generation with validation
result = model.secure_generate(
    prompt="Inspect building exterior",
    robot_task="Check for structural damage"
)

if result['success']:
    print(result['generated_text'])
    print(f"Security validated: {result['input_validation']}")
    print(f"Memory anomalies: {result['memory_security']}")
else:
    print(f"Security blocked: {result['error']}")
```

## Model Configurations

### Ablation Study Configurations

**WITH Memory (`bitmar_with_memory.yaml`):**
- Episodic memory: 32 slots, 128 dimensions
- Robot reasoning: Enabled
- Security: Full validation + memory anomaly detection
- Repository: `euhidaman/bitmar-with-memory-ablation`

**WITHOUT Memory (`bitmar_without_memory.yaml`):**
- Episodic memory: Disabled (direct fusion baseline)
- Robot reasoning: Enabled
- Security: Input/output validation only
- Repository: `euhidaman/bitmar-without-memory-ablation`

### Key Differences in Configurations

| Feature | With Memory | Without Memory |
|---------|-------------|----------------|
| Episodic Memory | ✅ 32 slots | ❌ Disabled |
| Memory Anomaly Detection | ✅ Enabled | ❌ Disabled |
| Cross-modal Fusion | Enhanced with memory | Direct fusion |
| Model Size | ~50MB | ~45MB |
| Target Similarity | 0.75 | 0.70 |

## Usage Examples

### Basic Multimodal Generation

```python
from src.model import create_bitmar_model
import torch

# Load model
config = yaml.safe_load(open('configs/bitmar_with_memory.yaml'))
model = create_bitmar_model(config['model'])

# Generate with vision and text
vision_features = torch.randn(1, 768)  # DiNOv3 features
text_input = "A beautiful sunset over the ocean"

output = model.generate(
    text_input,
    vision_features=vision_features,
    max_length=100
)
```

### Robot Reasoning Generation

```python
# Robot task reasoning
task = "Navigate through a crowded warehouse to deliver packages"

reasoning_result = model.generate_robot_reasoning(
    task=task,
    context="Warehouse environment with moving obstacles"
)

print(reasoning_result['reasoning'])
print(reasoning_result['selected_robots'])
```

### Secure Generation

```python
# Generation with security validation
secure_result = model.secure_generate(
    prompt="Plan robot deployment for emergency response",
    robot_task="Search and rescue in collapsed building"
)

if secure_result['success']:
    print(secure_result['generated_text'])
    print(f"Security checks passed: {secure_result['input_validation']}")
else:
    print(f"Security blocked: {secure_result['error']}")
```

## Monitoring and Visualization

### Weights & Biases Integration

BitGen automatically logs to Weights & Biases:

- **Training Metrics**: Loss, similarity, token usage
- **Attention Analysis**: Head-wise attention patterns
- **Memory Visualization**: Memory slot evolution and usage
- **Robot Reasoning**: Accuracy and format validation
- **Security Events**: Blocked inputs/outputs and anomalies
- **FLOPS Tracking**: Computational efficiency metrics

### Security Dashboard

Monitor security events in real-time:

```python
# Get security statistics
stats = model._security_guard.get_security_statistics()
print(f"Blocked inputs: {stats['blocked_inputs']}")
print(f"Memory anomalies: {stats['memory_anomalies']}")

# Export security report
report_path = model._security_guard.export_security_report()
```

## File Structure

```
BitGen/
├── train_unified.py              # 🆕 Unified training script (recommended)
├── train_100M_tokens.py          # Legacy multimodal training
├── train_robot_reasoning.py      # Legacy robot reasoning training
├── configs/
│   ├── bitmar_with_memory.yaml    # Ablation: WITH episodic memory + robot reasoning + security
│   └── bitmar_without_memory.yaml # Ablation: WITHOUT episodic memory + robot reasoning + security
├── src/
│   ├── model.py                   # BitMar model with robot reasoning
│   ├── dataset.py                 # Multimodal dataset handling
│   ├── robot_reasoning_dataset.py # Robot reasoning dataset integration
│   ├── robot_reasoning.py         # Robot reasoning implementation
│   ├── security_guard.py          # 🆕 Comprehensive security system
│   ├── fiber_fusion.py            # FIBER backbone fusion
│   ├── memory_*.py                # Episodic memory components
│   └── attention_sinks_integration.py # Attention sinks support
└── requirements.txt               # Updated with security dependencies
```

## Performance & Efficiency

### Model Scale
- **Parameters**: ~50M (with memory) / ~45M (without memory)
- **Quantization**: 1.58-bit BitNet weights
- **Memory**: 32 episodic memory slots (configurable)
- **Context Length**: Unlimited with attention sinks

### Training Efficiency
- **Dataset**: 1.78M image-caption pairs + robot selection data
- **Training Time**: ~24-48 hours on single RTX 4090
- **Memory Usage**: ~8GB VRAM with gradient checkpointing
- **Carbon Tracking**: Automatic CO2 footprint monitoring

### Security Performance
- **Input Validation**: <1ms per request
- **Memory Anomaly Detection**: Real-time during training
- **Rate Limiting**: 60 requests/minute, 10K tokens/minute
- **Security Event Logging**: Comprehensive audit trail

## Deployment

### Hugging Face Hub

Models are automatically uploaded to Hugging Face Hub:

- **With Memory**: `euhidaman/bitmar-with-memory-ablation`
- **Without Memory**: `euhidaman/bitmar-without-memory-ablation`

### Edge Deployment

```python
# Export for edge deployment
from src.memory_utils import MemoryManager

memory_manager = MemoryManager(model)
edge_package = memory_manager.create_edge_deployment_package()
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Dataset Download Fails**: Check internet connection and retry
3. **Security Dependencies Missing**: Install with `pip install llm-guard pyod`
4. **Robot Data Missing**: Ensure robot selection data is in `D:/BabyLM/robot_selection_data/data`

### Debugging

```bash
# Test security components
python -c "from src.security_guard import test_security_components; test_security_components()"

# Validate configuration
python -c "import yaml; print('Config valid:', yaml.safe_load(open('configs/bitmar_with_memory.yaml')))"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with security considerations
4. Add tests for new security features
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@misc{bitgen2025,
  title={BitGen: BitNet-Quantized Vision-Language Transformer with FIBER Fusion, Robot Reasoning, and Security},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/euhidaman/BitGen}}
}
```

## Acknowledgments

- **BitNet**: Microsoft Research's 1.58-bit quantization
- **FIBER**: Microsoft Research's backbone fusion
- **Larimar**: Episodic memory architecture inspiration
- **deepseek-r1**: Structured reasoning approach
- **LLM Guard**: Security and safety framework
- **Attention Sinks**: MIT's efficient attention mechanism
