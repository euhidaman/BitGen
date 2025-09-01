# BitGen: BitNet-Quantized Vision-Language Transformer with FIBER Fusion and Robot Reasoning

BitGen (BitMar) is an advanced multimodal transformer that combines BitNet 1.58-bit quantization, FIBER-style backbone fusion, episodic memory, and **structured robot reasoning capabilities** for efficient vision-language understanding, generation, and robot task planning.

## Key Features

- **BitNet Quantization**: 1.58-bit weight quantization for efficient inference
- **FIBER Backbone Fusion**: Microsoft Research's FIBER implementation for superior cross-modal understanding
- **Episodic Memory**: Larimar-inspired memory mechanism for multimodal associations
- **Robot Reasoning**: deepseek-r1 inspired structured reasoning for robot selection and task planning
- **Attention Sinks**: Support for unlimited sequence generation
- **Adaptive Training**: Intelligent cross-modal similarity monitoring and intervention
- **Hybrid Training**: Unified multimodal + robot reasoning training pipeline
- **Edge Deployment**: Optimized for deployment with memory compression and external storage

## Architecture Overview

### Model Components

1. **BitNet Text Encoder/Decoder**: Quantized transformer blocks with 1.58-bit weights
2. **Vision Encoder**: Quantized processing of DiNOv2/DiNOv3 features  
3. **FIBER Fusion**: Cross-modal attention fusion in transformer backbone
4. **Episodic Memory**: Multimodal memory slots for association learning
5. **Robot Reasoning Head**: Structured reasoning for robot selection tasks
6. **Attention Sinks**: KV-cache optimization for long sequence generation

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

## Quick Start - Unified Training Pipeline

### Step 1: Download and Prepare Data

You can choose to download individual datasets or both datasets depending on your needs:

#### Option A: Download Both Datasets (Recommended - Full Training)
```bash
# Download both Localized Narratives + COCO (1.78M total pairs)
python download_multimodal_data.py --both --output_dir ./data

# This downloads:
# - Localized Narratives: ~1.16M dense paragraph descriptions
# - COCO Captions: ~615K high-quality human annotations
# - Total: 1.78M perfectly aligned image-caption pairs
```

#### Option B: Download Only Localized Narratives (Smaller Dataset)
```bash
# Download only Localized Narratives (~1.16M samples)
python download_multimodal_data.py --localized-narratives-only --output_dir ./data

# Best for:
# - Limited storage/bandwidth
# - Dense paragraph-style captions
# - Detailed multimodal descriptions
```

#### Option C: Download Only COCO (Smallest Dataset)
```bash
# Download only COCO Captions (~615K samples) 
python download_multimodal_data.py --coco-only --output_dir ./data

# Best for:
# - Quick experiments
# - High-quality human annotations
# - Standard computer vision tasks
```

**Ensure you have the robot selection data:** The robot reasoning data should be located at `D:\BabyLM\robot_selection_data\data\` with the following structure:
```
robot_selection_data/
├── data/
│   ├── Single-Robot-Selection/
│   │   ├── single_robot_selection_dataset.json
│   │   └── single_robot_selection_dataset.csv
│   └── Multi-Robot-Selection/
│       ├── multi_robot_selection_dataset.json
│       └── multi_robot_selection_dataset.csv
```

### Step 2: Choose Your Training Configuration

BitGen provides **TWO** configurations for ablation study comparison:

#### Option A: Train WITH Episodic Memory (Full Model)
```bash
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml --device cuda:0
```

#### Option B: Train WITHOUT Episodic Memory (Baseline)
```bash
python train_100M_tokens.py --config configs/bitmar_without_memory.yaml --device cuda:0
```

### Step 3: Understanding the Hybrid Training Process

The unified training script (`train_100M_tokens.py`) automatically handles **two-phase training**:

#### Phase 1: Multimodal Training (Epochs 1-15)
- Trains on **Localized Narratives + COCO** data (1.78M image-caption pairs)
- Builds strong multimodal representations
- Robot reasoning temporarily disabled during this phase
- Focuses on vision-language alignment and cross-modal understanding

#### Phase 2: Robot Reasoning Grounding (Epochs 16-20)
- Switches to **robot selection data** from `robot_selection_data`
- Grounds learned representations on robot reasoning tasks
- Uses deepseek-r1 style XML format reasoning
- Tests robot reasoning capabilities every 2 epochs

### Step 4: Monitor Training Progress

#### Weights & Biases Integration
Both configurations automatically log to separate W&B projects:
- **With Memory**: `bitgen-ablation-with-memory`
- **Without Memory**: `bitgen-ablation-without-memory`

#### Hugging Face Hub Integration
Models are automatically uploaded to separate repositories:
- **With Memory**: `euhidaman/bitmar-with-memory-ablation`
- **Without Memory**: `euhidaman/bitmar-without-memory-ablation`

### Step 5: Evaluate Robot Reasoning (Optional)

You can test robot reasoning capabilities during or after training:

```python
# Example robot reasoning test
from src.model import create_bitmar_model
from src.robot_reasoning import create_robot_reasoning_integration

# Load trained model
model = create_bitmar_model(config)
# Add robot reasoning capabilities
robot_integration = create_robot_reasoning_integration(model)

# Test reasoning
task = "Inspect underwater pipes for structural damage"
result = robot_integration.generate_robot_reasoning(task)

print("Reasoning:", result['reasoning'])
print("Selected Robot(s):", result['selected_robots'])
print("Format Valid:", result['format_valid'])
```

## Configuration Details

### Robot Reasoning Configuration

Both YAML configs now include robot reasoning settings:

```yaml
model:
  # Robot Reasoning Configuration (deepseek-r1 style)
  enable_robot_reasoning: true
  robot_data_dir: "D:/BabyLM/robot_selection_data/data"
  robot_reasoning_loss_weight: 0.2
  reasoning_format_validation: true
  enable_multi_robot_coordination: true

data:
  # Hybrid Training Configuration
  include_robot_data: true
  robot_data_ratio: 0.3  # 30% robot data, 70% multimodal
  hybrid_training: true
  multimodal_epochs: 15  # Phase 1: Multimodal training
  reasoning_epochs: 5    # Phase 2: Robot reasoning grounding
```

### Ablation Study Configurations

**With Memory (`bitmar_with_memory.yaml`):**
```yaml
model:
  use_episodic_memory: true
  memory_size: 32
  episode_dim: 128
  memory_alpha: 0.2

output:
  checkpoint_dir: "checkpoints_with_memory"
  
wandb:
  project: "bitgen-ablation-with-memory"
```

**Without Memory (`bitmar_without_memory.yaml`):**
```yaml
model:
  use_episodic_memory: false
  # Memory-related settings commented out

output:
  checkpoint_dir: "checkpoints_without_memory"
  
wandb:
  project: "bitgen-ablation-without-memory"
```

## Advanced Usage

### Custom Training Options

```bash
# Rebuild dataset cache
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml --rebuild_cache

# Save checkpoints every N steps
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml --save_every_n_steps 1000

# Specify GPU device
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml --device cuda:1
```

### Robot Reasoning Format

BitGen follows deepseek-r1's XML-based reasoning format:

```xml
<reasoning>
Task Analysis: Inspect underwater pipes for structural damage
Environment Assessment: Underwater environment requires waterproof robot with underwater navigation capabilities
Required Capabilities: inspection sensors, underwater navigation, marine operations
Robot Selection Rationale: Underwater Robot provides specialized underwater navigation, marine inspection capabilities, and waterproof design
Conclusion: Underwater Robot is the optimal choice for this underwater inspection task
</reasoning>
<answer>
Selected robot(s): Underwater Robot
</answer>
```

### Available Robot Types

The system can reason about and select from 5 specialized robots:

1. **Drone**: Aerial operations, surveillance, elevated access
2. **Underwater Robot**: Marine operations, underwater navigation, seabed inspection
3. **Humanoid**: Complex manipulation, human interaction, tool use
4. **Robot with Wheels**: Fast transport, flat surfaces, heavy payloads
5. **Robot with Legs**: Rough terrain, stability, uneven surfaces

## Training Data Sources

### Multimodal Data (Phase 1)
- **Localized Narratives**: ~1.16M dense paragraph descriptions
- **COCO Captions**: ~615K high-quality human annotations
- **Total**: 1.78M perfectly aligned image-caption pairs

### Robot Reasoning Data (Phase 2)
- **Single Robot Selection**: Task → Single robot selection with reasoning
- **Multi Robot Selection**: Complex task → Multiple robot coordination
- **Format**: deepseek-r1 style XML with structured reasoning

## Ablation Study Results

Compare model performance with and without episodic memory:

| Metric                   | With Memory | Without Memory |
| ------------------------ | ----------- | -------------- |
| Cross-modal Similarity   | Higher      | Baseline       |
| Robot Reasoning Accuracy | Enhanced    | Standard       |
| Memory Efficiency        | Optimized   | N/A            |
| Parameter Count          | ~50MB       | ~45MB          |

## Monitoring and Visualization

### Weights & Biases Dashboards

The training automatically logs comprehensive metrics to W&B:

**Core Metrics:**
- Training loss and learning rate
- Cross-modal similarity scores
- Token processing statistics
- FLOPS and computational efficiency

**Robot Reasoning Metrics:**
- Robot selection accuracy
- Reasoning format adherence
- XML structure quality scores
- Multi-robot coordination success

**Memory Metrics (With Memory Only):**
- Memory slot utilization
- Memory diversity scores
- Cross-modal memory specialization
- Memory access patterns

### Attention Analysis

```python
# Attention patterns are automatically saved to:
# - attention_with_memory/ (with memory config)
# - attention_without_memory/ (without memory config)

# View attention heatmaps and cross-modal alignment patterns
```

## Model Outputs and Usage

### Text Generation with Vision

```python
from transformers import AutoTokenizer
from src.model import create_bitmar_model

# Load model
config = {...}  # Your training config
model = create_bitmar_model(config)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Generate with vision + text
input_text = "This image shows"
vision_features = torch.randn(1, 768)  # DiNOv3 features

output = model.generate(
    input_text=input_text,
    vision_features=vision_features,
    max_length=256,
    temperature=0.7
)
```

### Robot Reasoning Generation

```python
# After training with robot reasoning enabled
task = "Survey agricultural fields for crop health assessment"
result = model.robot_reasoning_integration.generate_robot_reasoning(task)

print("🤖 Robot Reasoning:")
print(f"Task: {result['task']}")
print(f"Reasoning: {result['reasoning']}")
print(f"Selected Robot(s): {result['selected_robots']}")
print(f"Format Valid: {result['format_valid']}")
```

## Advanced Features

### Attention Sinks for Unlimited Generation

```yaml
attention_sinks:
  enabled: true
  attention_sink_size: 4
  attention_sink_window_size: 1020
  inject_to_text_encoder: true
  inject_to_text_decoder: true
```

This enables:
- **Unlimited sequence length**: No context window limitations
- **Efficient KV-cache**: Memory-optimized attention computation
- **Fluent long-form generation**: Maintains coherence across long sequences

### Episodic Memory Configuration

```yaml
model:
  memory_size: 32  # Number of memory slots
  episode_dim: 128  # Memory slot dimensionality
  memory_alpha: 0.2  # Memory update rate
  direct_writing: true  # Enable direct memory writing
  memory_compression: true  # Compress memory for efficiency
```

### FIBER Fusion Settings

```yaml
model:
  num_fiber_fusion_layers: 6  # Layers with cross-modal fusion
  fiber_backbone_integration: true  # Deep backbone fusion
  fiber_bidirectional_fusion: true  # Vision ↔ Text attention
  fiber_learnable_alpha: true  # Adaptive fusion strength
```

## Troubleshooting

### Common Issues

**GPU Memory Issues:**
```bash
# Reduce batch size in config
batch_size: 32  # Instead of 64

# Enable gradient checkpointing
use_gradient_checkpointing: true
```

**Data Loading Issues:**
```bash
# Rebuild dataset cache
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml --rebuild_cache

# Reduce number of workers
num_workers: 2  # Instead of 6
```

**Robot Data Not Found:**
```bash
# Ensure robot data is properly located at:
D:\BabyLM\robot_selection_data\data\Single-Robot-Selection\
D:\BabyLM\robot_selection_data\data\Multi-Robot-Selection\
```

### Validation Steps

1. **Verify Data Integrity:**
   ```bash
   # Check if datasets are properly aligned
   python -c "from src.dataset import create_data_module; dm = create_data_module({'dataset_dir': './data'}); dm.setup(); print('✅ Data loaded successfully')"
   ```

2. **Test Robot Reasoning:**
   ```bash
   # Test robot reasoning integration
   python -c "from src.robot_reasoning import create_robot_reasoning_integration; print('✅ Robot reasoning available')"
   ```

3. **GPU Memory Test:**
   ```bash
   # Test GPU functionality
   python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
   ```

## Example Training Session

Here's what a complete training session looks like:

### 1. Start Training with Memory
```bash
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml --device cuda:0
```

### 2. Expected Training Flow
```
🚀 Starting BitMar unified training with robot reasoning grounding...
🔄 Hybrid training enabled - will train in phases:
   Phase 1: Multimodal training (epochs 1-15)
   Phase 2: Robot reasoning grounding (epochs 16-20)

🖼️ Starting Phase 1: Multimodal Training
Multimodal epoch 1/15
  • Loss: 2.4567
  • Cross-modal similarity: 0.6543
  • Tokens in epoch: 45,678,901
  • Total tokens processed: 45,678,901

🤖 Switching to Phase 2: Robot Reasoning Grounding
Robot reasoning grounding epoch 16/20
  • Loss: 1.8765
  • Robot reasoning batches: 320/1000
  • Robot reasoning accuracy: 0.7654
  • Reasoning format accuracy: 0.8901

🧪 Testing robot reasoning with 5 examples...
🤖 Test 1: Inspect a high-rise building's exterior for damage
   Reasoning: Task Analysis: Inspect a high-rise building's exterior...
   Selected: Drone
   Format valid: True

✅ Training completed successfully!
```

### 3. Compare Results
After training both configurations, compare:
- Cross-modal similarity scores
- Robot reasoning accuracy
- Memory efficiency (with memory config only)
- Model size and inference speed

## Performance Expectations

### Training Time
- **Single GPU (RTX 4090)**: ~12-15 hours for full training
- **Phase 1 (Multimodal)**: ~10 hours (15 epochs)
- **Phase 2 (Robot Reasoning)**: ~2-3 hours (5 epochs)

### Model Performance
- **Cross-modal Similarity**: >0.75 (with memory), >0.70 (without memory)
- **Robot Reasoning Accuracy**: >0.80 for single robot tasks
- **Multi-robot Coordination**: >0.75 for complex task decomposition
- **Model Size**: ~50MB (with memory), ~45MB (without memory)

### Hardware Requirements
- **Minimum**: RTX 3060 (12GB VRAM)
- **Recommended**: RTX 4090 (24GB VRAM)
- **CPU**: 8+ cores for efficient data loading
- **RAM**: 32GB+ recommended for large datasets

## Output Files and Artifacts

### Checkpoints
```
checkpoints_with_memory/          # With memory training
├── latest_checkpoint.pt
├── checkpoint_epoch_19_step_*.pt
└── memory_exports/              # Edge deployment packages

checkpoints_without_memory/       # Without memory training
├── latest_checkpoint.pt
└── checkpoint_epoch_19_step_*.pt
```

### Logs and Analysis
```
logs_with_memory/                 # Training logs with memory
attention_with_memory/            # Attention visualizations
results_with_memory/              # Robot reasoning test results

logs_without_memory/              # Training logs without memory
attention_without_memory/         # Attention visualizations
results_without_memory/           # Robot reasoning test results
```

### Robot Reasoning Test Results
```json
{
  "task": "Inspect underwater pipes for leaks",
  "reasoning": "Task Analysis: Inspect underwater pipes...",
  "selected_robots": "Underwater Robot",
  "format_valid": true,
  "xml_structure_score": 0.875
}
```

## Key Configuration Sections

### Robot Reasoning Settings
```yaml
model:
  enable_robot_reasoning: true
  robot_data_dir: "D:/BabyLM/robot_selection_data/data"
  robot_reasoning_loss_weight: 0.2
  reasoning_format_validation: true
  enable_multi_robot_coordination: true

data:
  include_robot_data: true
  robot_data_ratio: 0.3
  hybrid_training: true
  multimodal_epochs: 15
  reasoning_epochs: 5
```

### Memory Control (Ablation Study)
```yaml
# WITH MEMORY
model:
  use_episodic_memory: true
  memory_size: 32
  episode_dim: 128

# WITHOUT MEMORY  
model:
  use_episodic_memory: false
  # memory settings commented out
```
