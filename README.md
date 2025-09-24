# BitGen: Advanced Tiny Language Model for Embedded Systems

An advanced tiny language model that integrates **Larimar Episodic Memory**, **BitNet 1.58-bit Quantization**, **FIBER Cross-Modal Fusion**, **Attention Sinks**, **Tiny-R1 Reasoning**, and **Robot Selection** capabilities.

## 🚀 Features

- **Larimar Episodic Memory**: Core memory architecture for storing and retrieving experiences
- **BitNet 1.58-bit Quantization**: Ultra-efficient quantization for embedded deployment
- **FIBER Cross-Modal Fusion**: Vision-language understanding with image-text association
- **Attention Sinks**: Memory-efficient attention mechanism for long sequences
- **Tiny-R1 Reasoning**: DeepSeek-R1 inspired reasoning capabilities
- **Robot Selection**: Intelligent robot selection based on task requirements
- **Comprehensive Monitoring**: FLOPS tracking, CodeCarbon energy monitoring, performance metrics
- **HuggingFace Integration**: Automatic model pushing after every epoch
- **WandB Tracking**: Real-time metrics logging to 'babylm-ntust' team with visualizations

## 📊 Advanced Metrics & Visualizations

### Episodic Memory Analysis
- Memory utilization and diversity tracking
- Access pattern heatmaps showing which memory slots are used
- Memory similarity matrices revealing relationships between stored experiences

### Attention Heatmaps
- Multi-head attention visualization focusing on important tokens
- Head specialization analysis (local vs global attention patterns)
- Attention sink detection and important token identification

### Reasoning Matrices
- Robot selection confusion matrices showing correct vs incorrect decisions
- Per-robot accuracy tracking that improves over training epochs
- Interactive reasoning dashboards with improvement trends

## 🛠️ Installation

```bash
git clone https://github.com/euhidaman/BitGen.git
cd BitGen
pip install -r requirements.txt
```

## 📥 Setup Data & Authentication

### 1. Download COCO Dataset
```bash
# Set up Kaggle API credentials first
# Place kaggle.json in ~/.kaggle/ with your API key

python bitgen_cli.py download --output_dir data/coco
```

### 2. Setup HuggingFace Hub (for model pushing)
```bash
# Set environment variable
export HF_TOKEN="your_huggingface_token"
# OR login via CLI
huggingface-cli login
```

### 3. Setup WandB (for metrics tracking)
```bash
# Login to WandB
wandb login
# Make sure you have access to 'babylm-ntust' team
```

## 🎓 Training

### Basic Training with All Features
```bash
python bitgen_cli.py train \
  --coco_data data/coco/validated_coco.json \
  --model_size tiny \
  --num_epochs 10 \
  --enable_carbon_tracking \
  --track_flops \
  --push_to_hub \
  --use_wandb
```

### Advanced Training with Custom Settings
```bash
python bitgen_cli.py train \
  --coco_data data/coco/validated_coco.json \
  --robot_data data/robot_selection/robot_tasks.json \
  --model_size tiny \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --num_epochs 15 \
  --enable_carbon_tracking \
  --track_flops \
  --push_to_hub \
  --hf_repo_name "my-bitgen-model" \
  --use_wandb \
  --wandb_entity babylm-ntust \
  --wandb_project bitgen-training \
  --wandb_tags bitgen multimodal robotics
```

**What Training Provides:**
- ✅ **FLOPS Tracking**: Real-time computational complexity monitoring
- ✅ **CodeCarbon Energy Monitoring**: Energy consumption and carbon footprint tracking
- ✅ **HuggingFace Hub Pushing**: Automatic model upload after every epoch
- ✅ **WandB Logging**: Comprehensive metrics to babylm-ntust team
- ✅ **Advanced Visualizations**: Memory heatmaps, attention patterns, reasoning matrices

## 🔮 Inference with Performance Metrics

### Interactive Inference with Real-time Monitoring
```bash
python bitgen_cli.py inference \
  --model_path checkpoints/bitgen_checkpoint_best.pt \
  --interactive
```

### Comprehensive Benchmark
```bash
python bitgen_cli.py inference \
  --model_path checkpoints/bitgen_checkpoint_best.pt \
  --benchmark \
  --num_samples 20 \
  --show_metrics
```

### Single Inference with Metrics
```bash
python bitgen_cli.py inference \
  --model_path checkpoints/bitgen_checkpoint_best.pt
```

**Inference Metrics Provided:**
- 🎯 **Model Response Throughput**: tokens/sec
- ⏱️ **Latency**: ms per token and per response
- 💾 **Memory Footprint**: RAM usage and peak memory
- ⚡ **Power Consumption**: mW power usage
- 🌡️ **Thermal Profile**: CPU temperature monitoring

### Example Inference Output
```
📊 COMPREHENSIVE INFERENCE METRICS:
🎯 PERFORMANCE:
   Model Response Throughput: 3.45 tokens/sec
   Latency per Token: 289.2 ms/token
   Response Time: 1247.8 ms

💾 MEMORY FOOTPRINT:
   Peak RAM Usage: 87.3 MB
   Memory Delta: +12.4 MB

⚡ POWER & ENERGY:
   Power Consumption: 387.2 mW
   Energy Consumed: 12.4 mJ

🌡️ THERMAL PROFILE:
   CPU Temperature: 62.4°C
   Thermal Delta: +2.1°C
```

## 📈 Monitoring & Analysis

### System Monitoring
```bash
python bitgen_cli.py monitor --duration 300 --real_time
```

### Results Analysis
```bash
python bitgen_cli.py analyze --results_dir training_monitoring --generate_report
```

## 🏗️ Model Architecture

- **Embed Dimensions**: 128D (tiny), 256D (small), 64D (nano)
- **Layers**: 4 layers (tiny), 6 layers (small), 2 layers (nano)
- **Episodic Memory**: 64 memory slots with retrieval and update mechanisms
- **Attention Sinks**: 4 sink tokens for efficient long-sequence processing
- **Cross-Modal Fusion**: Text-image understanding with FIBER architecture
- **Robot Selection**: 16 robot types with task-based selection
- **Quantization**: 1.58-bit weights for deployment efficiency

## 🎯 Use Cases

### 1. Multimodal Image Captioning
```python
from src import BitGen

bitgen = BitGen(model_size='tiny')
bitgen.load_checkpoint('checkpoints/best.pt')

# Process image with text
result = bitgen.process_image_and_text('image.jpg', 'Describe the scene')
```

### 2. Robot Task Selection
```python
# Select appropriate robot for task
robot_selection = bitgen.select_robot_for_task('Pick up heavy objects from floor')
print(f"Selected: {robot_selection['selected_robot']} (confidence: {robot_selection['confidence']:.3f})")
```

### 3. Text Generation with Reasoning
```python
# Generate text with reasoning
response = bitgen.generate_text('<reasoning>The robot should move to</reasoning><answer>')
```

## 📊 WandB Dashboard (babylm-ntust team)

Training automatically logs to WandB with:
- **Loss curves** and training metrics
- **Episodic memory heatmaps** showing memory utilization
- **Attention pattern visualizations** highlighting important tokens
- **Robot selection matrices** showing reasoning improvement
- **Performance dashboards** with throughput, latency, power consumption
- **Energy tracking** with FLOPS and carbon footprint analysis

Access your runs at: `https://wandb.ai/babylm-ntust/bitgen-training`

## 🤗 HuggingFace Hub Integration

Models are automatically pushed to HuggingFace Hub after every epoch:
- **Auto-generated model names**: `bitgen-{size}-{timestamp}`
- **Detailed model cards** with training metrics and usage instructions
- **Version tracking** with epoch-specific commits
- **Public sharing** for easy collaboration

## 🔧 Configuration Options

### Model Sizes
- **nano**: 64D embed, 2 layers (ultra-lightweight)
- **tiny**: 128D embed, 4 layers (default)
- **small**: 256D embed, 6 layers (higher capacity)

### CLI Commands
- `download`: Download and prepare COCO dataset
- `train`: Train model with comprehensive monitoring
- `inference`: Run inference with performance metrics
- `evaluate`: Evaluate model capabilities
- `deploy`: Deploy for embedded systems
- `monitor`: System performance monitoring
- `analyze`: Analyze training/inference results

## 📈 Performance Targets

### Training Efficiency
- **FLOPS Tracking**: Real-time computational complexity monitoring
- **Energy Monitoring**: CodeCarbon integration for sustainability
- **Memory Optimization**: Episodic memory utilization tracking

### Inference Performance
- **Throughput**: 2-5 tokens/sec on Raspberry Pi
- **Latency**: 200-500ms per token
- **Memory**: <100MB RAM usage
- **Power**: <500mW consumption

## 🎨 Visualizations Available

1. **Memory Access Heatmaps**: Show episodic memory usage patterns
2. **Attention Focus Maps**: Highlight most important tokens per head
3. **Robot Selection Matrices**: Track reasoning accuracy improvement
4. **Performance Dashboards**: Real-time metrics and trends
5. **Energy Efficiency Charts**: FLOPS per mJ, carbon per token

## 🔍 Troubleshooting

### Common Issues
- **HuggingFace Login**: Ensure `HF_TOKEN` environment variable is set
- **WandB Access**: Verify access to 'babylm-ntust' team
- **Memory Issues**: Reduce batch size or model size for resource-constrained systems
- **Temperature Warnings**: Training automatically pauses if CPU temperature >75°C

### Monitoring
All training runs include automatic monitoring and will generate:
- Comprehensive training reports with FLOPS and energy data
- Advanced metrics visualizations saved locally and to WandB
- Model checkpoints with full state information

## 📝 Example Complete Workflow

```bash
# 1. Download data
python bitgen_cli.py download

# 2. Train with full monitoring
python bitgen_cli.py train \
  --coco_data data/coco/validated_coco.json \
  --enable_carbon_tracking \
  --track_flops \
  --push_to_hub \
  --use_wandb

# 3. Run inference with metrics
python bitgen_cli.py inference \
  --model_path checkpoints/bitgen_checkpoint_best.pt \
  --benchmark \
  --show_metrics

# 4. Analyze results
python bitgen_cli.py analyze \
  --results_dir training_monitoring \
  --generate_report
```

The BitGen system provides state-of-the-art multimodal capabilities with comprehensive monitoring, automatic model sharing, and advanced internal analysis - all optimized for efficient deployment and team collaboration.
