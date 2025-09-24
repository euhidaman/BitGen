# BitGen: Advanced Tiny Language Model for Embedded Systems

An advanced tiny language model that integrates **Larimar Episodic Memory**, **BitNet 1.58-bit Quantization**, **FIBER Cross-Modal Fusion**, **Attention Sinks**, **Tiny-R1 Reasoning**, and **Robot Selection** capabilities.

## 🏗️ BitGen Architecture Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           BitGen Complete Architecture Flow                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘

INPUT LAYER:
┌─────────────┐    ┌─────────────┐    ┌──────────────────────────────────────────┐
│ Text Input  │    │ Image Input │    │         Monitoring & Integration          │
│ (Token IDs) │    │ (RGB Tensor)│    │ • WandB (babylm-ntust team)             │
└─────────────┘    └─────────────┘    │ • HuggingFace Hub (auto-push)           │
       │                   │          │ • FLOPS Tracking                        │
       ▼                   ▼          │ • CodeCarbon Energy Monitoring          │
                                      └──────────────────────────────────────────┘

EMBEDDING LAYER:
┌─────────────┐    ┌─────────────┐    ┌──────────────────────────────────────────┐
│Token Embed  │    │Vision Encode│    │         BitNet 1.58-bit Quantization     │
│+ Positional │    │(DinoV2-like)│    │ • Weights: {-1, 0, +1}                  │
│Encoding     │    │14x14 Patches│    │ • Activations: 8-bit                    │
└─────────────┘    └─────────────┘    │ • 4x Compression Ratio                  │
       │                   │          │ • Integer Arithmetic for Edge           │
       ▼                   ▼          └──────────────────────────────────────────┘
                                                           │
                                                           ▼

CORE PROCESSING LAYERS:
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          Larimar Episodic Memory (Key Component)                        │
│ ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────────────────────────────┐  │
│ │Memory Keys  │  │Memory Values│  │              Edge Advantages:                     │  │
│ │(64 slots)   │  │(64 slots)   │  │ ⚡ Fast Fact Editing (no retraining) ~1-5ms      │  │
│ └─────────────┘  └─────────────┘  │ 🗑️ Selective Forgetting of outdated info       │  │
│        │               │          │ 📈 High Accuracy on Updated Knowledge           │  │
│        ▼               ▼          │ 🚀 Local Access (no cloud dependency)           │  │
│ ┌─────────────────────────────────┐ │ 💾 Latent Information Storage                  │  │
│ │    Similarity Computation       │ └──────────────────────────────────────────────────┘  │
│ │    & Attention Weights          │                                                      │
│ └─────────────────────────────────┘                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            FIBER Cross-Modal Fusion                                     │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────────────────────┐  │
│ │Text Features│◄─┤Text-Vision  │  │                Fusion Process:                  │  │
│ │             │  │ Attention   │  │ 1. Text-to-Vision Cross Attention              │  │
│ │             │  │             │  │ 2. Vision-to-Text Cross Attention              │  │
│ │Vision Feats │◄─┤Vision-Text  │  │ 3. Multimodal Representation Creation          │  │
│ │             │  │ Attention   │  │ 4. Joint Feature Space Mapping                 │  │
│ └─────────────┘  └─────────────┘  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Layer Attention with Sinks                                    │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────────────────────┐  │
│ │Attention    │  │ Sink Tokens │  │             Attention Features:                 │  │
│ │Head 1       │  │   (4 slots) │  │ • 4 Attention Sink Tokens                      │  │
│ │             │  │             │  │ • 128 Token Sliding Window                     │  │
│ │Attention    │  │Recent Window│  │ • Memory-Efficient Long Sequences              │  │
│ │Head 2-8     │  │ (128 tokens)│  │ • Multi-Head Specialization Analysis           │  │
│ │             │  │             │  │ • Important Token Identification               │  │
│ └─────────────┘  └─────────────┘  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         Tiny-R1 Reasoning Module                                       │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────────────────────┐  │
│ │Reasoning    │  │  LSTM Step  │  │            Reasoning Process:                   │  │
│ │Encoder      │  │ Processor   │  │ 1. Encode input to reasoning space             │  │
│ │             │  │             │  │ 2. Multi-step LSTM processing                  │  │
│ │Gate         │  │ Reasoning   │  │ 3. Gate mechanism (continue/stop)              │  │
│ │Mechanism    │  │ Decoder     │  │ 4. Aggregate reasoning steps                   │  │
│ │             │  │             │  │ 5. Decode back to embedding space              │  │
│ └─────────────┘  └─────────────┘  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           Robot Selection System                                        │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────────────────────┐  │
│ │Task Encoder │  │Robot        │  │           Selection Process:                    │  │
│ │             │  │Embeddings   │  │ 1. Encode task representation                  │  │
│ │Similarity   │  │(16 robots)  │  │ 2. Compare with robot embeddings              │  │
│ │Computation  │  │             │  │ 3. Calculate selection confidence              │  │
│ │             │  │Selection    │  │ 4. Generate confusion matrices                 │  │
│ │Confidence   │  │Network      │  │ 5. Track accuracy improvement                  │  │
│ └─────────────┘  └─────────────┘  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼

OUTPUT LAYER:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐
│Layer        │  │ Output      │  │ Generated   │  │     Performance Metrics:        │
│Normalization│  │Projection   │  │    Text     │  │ • Throughput: 2-5 tokens/sec   │
│             │  │(Vocab Size) │  │  (Logits)   │  │ • Latency: 200-500ms/token     │
│             │  │             │  │             │  │ • Memory: <100MB RAM           │
└─────────────┘  └─────────────┘  └─────────────┘  │ • Power: <500mW consumption    │
                                                    │ • Temperature: <70°C           │
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  └─────────────────────────────────┘
│Attention    │  │ Robot       │  │  Robot      │
│Cache Update │  │Selection    │  │Probabilities│
│(Next Token) │  │(Confidence) │  │ (16 types)  │
└─────────────┘  └─────────────┘  └─────────────┘
```

## 🔄 Data Flow Process

### **1. Input Processing**
```
Text Input → Tokenization → Token IDs [batch_size, seq_len]
Image Input → Patch Extraction → RGB Patches [batch_size, 3, 14, 14]
```

### **2. Embedding & Quantization**
```
Token IDs → Token Embedding + Positional Embedding [batch_size, seq_len, embed_dim]
RGB Patches → Vision Encoder → Vision Features [batch_size, 1, embed_dim]
All Weights → BitNet Quantization → {-1, 0, +1} ternary values
```

### **3. Episodic Memory Integration** 🧠
```
Input Embeddings → Key/Value Projection → Memory Query
Memory Keys (64 slots) → Similarity Computation → Attention Weights
Attention Weights × Memory Values → Retrieved Memories
Input + Retrieved Memories → Enhanced Representation

Edge Operations:
• Fast Fact Edit: Update memory slot directly (~1-5ms)
• Selective Forget: Decay specific memory strength 
• Online Update: Learn from deployment experiences
• Local Retrieval: Access relevant memories (<1ms)
```

### **4. Cross-Modal Fusion (FIBER)**
```
Text Embeddings + Vision Features → Cross-Attention
Text-to-Vision Attention → Vision-enhanced Text
Vision-to-Text Attention → Text-enhanced Vision  
Concatenate → Fusion MLP → Multimodal Representation
```

### **5. Multi-Layer Attention with Sinks**
```
For each layer (4 layers):
  Input → Q, K, V Projections
  Attention Sinks (4 tokens) + Sliding Window (128 tokens)
  Multi-Head Attention → Attention Weights [batch, heads, seq, seq]
  Attention Weights × Values → Attended Output
  Update Cache for Next Iteration
```

### **6. Reasoning Module (Tiny-R1)**
```
Attended Features → Reasoning Encoder → Reasoning Space
For each reasoning step (max 8 steps):
  LSTM Processing → Reasoning State
  Gate Network → Continue/Stop Decision
  Accumulate Reasoning States
Final Reasoning → Decoder → Enhanced Features
```

### **7. Robot Selection**
```
Task Representation → Task Encoder
For each robot (16 types):
  Task Features + Robot Embedding → Similarity Score
All Similarities → Selection Network → Robot Probabilities
Argmax → Selected Robot + Confidence Score
```

### **8. Output Generation**
```
Enhanced Features → Layer Normalization
Normalized Features → Output Projection → Logits [batch_size, seq_len, vocab_size]
Logits → Text Generation (sampling/greedy)
Robot Probabilities → Robot Selection Output
Attention States → Cache for Next Token
```

## 🧠 Episodic Memory: The Key Advantage

BitGen's episodic memory system provides **critical advantages for edge deployment**:

### 🚀 **Fast Local Knowledge Access**
- **Low Latency**: Memory accessed locally on device, eliminating network delays
- **Edge-Optimized**: Knowledge retrieval happens on-device without cloud dependency
- **Real-time Updates**: Immediate access to latest information and experiences

### ⚡ **Dynamic Knowledge Management** (No Retraining Required)
- **Fast Fact Editing**: Update knowledge instantly without model retraining
- **Selective Forgetting**: Remove outdated information while preserving important memories
- **High Accuracy on Updated Knowledge**: Maintains performance on new information
- **Continuous Learning**: Adapts to new experiences during deployment

### 🎯 **Latent Information Advantages**
- **Compressed Knowledge**: Stores experiences as latent representations for efficiency
- **Contextual Retrieval**: Accesses relevant memories based on current context
- **Memory Efficiency**: Compact storage suitable for edge device constraints
- **Experience-Based Learning**: Learns from actual deployment experiences

**Unlike traditional LLMs that require full retraining for knowledge updates, BitGen's episodic memory enables real-time knowledge management directly on edge devices.**

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
