# BitGen: Advanced Tiny Language Model for Embedded Systems

An advanced tiny language model that integrates **Larimar Episodic Memory**, **BitNet 1.58-bit Quantization**, **FIBER Cross-Modal Fusion**, **Attention Sinks**, **Tiny-R1 Reasoning**, and **Robot Selection** capabilities.

## 🚀 Quick Start: Training BitGen

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Download COCO dataset
python download_coco_dataset.py
```

### Stage 1: Vision-Language Pre-training
Train stable vision-language representations using COCO dataset (NO reasoning yet):

```bash
cd src
python train_stage1_vision_language.py
```

**What Stage 1 does:**
- FIBER-style cross-modal fusion with queue-based contrastive learning (4096 queues)
- Larima GPM episodic memory training (Bayesian inference)
- DINOv2 vision encoder (trainable, end-to-end)
- Contrastive accuracy target: >40% (image↔text matching)
- Output: `checkpoints/stage1/stage1_best.pt`

**Training time:** ~5-10 hours on A40 GPU (10 epochs, COCO 307K samples)

### Stage 2: Reasoning Module Training
Load Stage 1 checkpoint and add reasoning on top:

```bash
cd src
python train_stage2_reasoning.py
```

**What Stage 2 does:**
- Loads frozen Stage 1 vision-language base
- Adds tiny-r1 style reasoning module (chain-of-thought)
- GRPO training with reward functions (correctness, trace quality)
- Robot selection dataset (multi-label classification)
- Output: `checkpoints/stage2/stage2_best.pt`

**Training time:** ~2-4 hours on A40 GPU (20 epochs, Robot dataset ~1K samples)

### Dataset Structure
Your directory should look like:
```
BitGen/
├── data/
│   └── coco/
│       ├── validated_coco.json  # Validated COCO dataset (use this!)
│       ├── coco_dataset.json    # Original (before validation)
│       └── images/              # COCO images with captions aligned
├── robot_selection_data/
│   └── multi_robot_selection_dataset.json
├── checkpoints/
│   ├── stage1/  # Stage 1 checkpoints
│   └── stage2/  # Stage 2 checkpoints
└── src/
    ├── train_stage1_vision_language.py
    ├── train_stage2_reasoning.py
    ├── larima_memory.py
    ├── fiber_fusion.py
    └── ...
```

### Key Features
- **2-Stage Training**: Separate vision-language from reasoning (stable training)
- **Larima GPM**: Bayesian episodic memory (not simple storage)
- **FIBER Fusion**: Queue-based contrastive learning (4096 negatives)
- **Tiny-R1 Reasoning**: GRPO with reward functions (not supervised)
- **BitNet Ready**: Quantization hooks prepared for 1.58-bit deployment

## 🏗️ BitGen 2-Stage Training Architecture

### Stage 1: Vision-Language Pre-training (FIBER + Larima GPM)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  STAGE 1: VISION-LANGUAGE PRE-TRAINING                   │
│                     (COCO Dataset - 307K samples)                        │
└──────────────────────────────────────────────────────────────────────────┘

INPUT:
┌──────────────┐         ┌──────────────┐         ┌──────────────────────┐
│ Text Captions│         │COCO Images   │         │  WandB Monitoring:   │
│ (Tokenized)  │         │(224x224 RGB) │         │  • Contrastive Loss  │
└──────────────┘         └──────────────┘         │  • Memory KL Loss    │
       │                         │                 │  • T2I Accuracy      │
       ▼                         ▼                 │  • I2T Accuracy      │
┌──────────────┐         ┌──────────────┐         │  Target: >40% acc    │
│Token + Pos   │         │ DINOv2-base  │         └──────────────────────┘
│ Embeddings   │         │(Trainable)   │
│ [B, L, 256]  │         │ [B, P, 768]  │
└──────────────┘         └──────────────┘
       │                         │
       └─────────┬───────────────┘
                 ▼
┌───────────────────────────────────────────────────────────────┐
│              FIBER Cross-Modal Fusion                         │
│  ┌─────────────────┐      ┌──────────────────┐              │
│  │ ITC Transforms  │      │ Queue Negatives  │              │
│  │ (Separate)      │      │ 4096 text/image  │              │
│  │ Text → Common   │      │ Temperature: 0.07│              │
│  │ Image → Common  │      └──────────────────┘              │
│  └─────────────────┘                                         │
│                                                               │
│  Contrastive Learning: image ↔ text matching                │
│  Loss: (loss_t2i + loss_i2t) / 2                           │
└───────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────────────┐
│          Larimar GPM (Generative Parametric Memory)           │
│  ┌──────────────────────────────────────────────┐            │
│  │ memory_mean: [1000, 256] (trainable)         │            │
│  │ memory_logvar: [1] (Bayesian variance)       │            │
│  │                                               │            │
│  │ Read: Top-K similarity retrieval (k=5)       │            │
│  │ Write: Direct write with importance weights  │            │
│  │ Loss: KL divergence (prior ↔ posterior)     │            │
│  └──────────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────────────┐
│         6-Layer Transformer Attention                         │
│         (Multi-head, batch_first=True)                        │
└───────────────────────────────────────────────────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Checkpoint Saved │
        │ stage1_best.pt   │
        │ To HuggingFace:  │
        │ Every 2 epochs   │
        └──────────────────┘
```

### Stage 2: Reasoning Module Training (Tiny-R1 + Robot Selection)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  STAGE 2: REASONING MODULE TRAINING                      │
│                  (Robot Dataset - ~1K samples)                           │
└──────────────────────────────────────────────────────────────────────────┘

INPUT:
┌──────────────────────────────────────────────────────────────────────────┐
│ Load Stage 1 Checkpoint (FROZEN - No gradient updates)                  │
│ ✓ Vision-Language base with trained representations                     │
│ ✓ FIBER fusion weights locked                                           │
│ ✓ Larima GPM memory frozen                                              │
└──────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────────────┐
│          Tiny-R1 Reasoning Module (TRAINABLE)                 │
│  ┌──────────────────────────────────────────────┐            │
│  │ Reasoning Encoder: [256 → 64]                │            │
│  │ LSTM Processor: max 8 steps                  │            │
│  │ Gate Mechanism: continue/stop decision       │            │
│  │ Reasoning Decoder: [64 → 256]                │            │
│  └──────────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────────────┐
│        Robot Selector (TRAINABLE - Multi-label)               │
│  ┌──────────────────────────────────────────────┐            │
│  │ 5 Robot Types:                                │            │
│  │  1. Drone                                     │            │
│  │  2. Underwater Robot                          │            │
│  │  3. Humanoid                                  │            │
│  │  4. Robot with Wheels                         │            │
│  │  5. Robot with Legs                           │            │
│  │                                               │            │
│  │ Top-3 Selection (independent probabilities)  │            │
│  │ BCE Loss per robot                            │            │
│  └──────────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────────────┐
│         GRPO Training (Reward-based RL)                       │
│  ┌──────────────────────────────────────────────┐            │
│  │ Correctness Reward: Did model select right? │            │
│  │ Reasoning Trace Reward: Is trace valid?     │            │
│  │ Loss: -(correctness_reward + trace_reward)  │            │
│  └──────────────────────────────────────────────┘            │
│                                                               │
│  WandB Monitoring:                                           │
│  • Robot Selection Loss                                      │
│  • Correctness Reward                                        │
│  • Reasoning Reward                                          │
│  • Robot Selection Accuracy                                  │
│  • 5x5 Confusion Matrix                                      │
└───────────────────────────────────────────────────────────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Checkpoint Saved │
        │ stage2_best.pt   │
        │ To HuggingFace:  │
        │ Every epoch      │
        └──────────────────┘
```

## 📊 WandB Metrics & Visualizations

### Stage 1 Metrics (babylm-ntust team)
- `stage1/loss/total` - Total training loss
- `stage1/loss/contrastive` - Image-text contrastive loss
- `stage1/loss/memory_kl` - Larima GPM KL divergence
- `stage1/accuracy/text_to_image` - Text→Image retrieval accuracy
- `stage1/accuracy/image_to_text` - Image→Text retrieval accuracy
- `stage1/accuracy/average` - Average contrastive accuracy
- `stage1/learning_rate` - Current learning rate

**Visualizations:**
- Contrastive similarity matrices (image ↔ text matching)
- Larima GPM memory state heatmaps (memory utilization)
- Loss curves over epochs
- Accuracy trends (target: >40%)

### Stage 2 Metrics (babylm-ntust team)
- `stage2/loss/total` - Total training loss
- `stage2/loss/robot_selection` - BCE loss for robot classification
- `stage2/reward/correctness` - GRPO correctness reward
- `stage2/reward/reasoning_trace` - Reasoning quality reward
- `stage2/reward/total` - Combined rewards
- `stage2/accuracy/robot_selection` - Multi-label accuracy
- `stage2/learning_rate` - Current learning rate

**Visualizations:**
- 5x5 Robot confusion matrix (per robot type)
- Robot selection accuracy over epochs
- Reward progression (correctness + trace quality)
- Loss curves and convergence
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

### **7. Robot Selection (Top-3 Multi-Label with Chain-of-Thought)**
```
Task/Scene Representation → Tiny-R1 Reasoning (3-8 steps)
Reasoning Output → Task Encoder → Task Features
For each robot (5 types: Drone, Underwater, Humanoid, Wheels, Legs):
  Task Features + Robot Embedding → Binary Score (independent)
  Sigmoid Activation → Robot Suitability Probability [0, 1]
All Robot Probabilities → Top-K Selection (k=3)
Top-3 Robots + Confidences → Multi-Robot Deployment Decision
Ground Truth vs Predictions → 5x5 Confusion Matrix Update
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

## ✅ All Features & Modules Present

### Core Model Components (bitgen_model.py)
- ✅ **BitGenConfig** - Model configuration with all hyperparameters
- ✅ **BitNetLinear** - 1.58-bit quantization layer ({-1, 0, +1} weights)
- ✅ **EpisodicMemory** - Original episodic memory (64 slots, key-value)
- ✅ **VisionAttentionSink** - Vision-specific attention with sinks
- ✅ **AttentionSink** - Text attention with sliding window (4 sinks, 128 window)
- ✅ **CrossModalFusion** - DINOv2-based cross-modal fusion (legacy, kept for compatibility)
- ✅ **ReasoningModule** - Tiny-R1 style reasoning (LSTM, gate mechanism, 8 steps)
- ✅ **RobotSelector** - Multi-label robot selection (5 types, top-3 selection)
- ✅ **BitGenModel** - Complete integrated model (for inference/deployment)

### New 2-Stage Training Components
- ✅ **larima_memory.py** - Larima GPM with Bayesian inference (1000 slots)
- ✅ **fiber_fusion.py** - FIBER cross-modal with queue-based contrastive (4096 queues)
- ✅ **train_stage1_vision_language.py** - Stage 1 training script
- ✅ **train_stage2_reasoning.py** - Stage 2 training script

### Data Loaders (data_loader.py)
- ✅ **COCODataset** - COCO image-caption pairs (307K samples)
- ✅ **RobotDataset** - Robot selection with multi-label classification
- ✅ **BitGenTokenizer** - Embedded-optimized tokenizer (8192 vocab)

### Monitoring & Integration
- ✅ **wandb_integration.py** - WandB logging (babylm-ntust team)
  - Stage 1/2 specific metrics
  - Contrastive similarity visualizations
  - Robot confusion matrix
  - Larima GPM memory heatmaps
- ✅ **huggingface_integration.py** - Auto-push to HuggingFace Hub
  - Stage 1: `{username}/BitGen-Reasoning-stage1`
  - Stage 2: `{username}/BitGen-Reasoning-stage2`

### All Original Features Preserved
- ✅ **Attention Sinks** - 4 sink tokens + 128 sliding window
- ✅ **BitNet Quantization** - Ready for 1.58-bit deployment
- ✅ **Episodic Memory** - Both original (64 slots) and Larima GPM (1000 slots)
- ✅ **FIBER Fusion** - Queue-based contrastive learning
- ✅ **Tiny-R1 Reasoning** - GRPO training with rewards
- ✅ **Robot Selection** - Top-3 multi-label classification
- ✅ **DINOv2 Vision** - Trainable end-to-end vision encoder

## � Quick Start: Complete Setup (Sequential Steps)

### Step 1: Clone and Install Dependencies
```bash
git clone https://github.com/euhidaman/BitGen.git
cd BitGen
pip install -r requirements.txt
```

### Step 2: Setup Kaggle API for Dataset Download
```bash
# Place your kaggle.json in ~/.kaggle/ directory
# Get kaggle.json from: https://www.kaggle.com/settings/account
# Linux/Mac:
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell):
mkdir $env:USERPROFILE\.kaggle -Force
Move-Item kaggle.json $env:USERPROFILE\.kaggle\
```

### Step 3: Download COCO Dataset from Kaggle
```bash
python download_coco_dataset.py
# This downloads COCO 2017 validation set to data/coco/
```

### Step 4: Download Robot Selection Dataset
```bash
# Download multi_robot_selection_dataset.json from your data source
# Place it in: robot_selection_data/data/Multi-Robot-Selection/
mkdir -p ../robot_selection_data/data/Multi-Robot-Selection
# Copy your multi_robot_selection_dataset.json to this directory
```

### Step 5: Setup HuggingFace Hub (Optional - for model pushing)
```bash
# PowerShell:
$env:HF_TOKEN="your_huggingface_token"

# OR login via CLI:
huggingface-cli login
```

### Step 6: Setup WandB (Optional - for metrics tracking)
```bash
wandb login
# Make sure you have access to 'babylm-ntust' team
```

### Step 7: Train the Model
```bash
python bitgen_cli.py train `
  --coco_data data/coco `
  --robot_data ../robot_selection_data/data `
  --model_size tiny `
  --batch_size 16 `
  --num_epochs 10 `
  --use_wandb
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

## 🎯 Deployment Strategy

**BitGen is designed for this exact workflow:**

### 🚀 Training Environment (RTX 4090)
- **High-performance training** with full GPU acceleration
- **Comprehensive monitoring** with FLOPS, energy tracking, and visualizations
- **Advanced metrics** including episodic memory heatmaps and reasoning matrices
- **Automatic model pushing** to HuggingFace Hub after every epoch

### 📱 Inference Environment (Raspberry Pi Zero)
- **Optimized inference** with 1.58-bit quantization for ultra-low power
- **Edge monitoring** with thermal, power, and performance tracking
- **Fast episodic memory operations** (fact editing, selective forgetting)
- **Local knowledge management** without cloud dependencies

## 🔧 Platform-Specific Optimizations

### For Training (RTX 4090):
```bash
# Full-featured training with all monitoring
python bitgen_cli.py train \
  --coco_data data/coco/validated_coco.json \
  --model_size tiny \
  --batch_size 32 \
  --num_epochs 50 \
  --enable_carbon_tracking \
  --track_flops \
  --push_to_hub \
  --use_wandb
```

### For Inference (Raspberry Pi Zero):
```bash
# Optimized inference with Pi-specific monitoring
python bitgen_cli.py inference \
  --model_path checkpoints/bitgen_final.pt \
  --benchmark \
  --show_metrics
```

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
- **Tiny-R1 Reasoning**: Chain-of-thought reasoning with 3-8 LSTM steps
- **Robot Selection**: 5 robot types (Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs)
  - Top-3 multi-label classification with sigmoid activation
  - 5x5 confusion matrix tracking per epoch
  - Interleaved training: 90% COCO vision-language, 10% robot selection
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

### 2. Robot Task Selection (Top-3 Multi-Label)
```python
# Select top-3 most suitable robots for task/scene
robot_selection = bitgen.select_robots_for_task(
    task_description='Navigate rocky terrain to deliver medical supplies',
    image_path='scene.jpg'  # Optional: scene image for visual context
)

print("🤖 Top-3 Robot Recommendations:")
for i, (robot, confidence) in enumerate(zip(robot_selection['top_k_robots'], robot_selection['top_k_probs'])):
    print(f"  {i+1}. {robot:<25} (confidence: {confidence:.3f})")

# Output:
# 🤖 Top-3 Robot Recommendations:
#   1. Robot with Legs         (confidence: 0.892)
#   2. Drone                   (confidence: 0.745)
#   3. Robot with Wheels       (confidence: 0.623)
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

## 🏛️ BitGen Architecture Overview

### **Encoder-Decoder Design (BitMar-Inspired)**

BitGen follows an **encoder-decoder architecture** similar to BitMar, combining the best elements from three research projects:

1. **BitMar**: Encoder-decoder structure with multi-component loss
2. **FIBER**: Cross-modal fusion with queue-based contrastive learning
3. **Larimar**: Enhanced Generative Parametric Memory (GPM) with Bayesian inference

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BITGEN ARCHITECTURE                              │
│                  (BitMar + FIBER + Enhanced Larimar)                    │
└─────────────────────────────────────────────────────────────────────────┘

INPUT STAGE:
┌──────────────┐         ┌──────────────┐
│ Text Tokens  │         │ Image Patches│
│ [B, seq_len] │         │ [B, patches] │
└──────────────┘         └──────────────┘
       │                         │
       ▼                         ▼
┌──────────────┐         ┌──────────────┐
│Token + Pos   │         │DINOv2 Vision │
│Embeddings    │         │Encoder       │
└──────────────┘         └──────────────┘
       │                         │
       └─────────┬───────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FIBER CROSS-MODAL FUSION                      │
│  • Queue-based contrastive learning (4096 negative samples)    │
│  • Temperature-scaled similarity (τ = 0.1)                      │
│  • Bidirectional alignment: image ↔ text                        │
└─────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              ENHANCED LARIMAR GPM MEMORY                        │
│  • Bayesian parametric memory: mean + logvar                    │
│  • Top-K retrieval with cosine similarity (k=5)                 │
│  • Memory quality tracking (read/write counts)                  │
│  • KL divergence regularization                                 │
│  • Save/load capability for external storage                    │
└─────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│           MULTI-LAYER ATTENTION (6 layers)                      │
│  • Attention Sinks for streaming inference                      │
│  • Multi-head self-attention                                    │
│  • Gradient monitoring for stability                            │
└─────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              REASONING MODULE (Tiny-R1 Style)                   │
│  • LSTM-based chain-of-thought reasoning                        │
│  • Adaptive reasoning steps (max 8)                             │
│  • Gate mechanism for reasoning depth                           │
└─────────────────────────────────────────────────────────────────┘
                 │
                 ├────────────────┬────────────────┐
                 ▼                ▼                ▼
        ┌────────────────┐ ┌────────────┐ ┌──────────────┐
        │ ENCODER OUTPUT │ │  DECODER   │ │ROBOT SELECTOR│
        │   (Direct)     │ │   (New!)   │ │  (Top-K)     │
        └────────────────┘ └────────────┘ └──────────────┘

DECODER STAGE (BitMar-Style):
┌─────────────────────────────────────────────────────────────────┐
│                 BITNET TEXT DECODER                             │
│  • Multi-layer decoder with causal attention                    │
│  • Cross-attention to encoder output                            │
│  • Teacher forcing during training                              │
│  • Autoregressive generation during inference                   │
└─────────────────────────────────────────────────────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │Text Reconstruction│
        │    Logits        │
        │ [B, tgt_len, V]  │
        └─────────────────┘
```

### **Multi-Component Loss Function (BitMar-Style)**

The model is trained with three loss components, following BitMar's approach:

```
Total Loss = α·Text Loss + β·Contrastive Loss + γ·Memory KL Loss

Where:
  α = 1.0   (text reconstruction - main learning signal)
  β = 0.1   (image-text alignment - FIBER-style)
  γ = 0.05  (memory regularization - Larimar-style)
```

**Loss Components:**

1. **Text Reconstruction Loss** (α = 1.0) - PRIMARY SIGNAL
   - Cross-entropy on decoder output
   - Next-token prediction with teacher forcing
   - Metrics: Perplexity, Token Accuracy
   - This is the main learning objective

2. **Contrastive Loss** (β = 0.1) - ALIGNMENT
   - FIBER-style queue-based contrastive learning
   - Symmetric: loss_t2i + loss_i2t
   - Temperature-scaled similarity (τ = 0.1)
   - Ensures vision-language alignment

3. **Memory KL Divergence** (γ = 0.05) - REGULARIZATION
   - Larimar-style Bayesian memory regularization
   - KL(posterior || prior) on memory distributions
   - Prevents memory overfitting
   - Encourages generalization

### **Key Improvements from Original BitGen**

| Component           | Before                  | After (Current)                                         |
| ------------------- | ----------------------- | ------------------------------------------------------- |
| **Architecture**    | Encoder only            | Encoder-decoder (BitMar-style)                          |
| **Loss Function**   | Contrastive only        | Multi-component (text + contrastive + memory KL)        |
| **Memory**          | Simple GPM              | Enhanced GPM with Bayesian inference + quality tracking |
| **Cross-Modal**     | Basic fusion            | FIBER-style queue-based (4096 negatives)                |
| **Text Generation** | Direct projection       | Proper decoder with causal attention                    |
| **Learning Signal** | Weak (contrastive only) | Strong (text reconstruction primary)                    |
| **Optimization**    | Fixed LR                | Warmup + adaptive (ReduceLROnPlateau)                   |

### **Training Stability Features**

- **Warmup Scheduler**: 1000 steps linear warmup
- **Adaptive LR**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Temperature**: 0.1 (smoother gradients than 0.07)
- **Gradient Monitoring**: Track gradient norms every 100 steps
- **Mixed Precision**: torch.amp.autocast for faster training
- **Gradient Clipping**: Prevents exploding gradients

### **Architecture Diagram: Loss Computation Flow**

```
Forward Pass → Model Outputs → Compute Multi-Loss → Backprop
     │              │                    │                │
     │              ├── decoder_logits ──┤                │
     │              ├── contrastive_feat─┤                │
     │              └── memory_kl ────────┤                │
     │                                    │                │
     ▼                                    ▼                ▼
 [Input]                        [Loss Components]    [Gradients]
  Text                           • text_loss (1.0)      │
  Images                         • contrastive (0.1)    │
  Targets                        • memory_kl (0.05)     │
                                 ────────────────────   │
                                 Total Loss = Σ(αᵢ·Lᵢ)  │
                                                        ▼
                                                   [Optimizer]
```

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
