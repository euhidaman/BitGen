# BitGen Enhanced Training with HuggingFace Hub and WandB Integration

This guide explains how to use the enhanced BitGen training system with automatic HuggingFace Hub model pushing and comprehensive WandB tracking for the 'babylm-ntust' team.

## 🚀 Quick Start

### Prerequisites

1. **HuggingFace Hub Setup:**
   ```bash
   # Install huggingface-hub if not already installed
   pip install huggingface-hub
   
   # Login to HuggingFace
   huggingface-cli login
   # OR set environment variable
   export HF_TOKEN="your_huggingface_token_here"
   ```

2. **WandB Setup:**
   ```bash
   # Install wandb if not already installed
   pip install wandb
   
   # Login to WandB
   wandb login
   # Make sure you have access to the 'babylm-ntust' team
   ```

### Basic Training with HuggingFace and WandB

```bash
python bitgen_cli.py train \
  --coco_data data/coco/coco_train.json \
  --model_size tiny \
  --num_epochs 10 \
  --enable_carbon_tracking \
  --track_flops \
  --push_to_hub \
  --use_wandb
```

This will:
- ✅ Train a BitGen model with FLOPS tracking and CodeCarbon energy monitoring
- ✅ Push the model to HuggingFace Hub **after every epoch**
- ✅ Log comprehensive metrics to WandB in the **'babylm-ntust' team**
- ✅ Create detailed visualizations and performance dashboards

## 📊 What Gets Logged to WandB

### Training Metrics:
- **Loss curves**: Training loss, individual loss components, adaptive loss weights
- **Performance**: Tokens/second, FLOPS per step, training efficiency
- **System metrics**: CPU usage, memory usage, temperature, power consumption
- **Energy tracking**: Energy consumption (kWh), carbon emissions (kg CO2)

### Visualizations Created:
- **Loss Analysis Dashboard**: Multi-panel loss curves with trends
- **Performance Dashboard**: Throughput, latency, and efficiency over time
- **System Utilization**: CPU, memory, disk, and thermal monitoring
- **Architecture Visualization**: Model parameters distribution
- **Energy Efficiency Charts**: FLOPS per mJ, carbon efficiency

### Model Artifacts:
- **Model checkpoints**: Saved at each epoch with metadata
- **Configuration files**: Model architecture and training settings
- **Performance summaries**: Best metrics achieved during training

## 🤗 HuggingFace Hub Integration

### Automatic Model Pushing:
- **After every epoch**: Model is automatically pushed to HuggingFace Hub
- **Final model**: Complete trained model with all artifacts
- **Model cards**: Auto-generated with training metrics and usage instructions
- **Version tracking**: Each epoch creates a new commit with detailed metrics

### Repository Structure:
```
your-username/bitgen-tiny-20250924-143052/
├── config.json              # Model configuration
├── pytorch_model.bin         # Model weights
├── README.md                 # Auto-generated model card
├── training_info.json        # Training metadata
└── tokenizer_config.json     # Tokenizer configuration
```

## 🔧 Advanced Usage

### Custom HuggingFace Repository:
```bash
python bitgen_cli.py train \
  --coco_data data/coco/coco_train.json \
  --push_to_hub \
  --hf_repo_name "my-custom-bitgen-model" \
  --hf_organization "my-org" \
  --hf_private  # Create private repository
```

### Custom WandB Configuration:
```bash
python bitgen_cli.py train \
  --coco_data data/coco/coco_train.json \
  --use_wandb \
  --wandb_project "my-custom-project" \
  --wandb_entity "babylm-ntust" \
  --wandb_run_name "bitgen-experiment-v2" \
  --wandb_tags bitgen multimodal experiment v2
```

### Full Monitoring Setup:
```bash
python bitgen_cli.py train \
  --coco_data data/coco/coco_train.json \
  --robot_data data/robot_selection/robot_tasks.json \
  --model_size tiny \
  --num_epochs 15 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --enable_carbon_tracking \
  --track_flops \
  --push_to_hub \
  --hf_repo_name "bitgen-tiny-multimodal" \
  --use_wandb \
  --wandb_project "bitgen-training" \
  --wandb_entity "babylm-ntust" \
  --wandb_tags bitgen tiny multimodal robotics
```

## 📈 Monitoring Output Examples

### During Training:
```
🚀 Starting enhanced training with HuggingFace and WandB integration...
📊 WandB Project: babylm-ntust/bitgen-training
🤗 HuggingFace Repo: bitgen-tiny-20250924-143052
🔢 Calculating model FLOPS...
   Forward pass FLOPS: 12.5M
   Parameters: 3.2M
🌍 CodeCarbon tracking enabled

Epoch 1, Step 10: Loss=2.4567, Tokens/s=4.23, RAM=87.3MB, CPU=45.2%, Temp=58.1°C, Power=387mW
🚀 Pushing model to HuggingFace Hub for epoch 1...
✅ Model pushed to HuggingFace: https://huggingface.co/username/bitgen-tiny-20250924-143052

Epoch 1 completed in 245.7s
  Average loss: 2.4567
  Average tokens/s: 4.23
  Total FLOPS: 1,250,000,000
  🤗 Model URL: https://huggingface.co/username/bitgen-tiny-20250924-143052
```

### WandB Dashboard:
- **Real-time loss curves** with automatic best metric tracking
- **Performance visualizations** showing throughput and efficiency trends
- **System monitoring** with CPU, memory, and thermal profiles
- **Energy consumption** tracking with carbon footprint analysis
- **Model artifacts** with downloadable checkpoints

### HuggingFace Model Card (Auto-generated):
```markdown
# BitGen: Advanced Tiny Language Model

## Training Information
- **Current Epoch**: 5
- **Training Loss**: 1.8432
- **Tokens per Second**: 4.23
- **Total FLOPS**: 5,000,000,000
- **Average Power**: 387mW
- **Training Time**: 2.5 hours

## Usage
```python
from transformers import AutoModel, AutoConfig
model = AutoModel.from_pretrained("username/bitgen-tiny-20250924-143052")
```

### 🔍 Inference Monitoring:

When running inference, you get comprehensive metrics:
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

## 🎯 Key Features

1. **Automatic Model Versioning**: Each epoch creates a new version on HuggingFace Hub
2. **Comprehensive Metrics**: FLOPS, energy, carbon, performance, thermal
3. **Team Collaboration**: All runs logged to 'babylm-ntust' WandB team
4. **Interactive Dashboards**: Real-time monitoring and historical analysis
5. **Artifact Management**: Model checkpoints and metadata automatically saved

## 🔧 Environment Variables

Set these for seamless integration:
```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_ENTITY="babylm-ntust"
```

The system now provides complete integration with both HuggingFace Hub (for model sharing and versioning) and WandB (for comprehensive training monitoring and team collaboration) while maintaining all the requested performance tracking capabilities.
