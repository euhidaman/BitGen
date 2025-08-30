# BitMar: Multimodal Vision-Language Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** that combines BitNet-quantized text processing, generic vision encoding, and episodic memory mechanisms for efficient multimodal understanding.

## 🌟 Key Features

- **Unlimited Training**: No token constraints - trains on entire dataset
- **BitNet Quantization**: 1.58-bit quantized text encoder/decoder for efficient inference
- **Episodic Memory**: Cross-modal memory system for visual-text associations
- **Generic Dataset Support**: Works with Localized Narratives, COCO, and any image-caption dataset
- **Comprehensive Logging**: Detailed WandB visualizations and metrics tracking
- **Hugging Face Integration**: Automatic model uploads after each epoch
- **Carbon Tracking**: Environmental impact monitoring

## 📊 Dataset Information

**Localized Narratives + COCO Dataset Counts:**

### Localized Narratives (~1,160,000 samples):
- **Open Images Train**: ~870,000 narrative samples
- **Open Images Validation**: ~42,000 narrative samples  
- **Open Images Test**: ~125,000 narrative samples
- **COCO Train**: ~118,000 narrative samples
- **COCO Validation**: ~5,000 narrative samples
- **Total Localized Narratives**: ~1,160,000 samples

### COCO Captions (~615,000 samples):
- **COCO Train 2017**: ~590,000 caption samples (118K images × 5 captions each)
- **COCO Val 2017**: ~25,000 caption samples (5K images × 5 captions each)
- **Total COCO Captions**: ~615,000 samples

### **TOTAL DATASET: ~1,775,000 image-caption pairs**

## 🛠️ Installation

```bash
git clone https://github.com/euhidaman/BitGen.git
cd BitGen
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers datasets wandb huggingface_hub
pip install numpy scipy scikit-learn matplotlib seaborn
pip install tqdm pyyaml requests pillow
pip install codecarbon  # Optional: for carbon tracking
```

## 📥 Complete Dataset Setup Commands

### Step 1: Download Datasets (NO BabyLM, NO DiNOv2 dependencies)

```bash
# Download Localized Narratives and COCO datasets
python download_multimodal_data.py --data_dir ./data

# This downloads:
# - Localized Narratives (Open Images + COCO): ~1.16M narrative samples
# - COCO 2017 Captions: ~615K caption samples  
# - Total: ~1.78M image-caption pairs
# - Size: ~2GB download (annotations only)
```

### Step 2: Verify Dataset Loading

```bash
# Test that datasets load correctly
python src/dataset.py

# Expected output:
# ✅ Loaded 1,775,000 image-caption pairs from Localized Narratives + COCO
# 📊 Dataset Information:
#   • total_samples: 1,775,000
#   • datasets: Localized Narratives + COCO
```

### Step 3: Quick Dataset Stats Check

```bash
# Get exact dataset counts
python -c "
from src.dataset import test_dataset
config = {
    'dataset_dir': './data',
    'text_encoder_name': 'gpt2', 
    'max_seq_length': 256,
    'batch_size': 4,
    'num_workers': 0,
    'pin_memory': False
}
test_dataset(config)
"
```

## 🚀 Training Commands (No Token Limits)

### Basic Training (All 1.78M Samples)

```bash
# Standard unlimited training
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml

# With specific GPU
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:0

# With frequent checkpoints
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_every_n_steps 1000
```

### Training Variations

```bash
# Rebuild cache if dataset changes
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --rebuild_cache

# Save every 500 steps for monitoring
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --save_every_n_steps 500

# Use specific GPU (if multiple available)
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:1
```

### Background Training (Linux/Mac)

```bash
# Run in background with logging
nohup python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml > training_output.log 2>&1 &

# Monitor progress
tail -f training_output.log
```

## 📈 Expected Training Statistics

**With 1,775,000 image-caption pairs:**
- **Estimated tokens**: ~450M+ tokens (far exceeding 100M constraints)
- **Batch size**: 64 samples
- **Steps per epoch**: ~27,734 steps  
- **Total steps (20 epochs)**: ~554,680 steps
- **Training time estimates**:
  - RTX 3090: 15-20 hours
  - RTX 4090: 10-15 hours  
  - A100: 8-12 hours
  - CPU: 200+ hours (not recommended)

## 🔧 Configuration Customization

Edit `configs/bitmar_100M_tokens.yaml`:

```yaml
data:
  dataset_dir: "./data"     # Your dataset location
  batch_size: 64           # Adjust for GPU memory (32/64/128)
  num_workers: 6           # CPU cores for data loading

training:
  max_epochs: 20           # Number of full dataset passes
  learning_rate: 0.0002    # Learning rate
  
# Confirmed: NO TOKEN LIMITS
token_tracking:
  enforce_token_limits: false      # Never stop at token limits
  unlimited_training: true         # Use all available data
  stop_at_token_limit: false      # Never stop training
```

## 📊 Monitoring Training Progress

### Weights & Biases Dashboard

Training automatically logs to WandB:
- Real-time loss curves
- Cross-modal similarity metrics
- Token processing statistics
- Memory usage patterns
- Attention visualizations
- Model quantization stats

### Local Monitoring

```bash
# View training logs in real-time
tail -f training.log

# Monitor GPU usage
nvidia-smi -l 1

# Check disk space (checkpoints can be large)
df -h

# Monitor process
ps aux | grep python
```

## 📁 Output Directory Structure

```
BitGen/
├── checkpoints_100M_dataset/         # Model checkpoints
│   ├── latest_checkpoint.pt          # Most recent model
│   ├── checkpoint_epoch_0_step_X.pt  # Epoch checkpoints
│   └── ...
├── logs_100M_dataset/               # Training logs
├── attention_100M_dataset/          # Attention analysis
├── memory_100M_dataset/             # Memory visualizations  
├── results_100M_dataset/            # Final results
├── carbon_logs/                     # CO2 emissions tracking
└── training.log                     # Main training log
```

## 🚨 System Requirements

### Minimum Requirements:
- **GPU**: 8GB VRAM (RTX 3070/4060 or better)
- **RAM**: 16GB system RAM
- **Storage**: 15GB free space
- **Python**: 3.9+

### Recommended:
- **GPU**: 16GB+ VRAM (RTX 4080/4090, A100)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ SSD space
- **Internet**: Stable connection for dataset download

## 🎯 Complete Workflow

### 1. Setup Environment
```bash
git clone https://github.com/euhidaman/BitGen.git
cd BitGen
python -m venv venv
venv\Scripts\activate  # Windows
pip install torch transformers wandb requests tqdm pyyaml numpy
```

### 2. Download Datasets  
```bash
python download_multimodal_data.py --data_dir ./data
# Downloads 1.78M image-caption pairs (~2GB)
```

### 3. Verify Setup
```bash
python src/dataset.py
# Should show: "✅ Loaded 1,775,000 image-caption pairs"
```

### 4. Start Training
```bash
python train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml
# Trains on ALL 1.78M samples with NO limits
```

### 5. Monitor Progress
- Check WandB dashboard
- View `training.log`
- Monitor GPU with `nvidia-smi`

The system will train on the complete **1,775,000 image-caption pairs** without any token constraints, giving you maximum data utilization!
