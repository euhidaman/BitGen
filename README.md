# BitMar: Multimodal Vision-Language Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** that combines BitNet-quantized text processing, DiNOv3 vision embeddings, and episodic memory mechanisms for efficient multimodal understanding.

## 🌟 Key Features

- **Unlimited Training**: No token constraints - trains on entire dataset
- **BitNet Quantization**: 1.58-bit quantized text encoder/decoder for efficient inference
- **DiNOv3 Vision Embeddings**: Pre-computed 768-dim features for visual understanding
- **Episodic Memory**: Cross-modal memory system for visual-text associations
- **Selective Dataset Support**: Choose between Localized Narratives, COCO, or both
- **Comprehensive Logging**: Detailed WandB visualizations and metrics tracking
- **Hugging Face Integration**: Automatic model uploads after each epoch
- **Carbon Tracking**: Environmental impact monitoring

## 📊 Dataset Information & Size Selection

Choose your dataset size based on compute resources:

### Option 1: COCO Only (~615K samples) - **Smallest**
- **COCO Train 2017**: ~590K caption samples (118K images × 5 captions each)
- **COCO Val 2017**: ~25K caption samples (5K images × 5 captions each)
- **Total**: ~615,000 image-caption pairs
- **Download size**: ~500MB
- **Recommended for**: Quick experiments, limited compute

### Option 2: Localized Narratives Only (~1.16M samples) - **Medium**
- **Open Images Train**: ~870K narrative samples
- **Open Images Validation**: ~42K narrative samples  
- **Open Images Test**: ~125K narrative samples
- **COCO Train**: ~118K narrative samples
- **COCO Validation**: ~5K narrative samples
- **Total**: ~1,160,000 narrative samples
- **Download size**: ~1.5GB
- **Recommended for**: Balanced training with richer descriptions

### Option 3: Both Datasets (~1.78M samples) - **Largest**
- **Localized Narratives**: ~1.16M samples (detailed narratives)
- **COCO Captions**: ~615K samples (concise captions)
- **Total**: ~1,775,000 image-caption pairs
- **Download size**: ~2GB
- **Recommended for**: Maximum performance, sufficient compute resources

## 🛠️ Installation & Setup

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/euhidaman/BitGen.git
cd BitGen

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install PyTorch with CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers datasets wandb huggingface_hub
pip install numpy scipy scikit-learn matplotlib seaborn
pip install tqdm pyyaml requests pillow
pip install codecarbon  # Optional: for carbon tracking
```

### Step 2: Download Dataset (Choose Your Size)

#### Option A: COCO Only (Smallest - 615K samples)
```bash
python download_multimodal_data.py --coco-only --data_dir ./data
```

#### Option B: Localized Narratives Only (Medium - 1.16M samples)
```bash
python download_multimodal_data.py --localized-narratives-only --data_dir ./data
```

#### Option C: Both Datasets (Largest - 1.78M samples)
```bash
python download_multimodal_data.py --both --data_dir ./data
```

#### Get Help on Dataset Options
```bash
python download_multimodal_data.py --help
```

### Step 3: Verify Dataset Loading
```bash
# Test dataset loading with DiNOv3 embedding creation
python -c "
from src.dataset import LocalizedNarrativesCOCODataset
dataset = LocalizedNarrativesCOCODataset(
    dataset_dir='./data',
    extract_vision_features=True,  # Enable DiNOv3 embeddings
    use_dummy_vision=False
)
print(f'✅ Dataset loaded: {len(dataset):,} samples')
print(f'Sample keys: {list(dataset[0].keys())}')
print(f'Vision features shape: {dataset[0][\"vision_features\"].shape}')
"
```

### Step 4: Configure Training
Edit `configs/bitmar_config.yaml` to match your dataset choice:

```yaml
data:
  dataset_dir: "./data"
  extract_vision_features: true   # Enable DiNOv3 embeddings
  use_dummy_vision: false        # Use real DiNOv3 features
  batch_size: 64                 # Adjust for your GPU (32/64/128)
  
training:
  max_epochs: 20                 # Full dataset passes
  learning_rate: 0.0002
```

### Step 5: Start Training
```bash
# Basic training with automatic DiNOv3 embedding extraction
python train_100M_tokens.py --config configs/bitmar_config.yaml

# With specific GPU
python train_100M_tokens.py --config configs/bitmar_config.yaml --device cuda:0

# With frequent checkpoints
python train_100M_tokens.py --config configs/bitmar_config.yaml --save_every_n_steps 1000

# Rebuild dataset cache if needed
python train_100M_tokens.py --config configs/bitmar_config.yaml --rebuild_cache
```

## 🔄 Data Flow Pipeline

The complete data flow showing where DiNOv3 fits:

```
1. Download Phase:
   Raw Images + Captions (COCO/Localized Narratives)
           ↓
2. Embedding Phase (during dataset loading):
   DiNOv3 processes images → 768-dim embeddings
           ↓
3. Training Phase:
   Pre-computed embeddings → Your BitNet Model (1.58-bit quantized)
```

**DiNOv3 Usage:**
- ✅ **ONLY for preprocessing**: Creates embeddings from images during dataset loading
- ✅ **NOT part of your main model**: DiNOv3 runs once during `dataset.py` initialization
- ✅ **Your model is separate**: Uses BitNet quantization with your own encoder/decoder

## 💾 Training Variations by Dataset Size

### COCO Only Training (Fastest)
```bash
# Download COCO only
python download_multimodal_data.py --coco-only

# Train on 615K samples
python train_100M_tokens.py --config configs/bitmar_config.yaml
# Expected: 2-4 hours on RTX 4090
```

### Localized Narratives Only Training (Balanced)
```bash
# Download Localized Narratives only  
python download_multimodal_data.py --localized-narratives-only

# Train on 1.16M samples
python train_100M_tokens.py --config configs/bitmar_config.yaml
# Expected: 6-10 hours on RTX 4090
```

### Full Dataset Training (Maximum Performance)
```bash
# Download both datasets
python download_multimodal_data.py --both

# Train on 1.78M samples
python train_100M_tokens.py --config configs/bitmar_config.yaml
# Expected: 10-15 hours on RTX 4090
```

## 📈 Expected Training Statistics by Dataset

### COCO Only (615K samples):
- **Estimated tokens**: ~155M tokens
- **Steps per epoch**: ~9,600 steps
- **Total steps (20 epochs)**: ~192K steps
- **Training time**: 2-4 hours (RTX 4090)

### Localized Narratives Only (1.16M samples):
- **Estimated tokens**: ~290M tokens
- **Steps per epoch**: ~18,100 steps
- **Total steps (20 epochs)**: ~362K steps
- **Training time**: 6-10 hours (RTX 4090)

### Both Datasets (1.78M samples):
- **Estimated tokens**: ~445M tokens
- **Steps per epoch**: ~27,700 steps
- **Total steps (20 epochs)**: ~554K steps
- **Training time**: 10-15 hours (RTX 4090)

## 🎛️ Advanced Configuration

### Vision Feature Processing
```yaml
data:
  # DiNOv3 embedding configuration
  extract_vision_features: true          # Enable real DiNOv3 features
  use_dummy_vision: false               # Disable dummy features
  vision_model: "facebook/dinov3-vits16-pretrain-lvd1689m"  # DiNOv3 small
  
  # Alternative: DiNOv3 base (higher quality, slower)
  # vision_model: "facebook/dinov3-vitb16-pretrain-lvd1689m"
```

### Memory and Performance
```yaml
memory_optimization:
  use_gradient_checkpointing: true      # Trade compute for memory
  use_fp16: true                       # Mixed precision training
  empty_cache_frequency: 10           # GPU memory cleanup
  
data:
  batch_size: 32    # Reduce if GPU memory issues
  num_workers: 4    # Reduce if CPU bottleneck
```

## 🔧 Troubleshooting

### Dataset Issues
```bash
# If dataset download fails, retry specific dataset
python download_multimodal_data.py --coco-only --data_dir ./data

# If vision features fail, use dummy features temporarily
# Edit configs/bitmar_config.yaml:
# data:
#   use_dummy_vision: true
#   extract_vision_features: false
```

### Memory Issues
```bash
# Reduce batch size in config
# data:
#   batch_size: 32  # or 16 for low memory

# Enable gradient checkpointing
# memory_optimization:
#   use_gradient_checkpointing: true
```

### GPU Issues
```bash
# Force CPU training (very slow)
python train_100M_tokens.py --config configs/bitmar_config.yaml --device cpu

# Use specific GPU
python train_100M_tokens.py --config configs/bitmar_config.yaml --device cuda:1
```

## 📊 Monitoring Training Progress

### Weights & Biases Dashboard
Training automatically logs comprehensive metrics:
- Real-time loss curves and learning rates
- Cross-modal similarity scores
- DiNOv3 embedding quality metrics
- Token processing statistics
- Memory usage and episodic memory evolution
- Attention pattern visualizations
- Model quantization statistics
- FLOPS and computational efficiency
- Carbon emissions tracking

### Local Monitoring Commands
```bash
# View training logs in real-time
tail -f training.log

# Monitor GPU usage (NVIDIA GPUs)
watch -n 1 nvidia-smi

# Check disk space (checkpoints grow over time)
watch -n 30 "df -h ."

# Monitor Python process memory
ps aux | grep python

# Check checkpoint sizes
ls -lh checkpoints_100M_dataset/
```

## 📁 Output Directory Structure

After training, you'll have:

```
BitGen/
├── data/                                  # Dataset files
│   ├── all_captions.json                 # Unified captions
│   ├── localized_narratives/             # LN annotations (if downloaded)
│   └── coco/                            # COCO annotations (if downloaded)
├── checkpoints_100M_dataset/             # Model checkpoints
│   ├── latest_checkpoint.pt             # Most recent model
│   ├── checkpoint_epoch_X_step_Y.pt     # Epoch checkpoints
│   └── memory_exports/                  # Edge deployment packages
├── logs_100M_dataset/                   # Training logs
├── attention_100M_dataset/              # Attention analysis
├── memory_100M_dataset/                 # Memory visualizations  
├── results_100M_dataset/                # Final results
├── flops_logs_100M/                     # FLOPS tracking data
├── carbon_logs/                         # CO2 emissions tracking
└── training.log                         # Main training log
```

## 🚨 System Requirements by Dataset Size

### COCO Only (~615K samples):
- **GPU**: 6GB VRAM minimum (RTX 3060 or better)
- **RAM**: 12GB system RAM
- **Storage**: 10GB free space
- **Training time**: 2-4 hours (RTX 4090)

### Localized Narratives Only (~1.16M samples):
- **GPU**: 8GB VRAM minimum (RTX 3070 or better)  
- **RAM**: 16GB system RAM
- **Storage**: 15GB free space
- **Training time**: 6-10 hours (RTX 4090)

### Both Datasets (~1.78M samples):
- **GPU**: 12GB VRAM minimum (RTX 4070 Ti or better)
- **RAM**: 24GB system RAM recommended
- **Storage**: 25GB free space
- **Training time**: 10-15 hours (RTX 4090)

## 🎯 Complete Step-by-Step Workflow

### Phase 1: Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/euhidaman/BitGen.git
cd BitGen

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets wandb huggingface_hub
pip install numpy scipy scikit-learn matplotlib seaborn
pip install tqdm pyyaml requests pillow codecarbon
```

### Phase 2: Dataset Selection & Download

#### Choose your dataset size:

**Small Dataset (Quick experiments):**
```bash
python download_multimodal_data.py --coco-only --data_dir ./data
# Downloads: 615K samples, ~500MB, 2-4 hour training
```

**Medium Dataset (Balanced training):**
```bash
python download_multimodal_data.py --localized-narratives-only --data_dir ./data
# Downloads: 1.16M samples, ~1.5GB, 6-10 hour training
```

**Large Dataset (Maximum performance):**
```bash
python download_multimodal_data.py --both --data_dir ./data
# Downloads: 1.78M samples, ~2GB, 10-15 hour training
```

**Check available options:**
```bash
python download_multimodal_data.py --help
# Shows all dataset selection options with size estimates
```

### Phase 3: Dataset Verification
```bash
# Verify dataset was downloaded correctly
python -c "
import json
from pathlib import Path
data_dir = Path('./data')
if (data_dir / 'all_captions.json').exists():
    with open(data_dir / 'all_captions.json', 'r') as f:
        captions = json.load(f)
    print(f'✅ Dataset loaded: {len(captions):,} captions')
else:
    print('❌ Dataset not found. Please run download script first.')
"
```

### Phase 4: DiNOv3 Embedding Configuration
Edit `configs/bitmar_config.yaml` for vision processing:

```yaml
data:
  # DiNOv3 vision embedding settings
  extract_vision_features: true   # Enable real DiNOv3 embeddings
  use_dummy_vision: false        # Disable dummy features for real training
  vision_model: "facebook/dinov3-vits16-pretrain-lvd1689m"  # DiNOv3 small (faster)
  
  # For higher quality (slower):
  # vision_model: "facebook/dinov3-vitb16-pretrain-lvd1689m"  # DiNOv3 base
```

### Phase 5: Training Configuration
Adjust training settings based on your hardware:

```yaml
data:
  batch_size: 64        # Reduce to 32 or 16 if GPU memory issues
  num_workers: 6        # Adjust based on CPU cores
  
training:
  max_epochs: 20        # Full dataset passes
  learning_rate: 0.0002
  gradient_clip_val: 0.3
```

### Phase 6: Start Training
```bash
# Basic training (uses config defaults)
python train_100M_tokens.py --config configs/bitmar_config.yaml

# With specific GPU selection
python train_100M_tokens.py --config configs/bitmar_config.yaml --device cuda:0

# With frequent checkpointing (every 1000 steps)
python train_100M_tokens.py --config configs/bitmar_config.yaml --save_every_n_steps 1000

# Rebuild dataset cache if you changed vision settings
python train_100M_tokens.py --config configs/bitmar_config.yaml --rebuild_cache
```

### Phase 7: Background Training (Linux/Mac)
```bash
# Run training in background with full logging
nohup python train_100M_tokens.py --config configs/bitmar_config.yaml > training_output.log 2>&1 &

# Monitor progress in real-time
tail -f training_output.log

# Check training process
ps aux | grep train_100M_tokens
```

## 🔄 DiNOv3 Data Flow Pipeline

Understanding when DiNOv3 runs in your pipeline:

```
Phase 1: Dataset Download
├── download_multimodal_data.py
├── Downloads image URLs + captions
└── Creates: data/all_captions.json

Phase 2: Embedding Creation (during dataset loading)
├── dataset.py loads LocalizedNarrativesCOCODataset
├── DiNOv3 processes images: Image URLs → 768-dim embeddings
├── Pre-computes all vision features
└── Stores embeddings for training

Phase 3: Training (DiNOv3 NOT involved)
├── Your BitNet model receives pre-computed embeddings
├── BitNet 1.58-bit quantized text processing
├── Episodic memory cross-modal fusion
└── No DiNOv3 code runs during training
```

**Key Point**: DiNOv3 is **preprocessing only** - your main model uses BitNet quantization and your own architecture.

## 📈 Training Monitoring & Logs

### Real-time Monitoring
```bash
# Training logs
tail -f training.log

# GPU monitoring (if using NVIDIA GPU)
watch -n 1 nvidia-smi

# Disk space monitoring (checkpoints grow over time)
watch -n 30 "df -h ."

# Training process status
ps aux | grep python
```

### Weights & Biases Dashboard
Access your training dashboard at: `https://wandb.ai/[your-username]/bitmar-100M-attention-epochs`

**Logged Metrics:**
- Loss curves and learning rates
- Cross-modal similarity scores
- DiNOv3 embedding quality metrics
- Token processing statistics
- Memory usage and episodic memory evolution
- Attention pattern visualizations
- BitNet quantization statistics
- FLOPS and computational efficiency
- Carbon emissions tracking

## 🚨 Troubleshooting Guide

### Dataset Download Issues
```bash
# If download fails, check internet connection and retry
python download_multimodal_data.py --coco-only --data_dir ./data

# Check downloaded files
ls -la data/
ls -la data/localized_narratives/  # If using LN
ls -la data/coco/                  # If using COCO
```

### DiNOv3 Embedding Issues
```bash
# If DiNOv3 download fails, temporarily use dummy features
# Edit configs/bitmar_config.yaml:
data:
  extract_vision_features: false
  use_dummy_vision: true

# Then run training to test other components
python train_100M_tokens.py --config configs/bitmar_config.yaml
```

### GPU Memory Issues
```bash
# Reduce batch size
# Edit configs/bitmar_config.yaml:
data:
  batch_size: 32  # or 16 for 8GB GPUs

# Enable gradient checkpointing
memory_optimization:
  use_gradient_checkpointing: true
  use_fp16: true
```

### CPU Training (Not Recommended)
```bash
# Force CPU training if GPU unavailable
python train_100M_tokens.py --config configs/bitmar_config.yaml --device cpu
# Warning: Will be 50-100x slower than GPU training
```

## 🎯 Quick Start Commands Summary

```bash
# 1. Setup
git clone https://github.com/euhidaman/BitGen.git && cd BitGen
python -m venv venv && venv\Scripts\activate
pip install torch transformers wandb requests tqdm pyyaml numpy pillow

# 2. Choose dataset size and download
python download_multimodal_data.py --coco-only        # Small (615K)
# OR
python download_multimodal_data.py --localized-narratives-only  # Medium (1.16M)
# OR  
python download_multimodal_data.py --both             # Large (1.78M)

# 3. Start training
python train_100M_tokens.py --config configs/bitmar_config.yaml

# 4. Monitor (optional)
tail -f training.log
```

**That's it!** Your model will train on the selected dataset with DiNOv3 embeddings and BitNet quantization, uploading checkpoints to Hugging Face Hub automatically.
