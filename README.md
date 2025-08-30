# BitMar: Multimodal Vision-Language Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** that combines BitNet-quantized text processing, DiNOv3 vision embeddings, and episodic memory mechanisms for efficient multimodal understanding.

## 🌟 Key Features

- **Unlimited Training**: No token constraints - trains on entire dataset
- **BitNet Quantization**: 1.58-bit quantized text encoder/decoder for efficient inference
- **DiNOv3 Vision Embeddings**: Pre-computed 768-dim features for visual understanding
- **Episodic Memory**: Cross-modal memory system for visual-text associations (optional for ablation studies)
- **Selective Dataset Support**: Choose between Localized Narratives, COCO, or both
- **Ablation Study Support**: Train with/without episodic memory for performance comparison
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

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install PyTorch with CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

## 🧪 Ablation Study: Episodic Memory vs. No Memory

The repository includes configurations for ablation studies to compare model performance with and without episodic memory.

### Configuration Files for Ablation Study:
- `configs/bitmar_with_memory.yaml` - **Full model with episodic memory**
- `configs/bitmar_without_memory.yaml` - **Baseline model without episodic memory**
- `configs/bitmar_config.yaml` - **Default configuration**

### Step 4A: Train Model WITH Episodic Memory
```bash
# Train the full BitMar model with episodic memory (32 slots, 128-dim episodes)
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml

# Expected output directories:
# - checkpoints_with_memory/
# - logs_with_memory/
# - WandB project: bitmar-ablation-with-memory
# - HF repo: euhidaman/bitmar-with-memory-ablation
```

### Step 4B: Train Model WITHOUT Episodic Memory
```bash
# Train the baseline model without episodic memory (direct fusion only)
python train_100M_tokens.py --config configs/bitmar_without_memory.yaml

# Expected output directories:
# - checkpoints_without_memory/
# - logs_without_memory/
# - WandB project: bitmar-ablation-without-memory
# - HF repo: euhidaman/bitmar-without-memory-ablation
```

### Step 5: Monitor Both Training Runs
```bash
# Monitor WITH memory model
tail -f logs_with_memory/training.log

# Monitor WITHOUT memory model (in separate terminal)
tail -f logs_without_memory/training.log

# Check GPU usage
watch -n 1 nvidia-smi
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
├── WITH Memory: BitNet + Episodic Memory + Cross-modal fusion
├── WITHOUT Memory: BitNet + Direct fusion (no memory)
└── No DiNOv3 code runs during training
```

**Key Point**: DiNOv3 is **preprocessing only** - your main model uses BitNet quantization and your own architecture.

## 📈 Ablation Study Results Comparison

After training both models, you can compare:

### Model Architecture Differences:

**WITH Episodic Memory:**
- ✅ 32 memory slots for cross-modal associations
- ✅ Episode creation from text+vision features
- ✅ Memory attention patterns
- ✅ ~10-15% more parameters due to memory components
- ✅ Memory consolidation and retrieval mechanisms

**WITHOUT Episodic Memory:**
- ❌ No memory components
- ✅ Direct fusion of text+vision → decoder
- ✅ Fewer parameters (baseline)
- ✅ Same BitNet quantization and DiNOv3 embeddings
- ❌ No long-term cross-modal associations

### Expected Performance Differences:
- **Cross-modal similarity**: Memory model should show higher similarity scores
- **Text generation quality**: Memory model should produce more contextually relevant text
- **Training efficiency**: No-memory model trains slightly faster
- **Parameter count**: Memory model has more parameters but better multimodal understanding

## 💾 Training Variations by Dataset Size

### COCO Only Training (Fastest)
```bash
# Download COCO only
python download_multimodal_data.py --coco-only

# Train WITH memory
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml
# Expected: 2-4 hours on RTX 4090

# Train WITHOUT memory  
python train_100M_tokens.py --config configs/bitmar_without_memory.yaml
# Expected: 1.5-3 hours on RTX 4090
```

### Localized Narratives Only Training (Balanced)
```bash
# Download Localized Narratives only  
python download_multimodal_data.py --localized-narratives-only

# Train WITH memory
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml
# Expected: 6-10 hours on RTX 4090

# Train WITHOUT memory
python train_100M_tokens.py --config configs/bitmar_without_memory.yaml
# Expected: 5-8 hours on RTX 4090
```

### Full Dataset Training (Maximum Performance)
```bash
# Download both datasets
python download_multimodal_data.py --both

# Train WITH memory
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml
# Expected: 10-15 hours on RTX 4090

# Train WITHOUT memory
python train_100M_tokens.py --config configs/bitmar_without_memory.yaml
# Expected: 8-12 hours on RTX 4090
```

## 📈 Expected Training Statistics by Configuration

### WITH Episodic Memory Model:
- **Architecture**: BitNet + DiNOv3 + Episodic Memory (32 slots, 128-dim)
- **Parameters**: ~15-20% more than baseline
- **Training time**: ~20-30% longer due to memory operations
- **Expected benefits**: Better cross-modal understanding, contextual associations

### WITHOUT Episodic Memory Model (Baseline):
- **Architecture**: BitNet + DiNOv3 + Direct Fusion
- **Parameters**: Baseline parameter count
- **Training time**: Faster (no memory operations)
- **Purpose**: Ablation baseline to measure memory contribution

## 🎛️ Advanced Configuration for Ablation Study

### Memory Configuration (bitmar_with_memory.yaml):
```yaml
model:
  use_episodic_memory: true        # ENABLED
  memory_size: 32                  # 32 memory slots
  episode_dim: 128                 # 128-dim episodes
  memory_alpha: 0.2               # Memory update rate
  direct_writing: true            # Direct memory writing

output:
  checkpoint_dir: "checkpoints_with_memory"
  log_dir: "logs_with_memory"

wandb:
  project: "bitmar-ablation-with-memory"
  
huggingface_hub:
  repo_id: "euhidaman/bitmar-with-memory-ablation"
```

### No Memory Configuration (bitmar_without_memory.yaml):
```yaml
model:
  use_episodic_memory: false       # DISABLED for ablation
  # memory_size: commented out    # No memory parameters
  # episode_dim: commented out
  # memory_alpha: commented out

output:
  checkpoint_dir: "checkpoints_without_memory"
  log_dir: "logs_without_memory"

wandb:
  project: "bitmar-ablation-without-memory"
  
huggingface_hub:
  repo_id: "euhidaman/bitmar-without-memory-ablation"
```

## 🔧 Troubleshooting Ablation Study

### If Memory Model Fails:
```bash
# Check if memory components are properly initialized
python -c "
from src.model import create_bitmar_model
import yaml
with open('configs/bitmar_with_memory.yaml', 'r') as f:
    config = yaml.safe_load(f)
model = create_bitmar_model(config['model'])
print(f'Memory enabled: {model.use_episodic_memory}')
print(f'Memory slots: {getattr(model.memory, \"memory_size\", \"None\")}')
"
```

### If No-Memory Model Fails:
```bash
# Check if direct fusion is properly configured
python -c "
from src.model import create_bitmar_model
import yaml
with open('configs/bitmar_without_memory.yaml', 'r') as f:
    config = yaml.safe_load(f)
model = create_bitmar_model(config['model'])
print(f'Memory enabled: {model.use_episodic_memory}')
print(f'Direct fusion: {hasattr(model, \"direct_fusion_proj\")}')
"
```

### Memory vs GPU Issues:
```bash
# Reduce batch size for memory-constrained GPUs
# Edit configs/bitmar_with_memory.yaml or bitmar_without_memory.yaml:
data:
  batch_size: 32  # or 16 for 8GB GPUs

# Enable gradient checkpointing
memory_optimization:
  use_gradient_checkpointing: true
  use_fp16: true
```

## 📊 Monitoring Both Training Runs

### Weights & Biases Dashboards:
- **With Memory**: `https://wandb.ai/[username]/bitmar-ablation-with-memory`
- **Without Memory**: `https://wandb.ai/[username]/bitmar-ablation-without-memory`

### Local Monitoring Commands:
```bash
# Monitor WITH memory training
tail -f logs_with_memory/training.log

# Monitor WITHOUT memory training (separate terminal)
tail -f logs_without_memory/training.log

# Compare GPU usage
watch -n 1 nvidia-smi

# Check checkpoint sizes
ls -lh checkpoints_with_memory/
ls -lh checkpoints_without_memory/

# Compare parameter counts
du -sh checkpoints_with_memory/
du -sh checkpoints_without_memory/
```

## 📁 Output Directory Structure for Ablation Study

After running both configurations:

```
BitGen/
├── data/                                    # Shared dataset
│   ├── all_captions.json                   # Unified captions
│   ├── localized_narratives/               # LN annotations (if downloaded)
│   └── coco/                              # COCO annotations (if downloaded)
├── checkpoints_with_memory/                # WITH memory checkpoints
│   ├── latest_checkpoint.pt
│   ├── checkpoint_epoch_X_step_Y.pt
│   └── memory_exports/
├── checkpoints_without_memory/             # WITHOUT memory checkpoints
│   ├── latest_checkpoint.pt
│   ├── checkpoint_epoch_X_step_Y.pt
│   └── (no memory exports)
├── logs_with_memory/                       # WITH memory logs
├── logs_without_memory/                    # WITHOUT memory logs
├── attention_with_memory/                  # WITH memory attention analysis
├── attention_without_memory/               # WITHOUT memory attention analysis
├── memory_with_memory/                     # Memory visualizations (only for memory model)
├── results_with_memory/                    # WITH memory results
├── results_without_memory/                 # WITHOUT memory results
└── training.log                           # Current training log
```

## 🚨 System Requirements by Configuration

### WITH Episodic Memory Model:
- **GPU**: 8GB VRAM minimum (RTX 3070 or better) - needs memory for 32 slots
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space (larger checkpoints due to memory)
- **Training time**: +20-30% longer than no-memory model

### WITHOUT Episodic Memory Model (Baseline):
- **GPU**: 6GB VRAM minimum (RTX 3060 or better) - lighter memory usage
- **RAM**: 12GB system RAM
- **Storage**: 15GB free space (smaller checkpoints)
- **Training time**: Baseline timing (faster)

## 🎯 Complete Ablation Study Workflow

### Phase 1: Environment Setup
```bash
# 1. Clone and setup environment
git clone https://github.com/euhidaman/BitGen.git
cd BitGen
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Phase 2: Dataset Preparation
```bash
# Choose your dataset size (recommend --both for complete ablation study)
python download_multimodal_data.py --both --data_dir ./data

# Verify dataset
python -c "
import json
with open('./data/all_captions.json', 'r') as f:
    captions = json.load(f)
print(f'✅ Dataset ready: {len(captions):,} samples')
"
```

### Phase 3: Run Ablation Study (Both Models)

#### Train Model WITH Episodic Memory:
```bash
# Start training WITH episodic memory
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml

# Monitor training
tail -f logs_with_memory/training.log

# Check memory utilization
python -c "
import torch
checkpoint = torch.load('checkpoints_with_memory/latest_checkpoint.pt', map_location='cpu')
print('WITH Memory Model:')
print(f'  Parameters: {sum(p.numel() for p in checkpoint[\"model_state_dict\"].values()):,}')
print(f'  Memory slots: 32')
print(f'  Episode dimensions: 128')
"
```

#### Train Model WITHOUT Episodic Memory:
```bash
# Start training WITHOUT episodic memory (baseline)
python train_100M_tokens.py --config configs/bitmar_without_memory.yaml

# Monitor training
tail -f logs_without_memory/training.log

# Check parameter difference
python -c "
import torch
checkpoint = torch.load('checkpoints_without_memory/latest_checkpoint.pt', map_location='cpu')
print('WITHOUT Memory Model (Baseline):')
print(f'  Parameters: {sum(p.numel() for p in checkpoint[\"model_state_dict\"].values()):,}')
print(f'  Memory slots: 0 (disabled)')
print(f'  Direct fusion: enabled')
"
```

### Phase 4: Compare Results

#### Parameter Count Comparison:
```bash
# Compare model sizes
python -c "
import torch
from pathlib import Path

# Load WITH memory checkpoint
if Path('checkpoints_with_memory/latest_checkpoint.pt').exists():
    with_memory = torch.load('checkpoints_with_memory/latest_checkpoint.pt', map_location='cpu')
    with_params = sum(p.numel() for p in with_memory['model_state_dict'].values())
    print(f'WITH Memory: {with_params:,} parameters')
else:
    print('WITH Memory checkpoint not found')

# Load WITHOUT memory checkpoint  
if Path('checkpoints_without_memory/latest_checkpoint.pt').exists():
    without_memory = torch.load('checkpoints_without_memory/latest_checkpoint.pt', map_location='cpu')
    without_params = sum(p.numel() for p in without_memory['model_state_dict'].values())
    print(f'WITHOUT Memory: {without_params:,} parameters')
    
    if 'with_params' in locals():
        difference = with_params - without_params
        percentage = (difference / without_params) * 100
        print(f'Memory overhead: {difference:,} parameters ({percentage:.1f}% increase)')
else:
    print('WITHOUT Memory checkpoint not found')
"
```

#### Training Metrics Comparison:
- **WandB Dashboards**: Compare projects `bitmar-ablation-with-memory` vs `bitmar-ablation-without-memory`
- **Cross-modal similarity**: Memory model should show higher sustained similarity
- **Loss convergence**: Compare convergence patterns between both models
- **Training time**: Memory model will be slower but should show better multimodal understanding

## 🎯 Quick Start Commands Summary

```bash
# 1. Setup environment
git clone https://github.com/euhidaman/BitGen.git && cd BitGen
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Download dataset (choose size)
python download_multimodal_data.py --both        # Full dataset (1.78M samples)

# 3. Run ablation study (both models)
python train_100M_tokens.py --config configs/bitmar_with_memory.yaml     # WITH memory
python train_100M_tokens.py --config configs/bitmar_without_memory.yaml  # WITHOUT memory

# 4. Monitor training
tail -f logs_with_memory/training.log      # Memory model
tail -f logs_without_memory/training.log   # Baseline model
```

## 📋 Ablation Study Checklist

- [ ] Environment setup completed
- [ ] Dataset downloaded and verified
- [ ] WITH memory model training started
- [ ] WITHOUT memory model training started
- [ ] Both WandB projects monitoring
- [ ] Both HuggingFace repos receiving uploads
- [ ] Performance comparison metrics collected
- [ ] Results documented for paper

**Result**: You'll have two trained models to demonstrate that episodic memory improves multimodal understanding compared to direct fusion baseline!
