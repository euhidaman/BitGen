# BitMar: Unlimited Vision-Language Episodic Memory Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** designed for unlimited multimodal learning. It combines BitNet-quantized text processing, Facebook's DINOv2-large vision encoding, and episodic memory mechanisms to achieve efficient multimodal understanding without dataset constraints.

## 🚀 **QUICK START - Step by Step Instructions**

### **Step 1: Environment Setup**

```bash
# Navigate to BitGen directory
cd D:\BabyLM\BitGen

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for DINOv2-large
pip install transformers torch torchvision timm
```

### **Step 2: Download Facebook DINOv2-large Model**

```bash
# Download and setup DINOv2-large (1024-dim features)
python download_dinov2_large.py

# Test the model (optional but recommended)
python test_dinov2_large.py
```

### **Step 3: Start Unlimited Multimodal Training**

```bash
# Train with unlimited datasets and Facebook DINOv2-large
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cuda:0

# Alternative: Train on CPU (slower)
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cpu

# Train with specific dataset sources only
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cuda:0
```

### **Step 4: Monitor Training**

```bash
# Check training logs
tail -f training_unlimited.log

# View checkpoints
ls -la checkpoints_unlimited/

# Check carbon emissions (if enabled)
ls -la carbon_logs_unlimited/
```

### **Step 5: Convert Model for HuggingFace**

```bash
# Convert trained checkpoint to HuggingFace format
python bitmar_hf_adapter.py \
  --checkpoint_path checkpoints_unlimited/best_checkpoint.pt \
  --output_dir hf_model_unlimited \
  --device cuda:0 \
  --download_dinov2
```

## 📋 **ALL IMPORTANT COMMANDS**

### **Model Training Commands**

```bash
# Basic unlimited training
python train_unlimited.py --config configs/bitmar_unlimited.yaml

# Training with specific GPU
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cuda:0

# Training with multiple GPUs (if supported)
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cuda:0

# Resume from checkpoint
python train_unlimited.py --config configs/bitmar_unlimited.yaml --resume checkpoints_unlimited/latest_checkpoint.pt
```

### **DINOv2-large Setup Commands**

```bash
# Download Facebook DINOv2-large
python download_dinov2_large.py

# Test DINOv2-large functionality
python test_dinov2_large.py

# Pre-download DINOv2 before training
python bitmar_hf_adapter.py --download_dinov2
```

### **Model Conversion Commands**

```bash
# Convert to HuggingFace format
python bitmar_hf_adapter.py \
  --checkpoint_path checkpoints_unlimited/best_checkpoint.pt \
  --output_dir hf_model_unlimited

# Convert with DINOv2 pre-download
python bitmar_hf_adapter.py \
  --checkpoint_path checkpoints_unlimited/best_checkpoint.pt \
  --output_dir hf_model_unlimited \
  --download_dinov2

# Convert specific checkpoint
python bitmar_hf_adapter.py \
  --checkpoint_path checkpoints_unlimited/checkpoint_epoch_10.pt \
  --output_dir hf_model_epoch_10
```

### **Evaluation Commands**

```bash
# Run 2024 evaluation
python evaluate_bitmar_2024.py --model_path hf_model_unlimited

# Run 2025 evaluation  
python evaluate_bitmar_2025.py --model_path hf_model_unlimited

# Download evaluation data first
python download_evaluation_data.py
```

### **Data Management Commands**

```bash
# Check dataset availability
ls -la ../babylm_dataset/

# Verify data files
python -c "
import numpy as np
features1 = np.load('../babylm_dataset/cc_3M_dino_v2_states_1of2.npy', mmap_mode='r')
features2 = np.load('../babylm_dataset/cc_3M_dino_v2_states_2of2.npy', mmap_mode='r')
print(f'Features1 shape: {features1.shape}')
print(f'Features2 shape: {features2.shape}')
"
```

### **Monitoring Commands**

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check training progress
tail -f training_unlimited.log

# Monitor disk space
df -h

# Check memory usage
free -h  # Linux
# Or on Windows: wmic OS get TotalVisibleMemorySize,FreePhysicalMemory
```

### **Troubleshooting Commands**

```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Verify transformers installation
python -c "from transformers import Dinov2Model; print('DINOv2 available')"

# Check model creation
python -c "
import sys
sys.path.append('src')
from src.model import create_bitmar_model
import yaml
with open('configs/bitmar_unlimited.yaml', 'r') as f:
    config = yaml.safe_load(f)
model = create_bitmar_model(config['model'])
print(f'Model created successfully with {sum(p.numel() for p in model.parameters())} parameters')
"
```

## 🔧 **Configuration Files**

### **Main Config: `configs/bitmar_unlimited.yaml`**
- Enhanced with Facebook DINOv2-large (1024-dim)
- No BabyLM constraints or token limits
- 64-slot episodic memory
- 192-dim text encoder with 6 layers
- Unlimited training epochs with early stopping

## 📊 **Key Features**

### **✅ Unlimited Training**
- **No Token Constraints**: Train on unlimited multimodal datasets
- **No BabyLM Limitations**: Removed all 100M token restrictions
- **No train_50M Dependencies**: Pure multimodal focus

### **🔥 Facebook DINOv2-large**
- **Superior Vision**: 1024-dimensional vision features
- **Official Model**: `facebook/dinov2-large` from HuggingFace
- **Enhanced Understanding**: Better image-text alignment

### **🧠 Enhanced Architecture**
- **BitNet Quantization**: 1.58-bit quantized weights for efficiency
- **64 Memory Slots**: Increased from 32 for larger datasets
- **192-dim Text Encoder**: Enhanced from 128-dim
- **6-Layer Processing**: Deeper understanding capabilities

### **📈 Multi-dataset Support**
- **Conceptual Captions**: High-quality image-text pairs
- **Visual Genome**: Rich scene descriptions
- **COCO Captions**: Detailed image descriptions
- **Custom Datasets**: Support for your own data

## 🎯 **What to Do Next - Step by Step Guide**

### **STEP 1: Initial Setup (5 minutes)**
```bash
cd D:\BabyLM\BitGen
venv\Scripts\activate
pip install -r requirements.txt
pip install transformers torch torchvision timm
```

### **STEP 2: Download DINOv2-large (2-3 minutes)**
```bash
python download_dinov2_large.py
# Wait for "✅ DINOv2-large setup completed successfully!"
```

### **STEP 3: Test Setup (1 minute)**
```bash
python test_dinov2_large.py
# Should show "✅ Correct DINOv2-large feature dimension (1024)"
```

### **STEP 4: Start Training (Ongoing)**
```bash
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cuda:0
# Training will start automatically with unlimited datasets
```

### **STEP 5: Monitor Progress**
- Watch the console output for loss and similarity metrics
- Training logs are saved to `training_unlimited.log`
- Checkpoints saved in `checkpoints_unlimited/`
- WandB dashboard (if configured) shows detailed metrics

### **STEP 6: Model Conversion (After training)**
```bash
python bitmar_hf_adapter.py \
  --checkpoint_path checkpoints_unlimited/best_checkpoint.pt \
  --output_dir hf_model_unlimited
```

### **STEP 7: Evaluation**
```bash
python evaluate_bitmar_2024.py --model_path hf_model_unlimited
python evaluate_bitmar_2025.py --model_path hf_model_unlimited
```

## 🚨 **Important Notes**

### **GPU Requirements**
- **Minimum**: 8GB VRAM for training
- **Recommended**: 16GB+ VRAM for optimal performance
- **CPU Training**: Possible but much slower

### **Storage Requirements**
- **Model checkpoints**: ~120MB per checkpoint
- **DINOv2-large cache**: ~1GB
- **Training logs**: Variable based on training length

### **Memory Management**
- Training uses gradient checkpointing for memory efficiency
- Mixed precision (FP16) enabled by default
- Automatic GPU memory clearing between epochs

## 🆘 **Troubleshooting**

### **Common Issues & Solutions**

#### **CUDA Out of Memory**
```bash
# Reduce batch size in config
# Change batch_size from 48 to 24 or 16
```

#### **DINOv2 Not Found**
```bash
python download_dinov2_large.py
# Re-download the model
```

#### **Import Errors**
```bash
pip install -r requirements.txt
pip install transformers torch torchvision timm
```

#### **Checkpoint Loading Issues**
```bash
# Use strict=False loading (automatically handled)
python train_unlimited.py --config configs/bitmar_unlimited.yaml --resume checkpoints_unlimited/latest_checkpoint.pt
```

## 📞 **Support & Next Steps**

1. **Start with Step 1-4** above for immediate training
2. **Monitor training** for first few epochs to ensure stability
3. **Convert model** after training completes
4. **Run evaluation** to test performance
5. **Upload to HuggingFace Hub** if desired

## 🎉 **Success Indicators**

You'll know everything is working when you see:
- ✅ "DINOv2-large setup completed successfully!"
- ✅ "BitMar Unlimited Training" starting message
- ✅ Loss decreasing and similarity increasing over epochs
- ✅ Regular checkpoint saves without errors

---

**Ready to start unlimited multimodal training with Facebook DINOv2-large!** 🚀
