# BitMar: Unlimited Vision-Language Episodic Memory Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** designed for unlimited multimodal learning. It combines BitNet-quantized text processing, Facebook's **DINOv3-Large** vision encoding, and episodic memory mechanisms to achieve efficient multimodal understanding without dataset constraints.

## 🧠 **WHAT'S HAPPENING - Complete Training Breakdown**

### **🏗️ Architecture Overview**

BitMar is a sophisticated multimodal AI system that learns to understand both images and text simultaneously. Here's what's happening inside:

```
📸 IMAGES → DINOv3-Large → 1024D Features → Compression → 192D Vision
📝 TEXT → BitNet Encoder → 192D Text Features
🧠 MEMORY → 64 Episodic Slots → Cross-modal Learning
🔄 FUSION → Multi-head Attention → Understanding
📖 OUTPUT → Text Generation → Captions/Responses
```

### **🔄 Training Process Step-by-Step**

#### **Step 1: Data Preparation**
- **Image Processing**: DINOv3-Large extracts 1024-dimensional features from 224×224 images
- **Text Processing**: GPT-2 tokenizer converts captions to tokens
- **Pairing**: Perfect alignment between images and corresponding text descriptions
- **Batching**: Groups of 48 image-caption pairs processed together

#### **Step 2: Feature Extraction & Compression**
```python
# What happens to each image:
image (224×224×3) → DINOv3-Large → features (1024D) → compression → vision_latent (192D)

# What happens to each text:
"A dog playing in park" → tokenizer → [2, 345, 1776, 287, 3952] → BitNet → text_features (192D)
```

#### **Step 3: Episodic Memory System**
- **64 Memory Slots**: Each slot stores cross-modal patterns
- **Memory Writing**: New image-text associations update memory
- **Memory Reading**: Retrieves relevant past experiences
- **Adaptive Learning**: Memory evolves based on similarity patterns

#### **Step 4: Cross-Modal Fusion**
```python
# Fusion process:
vision_latent (192D) + text_features (192D) → Multi-head Attention → fused_representation (192D)
# This is where the magic happens - the model learns connections between images and text
```

#### **Step 5: Loss Calculation & Learning**
- **Cross-modal Loss**: How well vision and text align (cosine similarity)
- **Generation Loss**: How well the model generates accurate captions
- **Memory Loss**: How efficiently memory is used
- **Total Loss**: Weighted combination guides learning

### **🎯 What Makes BitMar Special**

#### **1. BitNet Quantization**
- **1.58-bit weights**: Extremely memory efficient
- **Full precision during training**: Quality maintained
- **Quantization-aware training**: Learns to work with reduced precision

#### **2. DINOv3-Large Vision**
- **SUPERIOR to DINOv2**: Trained on 1.689 billion images
- **SafeTensors Format**: Modern, secure, fast loading
- **1024D Features**: Rich visual representations
- **Learned Compression**: Intelligently reduces to 192D

#### **3. Episodic Memory**
- **64 Slots**: Much larger than typical transformers
- **Cross-modal Storage**: Remembers image-text relationships
- **Adaptive Access**: Smart retrieval based on current input
- **Memory Evolution**: Continuously improves over time

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

# Install additional dependencies for DINOv3-Large
pip install transformers torch torchvision timm safetensors
```

### **Step 2: Download Facebook DINOv3-Large Model**

```bash
# Download and setup DINOv3-Large (1024-dim features, SUPERIOR to DINOv2)
python download_dinov3_large.py

# Test the model (optional but recommended)
python test_dinov3_large.py
```

**What's happening here:**
- Downloads `model.safetensors` (secure format)
- Downloads `config.json` and `preprocessor_config.json`
- Verifies 1024-dimensional output
- Tests batch processing capabilities

### **Step 3: Start Unlimited Multimodal Training**

```bash
# Train with unlimited datasets and Facebook DINOv3-Large
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cuda:0

# Alternative: Train on CPU (slower)
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cpu
```

**What's happening during training:**

#### **Every Training Step:**
1. **Batch Loading**: 48 image-caption pairs loaded
2. **Vision Processing**: DINOv3-Large extracts features
3. **Text Processing**: BitNet encoder processes captions
4. **Memory Access**: Retrieves relevant past experiences
5. **Cross-modal Fusion**: Learns image-text relationships
6. **Loss Calculation**: Measures alignment quality
7. **Backpropagation**: Updates all parameters
8. **Memory Update**: Stores new patterns in memory slots

#### **Every Epoch:**
1. **Full Dataset Pass**: Processes all available image-caption pairs
2. **Validation**: Tests on held-out data
3. **Checkpoint Save**: Preserves best model weights
4. **Metric Logging**: Records loss, similarity, memory usage
5. **Early Stopping Check**: Stops if no improvement

#### **Throughout Training:**
- **Adaptive Controller**: Monitors cross-modal similarity
- **Memory Visualization**: Tracks memory slot evolution
- **Attention Analysis**: Studies what model focuses on
- **FLOPS Tracking**: Measures computational efficiency
- **Carbon Tracking**: Monitors environmental impact

### **Step 4: Monitor Training**

```bash
# Check training logs
tail -f training_unlimited.log

# View checkpoints
ls -la checkpoints_unlimited/

# Check carbon emissions (if enabled)
ls -la carbon_logs_unlimited/
```

**What you'll see in logs:**
```
INFO - Epoch 1 completed:
INFO -   • Loss: 0.2847
INFO -   • Cross-modal similarity: 0.7234
INFO -   • Samples processed: 15,360
INFO -   • Best similarity so far: 0.7234
INFO - ✅ Memory diversity score: 0.8456
INFO - 🔢 FLOPS: 2.34T per batch
INFO - 🌱 Carbon emissions: 0.0045 kg CO2
```

### **Step 5: Convert Model for HuggingFace**

```bash
# Convert trained checkpoint to HuggingFace format
python bitmar_hf_adapter.py \
  --checkpoint_path checkpoints_unlimited/best_checkpoint.pt \
  --output_dir hf_model_unlimited \
  --device cuda:0 \
  --download_dinov3
```

## 📊 **DETAILED ARCHITECTURE EXPLANATION**

### **🔬 BitNet Text Processing**

```python
# What happens to text:
Input: "A golden retriever playing fetch in a sunny park"

1. Tokenization:
   → [32, 3585, 26, 220, 70, 10829, 11687, 2118, 4645, 287, 220, 64, 6290, 3952]

2. BitNet Encoding (6 layers):
   Layer 1: [tokens] → self-attention → feed-forward → residual
   Layer 2: [hidden] → self-attention → feed-forward → residual
   ...
   Layer 6: [hidden] → self-attention → feed-forward → [192D output]

3. Quantization-Aware Training:
   - Forward pass: Full precision (FP16/32)
   - Backward pass: Gradients computed normally
   - Weight updates: Prepared for 1.58-bit quantization
```

### **🔬 DINOv3-Large Vision Processing**

```python
# What happens to images:
Input: Image (224×224×3 pixels)

1. Patch Extraction:
   224×224 → 14×14 patches of 16×16 pixels → 196 patches

2. DINOv3-Large Processing:
   Patch embedding → Transformer (24 layers) → [196×1024] features

3. Spatial Pooling:
   [196×1024] → Global Average Pool → [1024D] global features

4. Learned Compression:
   [1024D] → Linear(1024→384) → ReLU → Linear(384→192) → [192D] vision_latent
```

### **🔬 Episodic Memory System**

```python
# Memory mechanics:
Memory: [64 slots × 192 dimensions] = [64×192] matrix

1. Memory Writing:
   new_pattern = fusion_output  # Current image-text understanding
   similarity = cosine_similarity(new_pattern, memory_slots)
   update_slot = argmax(similarity) if max(similarity) > threshold else least_used_slot
   memory_slots[update_slot] = alpha * new_pattern + (1-alpha) * memory_slots[update_slot]

2. Memory Reading:
   query = current_input_features
   attention_weights = softmax(query @ memory_slots.T)
   retrieved_memory = attention_weights @ memory_slots
   
3. Memory Integration:
   enhanced_features = input_features + retrieved_memory
```

### **🔬 Cross-Modal Fusion**

```python
# How vision and text are combined:
vision_latent: [batch_size, 192]  # From compressed DINOv3
text_features: [batch_size, seq_len, 192]  # From BitNet

1. Multi-head Attention:
   Q = text_features @ W_q  # Queries from text
   K = vision_latent @ W_k  # Keys from vision
   V = vision_latent @ W_v  # Values from vision
   
2. Attention Computation:
   attention_scores = Q @ K.T / sqrt(192)
   attention_weights = softmax(attention_scores)
   attended_vision = attention_weights @ V
   
3. Fusion:
   fused_features = text_features + attended_vision
```

## 📋 **ALL IMPORTANT COMMANDS**

### **Model Training Commands**

```bash
# Basic unlimited training
python train_unlimited.py --config configs/bitmar_unlimited.yaml

# Training with specific GPU
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cuda:0

# Resume from checkpoint
python train_unlimited.py --config configs/bitmar_unlimited.yaml --resume checkpoints_unlimited/latest_checkpoint.pt

# Debug mode (verbose logging)
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cuda:0 --debug
```

### **DINOv3-Large Setup Commands**

```bash
# Download Facebook DINOv3-Large
python download_dinov3_large.py

# Test DINOv3-Large functionality
python test_dinov3_large.py

# Verify SafeTensors format
python test_dinov3_large.py --test safetensors

# Test BitMar compatibility
python test_dinov3_large.py --test compatibility
```

### **Model Conversion Commands**

```bash
# Convert to HuggingFace format
python bitmar_hf_adapter.py \
  --checkpoint_path checkpoints_unlimited/best_checkpoint.pt \
  --output_dir hf_model_unlimited

# Convert with DINOv3 pre-download
python bitmar_hf_adapter.py \
  --checkpoint_path checkpoints_unlimited/best_checkpoint.pt \
  --output_dir hf_model_unlimited \
  --download_dinov3

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

# Verify DINOv2 features (legacy)
python -c "
import numpy as np
features1 = np.load('../babylm_dataset/cc_3M_dino_v2_states_1of2.npy', mmap_mode='r')
features2 = np.load('../babylm_dataset/cc_3M_dino_v2_states_2of2.npy', mmap_mode='r')
print(f'Features1 shape: {features1.shape}')
print(f'Features2 shape: {features2.shape}')
"

# Check captions
python -c "
import json
with open('../babylm_dataset/cc_3M_captions.json', 'r') as f:
    captions = json.load(f)
print(f'Total captions: {len(captions)}')
print(f'Sample: {list(captions.items())[0]}')
"
```

### **Monitoring Commands**

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check training progress
tail -f training_unlimited.log

# Monitor memory usage
python -c "
import psutil
import torch
print(f'RAM: {psutil.virtual_memory().percent}% used')
if torch.cuda.is_available():
    print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB')
"

# Check disk space
df -h  # Linux
# Or on Windows:
dir D:\ /-c
```

### **Troubleshooting Commands**

```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Verify DINOv3 installation
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', trust_remote_code=True); print('✅ DINOv3-Large available')"

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

# Test SafeTensors loading
python -c "
from transformers import AutoModel
import torch
model = AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', use_safetensors=True, trust_remote_code=True)
print('✅ SafeTensors loading successful')
print(f'Model type: {type(model)}')
"
```

## 🔧 **Configuration Files Explained**

### **Main Config: `configs/bitmar_unlimited.yaml`**

```yaml
# Key sections and what they do:

model:
  # Text processing with BitNet quantization
  text_encoder_dim: 192      # Feature dimension for text
  text_encoder_layers: 6     # Depth of text understanding
  
  # Vision processing with DINOv3-Large  
  vision_encoder_name: "facebook/dinov3-vitl16-pretrain-lvd1689m"
  vision_encoder_dim: 1024   # Raw DINOv3 features
  vision_latent_size: 192    # Compressed for efficiency
  
  # Episodic memory system
  memory_size: 64           # Number of memory slots
  episode_dim: 192          # Memory slot dimension
  memory_alpha: 0.15        # Memory update rate

data:
  # Training configuration
  batch_size: 48            # Images processed together
  num_workers: 8            # Parallel data loading
  dataset_type: "multimodal_unlimited"  # No token limits
  
  # Data quality
  ensure_alignment: true    # Perfect image-caption pairing
  validate_alignment: true  # Double-check alignment
  never_break_pairs: true   # Keep images with their captions

training:
  # Optimization
  max_epochs: 50            # Training duration
  learning_rate: 0.0003     # How fast to learn
  weight_decay: 0.015       # Regularization
  
  # Loss weights
  cross_modal_loss_weight: 2.0      # Image-text alignment importance
  text_generation_loss_weight: 1.0  # Caption generation importance
  memory_regularization_weight: 0.08 # Memory efficiency
```

## 📊 **Key Features**

### **✅ Unlimited Training**
- **No Token Constraints**: Train on unlimited multimodal datasets
- **No BabyLM Limitations**: Removed all 100M token restrictions
- **No train_50M Dependencies**: Pure multimodal focus
- **Streaming Data**: Can handle datasets larger than memory

### **🔥 Facebook DINOv3-Large**
- **SUPERIOR Vision**: 1024-dimensional vision features (better than DINOv2)
- **Latest Model**: `facebook/dinov3-vitl16-pretrain-lvd1689m` from HuggingFace
- **SafeTensors Format**: Modern, secure, fast loading
- **Enhanced Understanding**: Superior image-text alignment and representation quality
- **LVD-1689M Training**: Trained on 1.689 billion images for better generalization

### **🧠 Enhanced Architecture**
- **BitNet Quantization**: 1.58-bit quantized weights for efficiency
- **64 Memory Slots**: Increased from 32 for larger datasets
- **192-dim Text Encoder**: Enhanced from 128-dim
- **6-Layer Processing**: Deeper understanding capabilities
- **Learned Compression**: Intelligent 1024D→192D vision compression

### **📈 Multi-dataset Support**
- **Conceptual Captions**: 3M+ high-quality image-text pairs
- **Visual Genome**: Rich scene descriptions and object relationships
- **COCO Captions**: Detailed image descriptions (5 per image)
- **Custom Datasets**: Support for your own image-caption data

### **🔍 Advanced Monitoring**
- **FLOPS Tracking**: Computational efficiency measurement
- **Memory Visualization**: Real-time episodic memory analysis
- **Attention Analysis**: Study what the model focuses on
- **Carbon Tracking**: Environmental impact monitoring
- **Cross-modal Metrics**: Image-text alignment quality

## 🎯 **What to Do Next - Step by Step Guide**

### **STEP 1: Initial Setup (5 minutes)**
```bash
cd D:\BabyLM\BitGen
venv\Scripts\activate
pip install -r requirements.txt
pip install transformers torch torchvision timm safetensors
```

### **STEP 2: Download DINOv3-Large (2-3 minutes)**
```bash
python download_dinov3_large.py
# Wait for "✅ DINOv3-Large setup completed successfully!"
# This downloads model.safetensors (~1.5GB)
```

### **STEP 3: Test Setup (1 minute)**
```bash
python test_dinov3_large.py
# Should show "✅ Correct DINOv3-Large feature dimension (1024)"
# Tests SafeTensors format and BitMar compatibility
```

### **STEP 4: Start Training (Ongoing)**
```bash
python train_unlimited.py --config configs/bitmar_unlimited.yaml --device cuda:0
# Training will start automatically with unlimited datasets
```

**What happens during training:**
1. **Model Initialization**: Creates BitMar with DINOv3-Large vision
2. **Data Loading**: Streams image-caption pairs (no memory limits)
3. **Feature Extraction**: DINOv3 processes images, BitNet processes text
4. **Memory Learning**: 64 episodic slots learn cross-modal patterns
5. **Loss Optimization**: Improves image-text alignment and generation
6. **Checkpointing**: Saves best models every epoch
7. **Monitoring**: Tracks metrics, memory usage, computational efficiency

### **STEP 5: Monitor Progress**

**Real-time monitoring:**
- Watch the console output for loss and similarity metrics
- Training logs are saved to `training_unlimited.log`
- Checkpoints saved in `checkpoints_unlimited/`
- WandB dashboard (if configured) shows detailed metrics

**Key metrics to watch:**
- **Loss**: Should decrease over time (target: <0.2)
- **Cross-modal Similarity**: Should increase (target: >0.85)
- **Memory Diversity**: Should stabilize around 0.8-0.9
- **GPU Memory**: Should remain stable (check for leaks)

### **STEP 6: Model Conversion (After training)**
```bash
python bitmar_hf_adapter.py \
  --checkpoint_path checkpoints_unlimited/best_checkpoint.pt \
  --output_dir hf_model_unlimited
```

**What this creates:**
- `pytorch_model.bin`: Model weights
- `config.json`: Model configuration
- `tokenizer.json`: Text processing configuration
- `README.md`: Model documentation

### **STEP 7: Evaluation**
```bash
python evaluate_bitmar_2024.py --model_path hf_model_unlimited
python evaluate_bitmar_2025.py --model_path hf_model_unlimited
```

**Evaluation tests:**
- **Image Captioning**: Generate descriptions for images
- **Cross-modal Retrieval**: Find images matching text queries
- **Zero-shot Classification**: Classify images without training
- **Memory Efficiency**: Test episodic memory performance

## 🚨 **Important Notes**

### **GPU Requirements**
- **Minimum**: 8GB VRAM for training (batch_size=16)
- **Recommended**: 16GB+ VRAM for optimal performance (batch_size=48)
- **Memory Breakdown**:
  - DINOv3-Large model: ~1.2GB
  - BitMar model: ~120MB
  - Training overhead: ~2-6GB (depends on batch size)
  - Activation caching: ~1-3GB

### **Storage Requirements**
- **Model checkpoints**: ~120MB per checkpoint
- **DINOv3-Large cache**: ~1.5GB (larger than DINOv2)
- **Training logs**: Variable based on training length
- **Dataset cache**: Depends on data sources used

### **Memory Management**
- Training uses gradient checkpointing for memory efficiency
- Mixed precision (FP16) enabled by default
- Automatic GPU memory clearing between epochs
- SafeTensors format provides memory-efficient model loading

### **Training Time Estimates**
- **Small dataset (10K pairs)**: 2-4 hours on RTX 3080
- **Medium dataset (100K pairs)**: 1-2 days on RTX 3080
- **Large dataset (1M+ pairs)**: 1-2 weeks on RTX 3080
- **Scaling**: Roughly linear with dataset size

## 🆘 **Troubleshooting**

### **Common Issues & Solutions**

#### **CUDA Out of Memory**
```bash
# Solution 1: Reduce batch size in config
# Edit configs/bitmar_unlimited.yaml:
# batch_size: 48 → batch_size: 24 or 16

# Solution 2: Enable gradient checkpointing (already enabled)
# Solution 3: Use smaller model variant
```

#### **DINOv3 Not Found**
```bash
python download_dinov3_large.py
# Re-download the model
# Check internet connection
# Verify HuggingFace access
```

#### **SafeTensors Loading Issues**
```bash
pip install --upgrade transformers safetensors
# Upgrade to latest versions
python test_dinov3_large.py --test safetensors
# Verify SafeTensors support
```

#### **Training Stalls or Crashes**
```bash
# Check GPU memory
nvidia-smi
# Look for memory leaks or full utilization

# Check disk space
df -h  # Linux
dir D:\ /-c  # Windows

# Verify data integrity
python -c "
import json
with open('../babylm_dataset/cc_3M_captions.json', 'r') as f:
    captions = json.load(f)
print('Captions loaded successfully')
"
```

#### **Slow Training Speed**
```bash
# Check data loading bottleneck
# Increase num_workers in config
# Enable persistent_workers
# Use SSD storage for datasets
# Check CPU utilization
```

#### **Memory Visualization Errors**
```bash
# Disable memory visualization if causing issues
# Comment out memory_viz in train_unlimited.py
# Or set wandb logging to False
```

## 📞 **Support & Next Steps**

### **Immediate Actions:**
1. **Start with Step 1-4** above for immediate training
2. **Monitor training** for first few epochs to ensure stability
3. **Check logs regularly** for any warning messages
4. **Adjust batch size** if memory issues occur

### **After Training Completes:**
1. **Convert model** to HuggingFace format
2. **Run comprehensive evaluation** to test performance
3. **Upload to HuggingFace Hub** if desired
4. **Share results** with the community

### **Optimization Tips:**
1. **Dataset Quality**: Higher quality image-caption pairs = better results
2. **Memory Tuning**: Adjust memory_size based on dataset complexity
3. **Learning Rate**: Start with 0.0003, adjust based on convergence
4. **Early Stopping**: Use patience=5 to avoid overtraining

## 🎉 **Success Indicators**

You'll know everything is working when you see:

### **Setup Phase:**
- ✅ "DINOv3-Large setup completed successfully!"
- ✅ "SafeTensors format ready for training!"
- ✅ "BitMar Unlimited Training" starting message

### **Training Phase:**
- ✅ Loss decreasing steadily (0.8 → 0.5 → 0.3 → 0.2)
- ✅ Cross-modal similarity increasing (0.4 → 0.6 → 0.7 → 0.8+)
- ✅ Memory diversity score stable around 0.8-0.9
- ✅ Regular checkpoint saves without errors
- ✅ GPU memory usage stable (no leaks)

### **Completion Phase:**
- ✅ "Training completed successfully!"
- ✅ Final similarity score >0.8
- ✅ Model conversion successful
- ✅ Evaluation metrics showing good performance

## 🔬 **Technical Deep Dive**

### **BitNet Quantization Process**
```python
# During training:
def forward_pass():
    # 1. Forward with full precision
    hidden = F.linear(input, weight.float(), bias)
    
    # 2. Simulate quantization for gradients
    weight_quantized = quantize_to_1_58_bit(weight)
    
    # 3. Compute outputs
    return hidden

def quantize_to_1_58_bit(weight):
    # BitNet quantization: -1, 0, +1 (with some 0.5 values)
    scale = weight.abs().mean()
    quantized = torch.round(weight / scale).clamp(-1, 1)
    return quantized * scale
```

### **Episodic Memory Mechanics**
```python
class EpisodicMemory:
    def __init__(self, memory_size=64, episode_dim=192):
        self.memory = nn.Parameter(torch.randn(memory_size, episode_dim))
        self.usage_count = torch.zeros(memory_size)
        self.alpha = 0.15
    
    def write(self, new_episode):
        # Find most similar slot
        similarities = F.cosine_similarity(new_episode, self.memory)
        best_slot = torch.argmax(similarities)
        
        # Update with exponential moving average
        self.memory[best_slot] = (
            self.alpha * new_episode + 
            (1 - self.alpha) * self.memory[best_slot]
        )
        self.usage_count[best_slot] += 1
    
    def read(self, query):
        # Attention-based retrieval
        attention_weights = F.softmax(query @ self.memory.T, dim=-1)
        retrieved = attention_weights @ self.memory
        return retrieved
```

### **Cross-Modal Loss Function**
```python
def cross_modal_loss(vision_features, text_features):
    # Normalize features
    vision_norm = F.normalize(vision_features, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)
    
    # Compute similarity matrix
    similarity_matrix = vision_norm @ text_norm.T
    
    # Contrastive loss (InfoNCE)
    labels = torch.arange(len(vision_features))
    loss = F.cross_entropy(similarity_matrix / temperature, labels)
    
    return loss
```

---

**🚀 Ready to start unlimited multimodal training with Facebook DINOv3-Large and advanced episodic memory!** 

**🎯 BitMar combines the best of:**
- **BitNet quantization** for efficiency
- **DINOv3-Large vision** for superior understanding  
- **Episodic memory** for cross-modal learning
- **SafeTensors** for secure, fast model loading
- **Unlimited datasets** for maximum performance

**Start training now and watch your model learn to understand images and text together!** 🧠✨
