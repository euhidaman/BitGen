# BitGen LoRA Architecture

## Overview

This branch implements a **LoRA-enhanced vision-language architecture** for BitGen Stage 1 training, combining:
- **DINOv2-base** (300M params, frozen) with **LoRA adapters** (~2M params, trainable)
- **Feature compression** (768→128 dims, ~300K params, trainable)
- **FIBER-style cross-modal fusion** (ITC + ITM losses)
- **Larimar episodic memory** (GPM with Bayesian updates)

**Total trainable parameters: ~32M** (under 50M goal!)

---

## Architecture Flow

```
INPUT:
  Any RGB Image (auto-resize to 224×224)  +  Any Text (tokenized)
    ↓                                          ↓

VISION ENCODING:
  DINOv2-base (frozen 300M)
    ↓
  LoRA adapters (trainable ~2M, rank=16)
    ↓
  768-dim patch features (256 patches)
    ↓
  Compression: 768→384→128 (trainable ~300K)
    ↓
  [batch, 256, 128]

TEXT ENCODING:
  Token + Position embeddings (trainable)
    ↓
  [batch, seq_len, 128]

CROSS-MODAL FUSION (FIBER):
  1. ITC Loss: Image-Text Contrastive
     - Queue-based (4096 samples)
     - Temperature-scaled similarity
     
  2. ITM Loss: Image-Text Matching
     - Hard negatives from ITC queue
     - Binary classification
     
  3. Fusion MLP: Concat + Project
     - [vision, text] → fused features
     
  Output: [batch, seq_len, 128]

EPISODIC MEMORY (Larimar GPM):
  Write Phase:
    - Store fused features
    - Bayesian posterior update
    - 32 memory slots
    
  Read Phase:
    - Address with current input
    - Retrieve similar patterns
    - Memory-augmented features
    
  KL Loss:
    - Regularize memory updates
    - Prevent collapse

LANGUAGE MODEL:
  Transformer layers (3 layers)
    ↓
  Memory-augmented decoding
    ↓
  Output logits: [batch, seq_len, 50257]
    ↓

OUTPUT:
  Generated text (captions, descriptions, answers)
```

---

## Training Strategy

### **Two-Phase Training**

#### **Phase 1: Warmup (5 epochs)**
- **LoRA adapters: FROZEN** ❄️
- **Train only**: Compression + FIBER + Language Model
- **Goal**: Establish stable vision-language alignment
- **Trainable params**: ~30M

**Why?** 
- Prevents LoRA from interfering with initial alignment
- Language model learns to use vision features first
- Stable gradient flow before fine-tuning vision

#### **Phase 2: Main Training (45 epochs)**
- **LoRA adapters: TRAINABLE** 🔥
- **Train**: LoRA + Compression + FIBER + Language Model
- **Goal**: Adapt vision to your specific data
- **Trainable params**: ~32M

**Why?**
- Vision adapts to BabyLM datasets
- LoRA learns dataset-specific visual patterns
- End-to-end vision-language optimization

---

## Loss Functions

### **1. Image-Text Contrastive (ITC) Loss** - PRIMARY
```python
Weight: 1.0
Purpose: Learn vision-language alignment
Method: Queue-based contrastive (FIBER/ALBEF approach)
```

**How it works:**
1. Encode image → [B, D]
2. Encode text → [B, D]
3. Compute similarity with current batch + queue (4096 samples)
4. Maximize similarity for matching pairs
5. Minimize similarity for non-matching pairs

**Metrics:**
- `contrastive_loss`: Main alignment signal
- `acc_t2i`: Text→Image retrieval accuracy
- `acc_i2t`: Image→Text retrieval accuracy

### **2. Image-Text Matching (ITM) Loss**
```python
Weight: 0.5
Purpose: Fine-grained alignment with hard negatives
Method: Binary classification (match/no-match)
```

**How it works:**
1. Get hard negatives from ITC queue (multinomial sampling)
2. Positive pairs: (image[i], text[i]) → label=1
3. Negative pairs: (image[i], hard_text[j]) → label=0
4. Binary cross-entropy loss

**Metrics:**
- `itm_loss`: Hard negative matching loss
- `itm_acc`: Binary classification accuracy

### **3. Memory KL Loss**
```python
Weight: 0.1
Purpose: Regularize episodic memory
Method: KL divergence (prior vs posterior)
```

**How it works:**
1. Prior: Initial memory distribution
2. Posterior: Updated after writing
3. KL(posterior || prior)
4. Prevents memory collapse

**Metrics:**
- `memory_kl_loss`: Regularization term
- `memory_utilization`: Percentage of slots used
- `memory_quality`: Average memory quality score

### **4. Text Reconstruction Loss** (Optional)
```python
Weight: 0.0 (disabled in coarse-grained)
Purpose: Language modeling
Method: Next-token prediction
```

**Note:** Disabled during Stage 1 coarse-grained training (FIBER approach). Only enabled for fine-tuning tasks.

---

## Key Features

### **LoRA Adapters**
- **Rank**: 16 (balance between capacity and params)
- **Alpha**: 32 (LoRA scaling factor)
- **Target modules**: Query and Value in attention layers
- **Dropout**: 0.1
- **Trainable params**: ~2M

**Why LoRA?**
- Keeps base DINOv2 frozen (stable features)
- Only adapts attention mechanisms
- Efficient: 2M params vs 300M full fine-tuning
- Better generalization on small datasets

### **Feature Compression**
```python
Architecture:
  Linear(768 → 384)
  ReLU + Dropout(0.1)
  Linear(384 → 128)
  LayerNorm(128)

Trainable params: ~300K
```

**Why compress?**
- DINOv2 outputs 768 dims (too large for tiny model)
- Compression to 128 matches embed_dim
- Learned compression > fixed projection
- Similar to BitMar's approach

### **FIBER-style Queue**
```python
Queue size: 4096
Update: FIFO (dequeue oldest, enqueue newest)
Storage: Normalized features [D, 4096]
```

**Why queue?**
- More negative samples → better contrastive learning
- Hard negative mining (high similarity but wrong match)
- Momentum-based feature bank (like MoCo)
- FIBER/ALBEF proven approach

### **Larimar GPM Memory**
```python
Memory slots: 32
Memory dim: 128
Direct writing: True
Pseudoinverse steps: 3
Memory alpha: 0.2 (adaptation rate)
```

**How it works:**
1. **Write**: Store new experiences with Bayesian update
2. **Read**: Retrieve similar patterns from memory
3. **Address**: Solve for weights using pseudoinverse
4. **Update**: Posterior = prior + observation

**Why episodic memory?**
- Store reusable patterns across tasks
- Few-shot learning capability
- Continual learning without forgetting
- Larimar's proven approach

---

## Configuration

### **Default Settings** (Optimized for 8×A100 GPUs)

```python
# Model
embed_dim = 128
num_layers = 3
num_heads = 4
ffn_dim = 256
vision_embed_dim = 128

# Memory
memory_size = 32
memory_alpha = 0.2

# LoRA
lora_rank = 16
lora_alpha = 32
lora_warmup_epochs = 5

# Training
batch_size = 128  # per GPU
grad_accum_steps = 2  # effective batch = 256/GPU
learning_rate = 2e-4
num_epochs = 50
warmup_steps = 1000

# Losses
contrastive_weight = 1.0  # PRIMARY
itm_weight = 0.5
memory_kl_weight = 0.1
text_loss_weight = 0.0  # Disabled

# Queue
queue_size = 4096
temperature = 0.5  # Start high, prevents gradient explosion
```

---

## Usage

### **Installation**

```bash
# Install PEFT for LoRA
pip install peft

# Install other requirements
pip install -r requirements.txt
```

### **Training**

```bash
# Single GPU
python src/train_stage1_vision_language.py

# Multi-GPU (8×A100)
torchrun --nproc_per_node=8 src/train_stage1_vision_language.py \
  --use_ddp True \
  --num_gpus 8 \
  --batch_size 64 \
  --grad_accum_steps 2
```

### **Monitor Training**

Training logs show LoRA phase transitions:

```
🔧 LoRA Training Strategy:
   Phase 1 (Warmup, 5 epochs): LoRA FROZEN
   ➜ Train: Compression + FIBER + Language Model
   Phase 2 (Main, 45 epochs): LoRA TRAINABLE
   ➜ Train: LoRA + Compression + FIBER + Language Model
   Total trainable params with LoRA: ~32M (under 50M goal!)

✓ LoRA adapters FROZEN | Trainable params: 30,123,456

[Epoch 1-5 training with frozen LoRA...]

🔓 LoRA adapters UNFROZEN (Phase 2 started!)
   Trainable params: 32,456,789
   Vision can now adapt to your data!

[Epoch 6-50 training with trainable LoRA...]
```

---

## Advantages

### **vs Frozen DINOv2**
✅ Vision adapts to your data (via LoRA)  
✅ Better alignment with language model  
✅ Can learn new visual concepts  
✅ Only +2M params over frozen approach  

### **vs Fully Trainable DINOv2**
✅ 32M trainable vs 330M (10x fewer!)  
✅ Fits on smaller GPUs  
✅ Faster training (smaller gradients)  
✅ Less overfitting risk  
✅ More stable training  

### **vs BitMar Approach**
✅ Real-time image processing (no pre-extraction)  
✅ Can handle new images at inference  
✅ End-to-end vision-language learning  
✅ Similar param count (~32M trainable)  

---

## Expected Results

### **After Phase 1 (Warmup)**
- Stable vision-language alignment
- ITC accuracy: ~40-50%
- ITM accuracy: ~60-70%
- Loss: ~6.0-7.0

### **After Phase 2 (Main)**
- Adapted vision features
- ITC accuracy: ~60-70%
- ITM accuracy: ~75-85%
- Loss: ~4.0-5.0
- Memory utilization: ~70-80%

---

## Troubleshooting

### **LoRA not training?**
Check if Phase 2 started:
```bash
grep "LoRA adapters UNFROZEN" logs/stage1/*.log
```

### **Loss not decreasing?**
- Check temperature (should start at 0.5)
- Verify LoRA adapters are trainable (after epoch 5)
- Monitor gradient norms

### **OOM errors?**
- Reduce batch_size (128 → 64)
- Increase grad_accum_steps (2 → 4)
- Use gradient checkpointing

---

## Comparison Table

| Approach | Vision Encoder | Trainable Params | Memory | Training Speed | Inference |
|----------|----------------|------------------|--------|----------------|-----------|
| **Frozen DINOv2** | DINOv2 (frozen) | ~30M | Low | Fast | Real-time ✓ |
| **LoRA (this)** | DINOv2 + LoRA | ~32M | Low | Fast | Real-time ✓ |
| **Full Fine-tune** | DINOv2 (trainable) | ~330M | High | Slow | Real-time ✓ |
| **Pre-extracted** | Offline .npy | ~30M | Low | Fastest | Pre-computed ✗ |

**Winner: LoRA** 🏆 (Best trade-off!)

---

## Next Steps

After Stage 1 training completes:
1. ✅ Vision-language alignment learned
2. ✅ Episodic memory trained
3. ✅ Model ready for Stage 2 (reasoning module)
4. → Add Chain-of-Thought reasoning (Stage 2)
5. → Evaluation on BabyLM benchmarks

---

## References

- **DINOv2**: [https://arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)
- **LoRA**: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- **FIBER**: [https://arxiv.org/abs/2206.07643](https://arxiv.org/abs/2206.07643)
- **ALBEF**: [https://arxiv.org/abs/2107.07651](https://arxiv.org/abs/2107.07651)
- **Larimar**: [https://arxiv.org/abs/2304.13343](https://arxiv.org/abs/2304.13343)
- **BitMar**: BabyLM 2024 submission

---

## Contact

For questions or issues with the LoRA implementation, please open an issue on GitHub.

**Happy Training!** 🚀
