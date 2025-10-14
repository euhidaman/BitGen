# BitGen Stage 1 - Remote Training Guide

## ✅ Ready for Remote Device Training

All code has been pushed to GitHub. You can now pull and run on your remote training device.

## 🚀 Quick Start on Remote Device

```bash
# 1. Clone/pull latest code
git clone https://github.com/euhidaman/BitGen.git
cd BitGen
git pull origin main

# 2. Install dependencies
pip install -r requirements.txt
pip install huggingface_hub  # For automatic Hub pushing

# 3. Login to HuggingFace Hub (one-time setup)
huggingface-cli login
# Enter your HuggingFace token when prompted

# 4. Run training
cd src
python train_stage1_vision_language.py
```

## 📊 What's Changed - Summary

### 1. ✅ Model Size Reduction (BitMar Tiny Config)
- **Embedding Dimension**: 256 → **128**
- **Encoder Layers**: 6 → **4**
- **Decoder Layers**: 4 → **2**
- **Attention Heads**: 8 → **4**
- **FFN Dimension**: 512 → **256**
- **Vocabulary**: 8192 → **50257** (GPT-2 tokenizer)
- **Memory Slots**: 1000 → **32**
- **Max Sequence Length**: 512 → **256**
- **Estimated Parameters**: ~5-10M (tiny enough for edge devices!)

### 2. ✅ Improved Vision-Language Learning

**New Loss Components:**
- **Image-Text Contrastive (ITC)**: Weight 1.0 (PRIMARY) - FIBER queue-based
- **Image-Text Matching (ITM)**: Weight 0.5 - Hard negative mining
- **Text Reconstruction**: Weight 0.5 (AUXILIARY) - Language understanding
- **Memory KL Divergence**: Weight 0.1 - Episodic memory regularization

**Why this works:**
- ITC learns vision-language alignment (image ↔ caption matching)
- ITM adds fine-grained matching with hard negatives
- Text loss maintains language understanding (auxiliary)
- Memory KL keeps episodic memory diverse and useful

### 3. ✅ FIBER Cross-Modal Fusion
- ✅ Separate transforms for fusion vs contrastive learning
- ✅ Queue-based hard negative mining (4096 queue size)
- ✅ Proper pooling for text and image features
- ✅ Temperature-scaled similarity (0.07)
- ✅ DINOv2 vision encoder (trainable)

### 4. ✅ HuggingFace Hub Integration (BitMar-style)
- ✅ Push checkpoints at **iteration intervals** (every 1000 steps)
- ✅ Checkpoint folders: `checkpoint-1000`, `checkpoint-2000`, etc.
- ✅ Auto-cleanup: Keep only last 5 checkpoints
- ✅ **Automatic model card generation** on HuggingFace Hub
- ✅ Model card includes: architecture, usage, hyperparameters, limitations

### 5. ✅ New Metrics to Monitor

**Progress Bar Shows:**
- `itc`: Image-Text Contrastive loss (should decrease steadily)
- `itm`: Image-Text Matching loss (should decrease)
- `txt`: Text reconstruction loss
- `t2i`: Text→Image retrieval accuracy (should increase to 0.3-0.5)
- `i2t`: Image→Text retrieval accuracy (should increase to 0.3-0.5)
- `lr`: Current learning rate

**Additional Logged Metrics:**
- `itm_acc`: ITM classification accuracy (should reach 0.8-0.9)
- `memory_utilization`: Percentage of memory slots used
- `memory_quality`: Average quality of stored memories

## 📈 Expected Training Behavior

### Good Signs:
✅ **ITC loss decreasing**: 4.85 → 3.5 → 2.5 → ... (vision-language alignment improving)
✅ **ITM loss decreasing**: Should drop to ~0.3-0.5 (binary classification working)
✅ **t2i/i2t accuracy increasing**: 0.01 → 0.1 → 0.3 → 0.5 (retrieval improving)
✅ **ITM accuracy increasing**: Should reach 0.8-0.9 (hard negatives being learned)
✅ **Text loss stable**: ~1.0-3.0 (language understanding maintained)
✅ **Gradient norms healthy**: 0.05-0.5 range
✅ **Memory utilization >50%**: Episodic memory being used

### Bad Signs (If These Happen):
❌ **ITC loss stuck at ~4.85**: Contrastive not learning (check DINOv2 is trainable)
❌ **t2i/i2t accuracy stays near 0**: No vision-language alignment
❌ **ITM accuracy stuck at 0.5**: Hard negatives not working (random guessing)
❌ **Text loss collapse to 0**: Mode collapse (should be prevented by label smoothing)
❌ **Gradient norms vanishing (<0.01)**: Learning stopped
❌ **Memory utilization <10%**: Memory not being used

## 🔧 Training Configuration

```python
# Model Config (Tiny for Edge Devices)
embed_dim = 128
num_layers = 4  # encoder
decoder_layers = 2
num_heads = 4
ffn_dim = 256
vocab_size = 50257  # GPT-2
memory_size = 32
max_seq_len = 256

# Training Config
batch_size = 128
grad_accum_steps = 2  # Effective batch: 256
learning_rate = 2e-4
warmup_steps = 1000
num_epochs = 50
early_stopping_patience = 5

# Loss Weights (CRITICAL!)
contrastive_weight = 1.0  # PRIMARY - vision-language alignment
itm_weight = 0.5  # Hard negative mining
text_loss_weight = 0.5  # AUXILIARY - language understanding
memory_kl_weight = 0.1  # Memory regularization
```

## 📂 HuggingFace Hub Structure

After training starts, your Hub repo will have:

```
babylm-ntust/BitGen-PreReasoning-stage1/
├── README.md                          # ✅ Auto-generated model card
├── config.json                        # Model configuration
├── checkpoint-1000/
│   ├── checkpoint-1000.pt            # Model weights at step 1000
│   └── metrics.json                  # Training metrics
├── checkpoint-2000/
│   ├── checkpoint-2000.pt
│   └── metrics.json
└── ...
```

## 🧪 Testing Checklist

After training starts on remote device:

- [ ] **Step 1**: Verify model parameters are ~5-10M (check first log line)
- [ ] **Step 100**: Check gradient norm is 0.05-0.5 (not 0.0!)
- [ ] **Step 500**: Verify ITC loss is decreasing (not stuck at 4.85)
- [ ] **Step 1000**: Check t2i/i2t accuracy is >0.01 (not zero!)
- [ ] **Step 1000**: Verify HuggingFace Hub has `checkpoint-1000/` folder
- [ ] **Step 1000**: Check model card (README.md) exists on Hub
- [ ] **Epoch 1**: Verify early stopping is tracking val_loss
- [ ] **Epoch 2**: Check memory utilization is >30%

## 🐛 Troubleshooting

### Issue: "HuggingFace Hub not available"
**Solution**: `pip install huggingface_hub` and run `huggingface-cli login`

### Issue: "CUDA out of memory"
**Solution**: Reduce batch_size from 128 to 64 in `Stage1Config`

### Issue: ITC loss stuck at 4.85
**Solution**: Check DINOv2 is trainable (should see "Trainable" in logs)

### Issue: t2i/i2t accuracy stays at 0
**Solution**: Check that images and captions are aligned (not shuffled)

### Issue: Text loss collapse to 0
**Solution**: Label smoothing should prevent this (check it's set to 0.1)

## 📝 Important Files

All changes pushed to `main` branch:

1. **`src/train_stage1_vision_language.py`**:
   - Reduced model size (BitMar tiny config)
   - New loss computation with ITC + ITM
   - HuggingFace Hub pushing every 1000 steps
   - Model card auto-generation

2. **`src/huggingface_integration.py`**:
   - `push_checkpoint()` method for iterative pushing
   - `create_model_card()` method for auto-generating README.md
   - Checkpoint cleanup (keep last 5)

3. **`src/fiber_fusion.py`**:
   - Already has proper FIBER implementation
   - Separate transforms for fusion vs contrastive
   - Queue-based hard negatives

4. **`src/larima_memory.py`**:
   - Episodic memory with quality tracking
   - Memory utilization logging

## 🎯 Success Criteria

**After 5-10 epochs, you should see:**

1. **Vision-Language Alignment Working**:
   - ITC loss: 4.85 → 2.5-3.0
   - t2i accuracy: 0.01 → 0.3-0.5
   - i2t accuracy: 0.01 → 0.3-0.5

2. **Hard Negative Learning**:
   - ITM loss: Starting value → 0.3-0.5
   - ITM accuracy: 0.5 → 0.8-0.9

3. **Language Understanding Maintained**:
   - Text loss: Stable at 1.0-3.0
   - Perplexity: 3-7 range

4. **Memory Functioning**:
   - Memory utilization: >50%
   - Memory quality: >0.7

## 🚀 Next Steps After Training

1. **Evaluate on COCO validation set** (image-text retrieval)
2. **Test inference** with new image-caption pairs
3. **Deploy to edge device** (Raspberry Pi, mobile)
4. **Start Stage 2** (add reasoning module)

## 📞 Support

If you encounter issues during training:
1. Check gradient norms (should be 0.05-0.5)
2. Monitor ITC loss (should decrease)
3. Verify HuggingFace Hub is receiving checkpoints
4. Check model card exists on Hub

All code is ready for training. Good luck! 🎉

---

**Last Updated**: 2025-01-14  
**Git Commit**: a1ac1c5  
**Status**: ✅ Ready for remote training
