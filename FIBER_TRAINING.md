# BitGen Stage 1: FIBER-Style Vision-Language Pre-training

## Overview

BitGen Stage 1 implements **FIBER-inspired two-phase pre-training** while preserving core innovations:
- ✅ **Larimar GPM episodic memory** (trained throughout)
- ✅ **BitNet-style quantization** (DINOv2 + text encoders/decoders)
- ✅ **FIBER cross-modal fusion** (queue-based contrastive learning)

## Two-Phase Training Approach

### Phase 1: Coarse-Grained Pre-training
**Goal**: Learn image-level vision-language alignment

**Datasets**:
- COCO Captions (~120k images, ~600k captions)
- SBU Captions (~1M image-text pairs)
- Visual Genome (~108k images, ~5M region descriptions)
- Conceptual Captions 3M (~3M web image-text pairs)
- Flickr30k (~31k images for evaluation)

**Tasks**:
- **ITC (Image-Text Contrastive)**: Primary learning signal, queue-based contrastive loss
- **ITM (Image-Text Matching)**: Hard negative mining, binary classification
- **Text Reconstruction**: Auxiliary loss for language understanding
- **Memory KL**: Episodic memory regularization (Larimar GPM)

**Loss Weights**:
```python
contrastive_weight: 1.0  # PRIMARY signal
itm_weight: 0.5         # Hard negatives
text_loss_weight: 0.5   # AUXILIARY
memory_kl_weight: 0.1   # Regularization
```

**Training**:
- Epochs: 25 (configurable via `coarse_epochs`)
- Batch size: 128 (effective: 256 with grad_accum=2)
- Learning rate: 2e-4 (BitMar-style)
- Warmup: 1000 steps → cosine decay
- Early stopping: patience=5 epochs

### Phase 2: Fine-Grained Pre-training
**Goal**: Learn region-level spatial understanding

**Datasets**:
- RefCOCO (~50k referring expressions)
- RefCOCO+ (~50k referring expressions, no location words)
- RefCOCOg (~85k referring expressions)
- Visual Genome Regions (~108k images, ~3.8M objects with boxes)
- MixedGrounding (MDETR curated, ~1M annotations)

**Tasks**:
- **Phrase Grounding**: Match text phrases to image regions (bounding boxes)
- **Spatial Reasoning**: Understand position, size, relationships
- **Region-level Contrastive**: Similar to ITC but at region level
- **Episodic Memory**: Continue training Larimar GPM with spatial info

**Loss Weights**:
```python
grounding_weight: 0.5   # Phrase grounding
contrastive_weight: 1.0 # Region-level ITC
memory_kl_weight: 0.1   # Regularization
```

**Training**:
- Epochs: 25 (configurable via `fine_epochs`)
- Batch size: 128
- Learning rate: Inherited from Phase 1 (warm restart)
- Loads Phase 1 checkpoint as initialization

## Architecture

```
Input: Image + Text (+ optional bounding boxes)
  ↓
DINOv2 Vision Encoder (BitNet quantized)
  ↓
Text Encoder (GPT-2 tokenizer + BitNet quantized)
  ↓
FIBER Cross-Modal Fusion
  ├── Queue-based contrastive (4096 size)
  ├── Temperature clamping (0.001-1.0)
  └── log_softmax loss (ALBEF formula)
  ↓
Larimar GPM Episodic Memory
  ├── Memory size: 32 (tiny model)
  ├── Direct writing: True
  └── KL divergence regularization
  ↓
Transformer Encoder (4 layers, 4 heads, 128 dim)
  ↓
Text Decoder (2 layers, BitNet quantized)
  ↓
Outputs: Embeddings + Contrastive Features + Memory Info
```

## Dataset Download

### Automatic Download (Recommended)
```bash
# Download all datasets
python download_fiber_datasets.py --all --data_root ./data

# Download coarse-grained only
python download_fiber_datasets.py --coarse --data_root ./data

# Download fine-grained only
python download_fiber_datasets.py --fine --data_root ./data

# Download specific datasets
python download_fiber_datasets.py --datasets coco sbu vg --data_root ./data
```

### Manual Download (if needed)
Some datasets require manual download:

**SBU Captions**:
1. Download from: http://www.cs.virginia.edu/~vicente/sbucaptions/
2. Place files in: `data/sbu/`
3. Run: `cd data/sbu && python download_sbu_images.py`

**Conceptual Captions 3M**:
1. Download from: https://ai.google.com/research/ConceptualCaptions/download
2. Place TSV in: `data/conceptual_captions/`
3. Run: `cd data/conceptual_captions && python download_cc3m_images.py`

**Flickr30k**:
1. Request access: http://shannon.cs.illinois.edu/DenotationGraph/
2. Extract to: `data/flickr30k/flickr30k_images/`

**RefCOCO** (uses COCO 2014 images):
1. Download COCO 2014 first (automatic)
2. MDETR annotations downloaded automatically

## Training

### Quick Start (Single COCO Dataset)
```bash
# Legacy mode: COCO only
python src/train_stage1_vision_language.py
```

### FIBER-Style Multi-Dataset Training
```bash
# Two-phase training (coarse → fine)
python src/train_stage1_vision_language.py --use_multi_datasets --enable_two_phase

# Single-phase (coarse + fine together)
python src/train_stage1_vision_language.py --use_multi_datasets
```

### Configuration

Edit `src/train_stage1_vision_language.py`:

```python
@dataclass
class Stage1Config:
    # FIBER-style two-phase training
    enable_two_phase_training: bool = True  # True = coarse → fine
    coarse_epochs: int = 25                 # Phase 1 epochs
    fine_epochs: int = 25                   # Phase 2 epochs
    grounding_weight: float = 0.5           # Phrase grounding loss weight
    
    # Multi-dataset config
    use_multi_datasets: bool = True         # Use FIBER-style datasets
    data_root: str = "data"                 # Root directory
    max_vg_samples: int = 100000            # Limit VG (huge dataset)
    
    # Model (Tiny - BitMar-sized)
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 256
    
    # Memory (Larimar GPM)
    memory_size: int = 32
    memory_dim: int = 128
    direct_writing: bool = True
    
    # Training
    batch_size: int = 128
    grad_accum_steps: int = 2               # Effective: 256
    learning_rate: float = 2e-4
    weight_decay: float = 0.02
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    use_amp: bool = True
    
    # Contrastive
    queue_size: int = 4096
    temperature: float = 0.07
```

## Dataset Structure

After download, your `data/` directory should look like:

```
data/
├── coco/
│   ├── train2017/          # Coarse-grained (captions)
│   ├── val2017/
│   ├── train2014/          # Fine-grained (RefCOCO uses these)
│   └── annotations/
├── sbu/
│   ├── images_train/
│   └── annot.json
├── visual_genome/
│   ├── VG_100K/
│   ├── VG_100K_2/
│   ├── region_descriptions.json  # Coarse-grained
│   └── objects.json              # Fine-grained (with boxes)
├── conceptual_captions/
│   ├── images_train/
│   └── train_annot.json
├── flickr30k/
│   ├── flickr30k_images/
│   └── annotations/
├── mdetr_annotations/           # RefCOCO/+/g, MixedGrounding
│   ├── final_refcoco_train.json
│   ├── final_refcoco+_train.json
│   ├── final_refcocog_train.json
│   └── final_mixed_train_no_coco.json
├── gqa/
│   └── images/                  # For MixedGrounding
└── dataset_structure.json       # Generated metadata
```

## Monitoring Training

### WandB Integration
Automatically logs to WandB:
- Loss curves (ITC, ITM, text reconstruction, grounding)
- Accuracy (t2i, i2t retrieval)
- Gradient norms (before clipping)
- Learning rate schedule
- Memory utilization (Larimar GPM)
- Similarity matrices
- Queue quality heatmaps
- UMAP embedding visualizations

Project: `bitgen-training`
Entity: `babylm-ntust`

### HuggingFace Hub
Auto-pushes checkpoints every epoch:
- Repository: `BitGen-PreReasoning`
- Format: Single `pytorch_model.bin` (overwrites each epoch)
- Metadata: `training_metadata.json`

## Key Differences from FIBER

| Aspect | FIBER | BitGen Stage 1 |
|--------|-------|----------------|
| **Model Size** | Large (ViT-B/L, BERT-base) | Tiny (128 dim, 4 layers) |
| **Vision Encoder** | ViT/Swin Transformer | DINOv2 (BitNet quantized) |
| **Text Encoder** | BERT/RoBERTa | GPT-2 tokenizer + custom (BitNet) |
| **Memory** | No episodic memory | Larimar GPM (32 slots) |
| **Quantization** | Float32 | BitNet (1.58-bit) |
| **Training Data** | ~10M images | ~4M images (can scale up) |
| **Target Device** | GPU clusters | Edge devices (Raspberry Pi, mobile) |
| **Loss Formula** | ALBEF-style | EXACT ALBEF match |
| **Queue Size** | 65k | 4096 (memory constrained) |

## Next Steps

After Stage 1 completes:

1. **Verify Alignment**: Check t2i/i2t accuracy > 0.20 (should reach 0.30-0.50 with full datasets)
2. **Load for Stage 2**: Use Phase 2 checkpoint as initialization
3. **Stage 2 Training**: Add reasoning module + full generation capabilities
4. **Edge Deployment**: Deploy to Raspberry Pi / mobile devices

## Troubleshooting

### NaN Gradients at Step 600
**Fixed**: Queue initialization now uses `torch.randn() * 0.02` to match feature scale

### t2i/i2t Stuck at 0.01-0.02
**Solutions**:
- Increase warmup steps (1000 → 2000)
- Increase max_grad_norm (1.0 → 3.0)
- Train longer (50+ epochs)
- Use more datasets (SBU, VG, CC3M)

### Out of Memory
**Solutions**:
- Reduce batch_size (128 → 64)
- Reduce max_vg_samples (100k → 50k)
- Use fewer datasets initially
- Enable gradient checkpointing

### Dataset Download Fails
**Solutions**:
- Check internet connection
- Some require manual download (SBU, CC3M, Flickr30k)
- Follow instructions in `download_fiber_datasets.py` output

## References

1. **FIBER**: "Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone"
2. **Larimar**: "Larimar: Large Language Models with Episodic Memory Control"
3. **BitNet**: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
4. **ALBEF**: "Align before Fuse: Vision and Language Representation Learning with Momentum Distillation"

## License

MIT License (same as BitGen, FIBER, Larimar, BitNet)
