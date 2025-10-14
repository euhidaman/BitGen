# BitGen Stage 1: Vision-Language Pre-training

## Model Description

**BitGen** is a tiny, efficient vision-language model designed for edge devices and resource-constrained environments. This is the **Stage 1 checkpoint** focusing on vision-language pre-training using the COCO dataset.

### Architecture

BitGen combines three powerful components:

1. **BitMar Encoder-Decoder**: 1.58-bit quantized transformer (BitNet b1.58) for extreme efficiency
2. **FIBER Cross-Modal Fusion**: Queue-based contrastive learning for vision-language alignment
3. **Larimar GPM**: Generative Parametric Memory for episodic memory and reasoning

### Model Size (Tiny Configuration)

- **Embedding Dimension**: 128
- **Encoder Layers**: 4
- **Decoder Layers**: 2
- **Attention Heads**: 4
- **FFN Dimension**: 256
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)
- **Memory Slots**: 32
- **Max Sequence Length**: 256
- **Total Parameters**: ~5-10M (tiny enough for edge devices!)

### Training Data

- **Dataset**: MS-COCO Captions (validated subset)
- **Image-Caption Pairs**: ~118k training samples
- **Tokenizer**: GPT-2 BPE tokenizer

### Training Objectives

1. **Image-Text Contrastive (ITC) Loss** [Weight: 1.0 - PRIMARY]
   - FIBER-style queue-based contrastive learning
   - Aligns vision and language representations
   - Hard negative mining from queue

2. **Image-Text Matching (ITM) Loss** [Weight: 0.5]
   - Binary classification with hard negatives
   - Learns fine-grained image-caption associations

3. **Text Reconstruction Loss** [Weight: 0.5 - AUXILIARY]
   - Decoder reconstructs captions from fused features
   - Maintains language understanding
   - Label smoothing (0.1) to prevent mode collapse

4. **Memory KL Divergence** [Weight: 0.1]
   - Larimar GPM episodic memory regularization
   - Bayesian inference over memory parameters

### Key Features

✅ **Tiny Model**: Suitable for edge devices (Raspberry Pi, mobile phones)  
✅ **1.58-bit Quantization**: Extreme efficiency via BitNet b1.58  
✅ **Vision-Language Alignment**: FIBER-style contrastive learning  
✅ **Episodic Memory**: Larimar GPM for memory-augmented reasoning  
✅ **Hard Negative Mining**: ITM loss for robust alignment  
✅ **DINOv2 Vision Encoder**: State-of-the-art vision features (trainable)  

## Usage

### Loading the Model

```python
from src.train_stage1_vision_language import BitGenVisionLanguageModel, Stage1Config
import torch

# Load configuration
config = Stage1Config()

# Initialize model
model = BitGenVisionLanguageModel(config)

# Load checkpoint
checkpoint = torch.load("checkpoint-1000.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inference Example

```python
from transformers import GPT2Tokenizer
from PIL import Image
import torchvision.transforms as transforms

# Setup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load image and caption
image = Image.open("path/to/image.jpg").convert('RGB')
caption = "A cat sitting on a couch"

# Prepare inputs
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
tokens = tokenizer(caption, return_tensors='pt', padding=True, truncation=True, max_length=256)
input_ids = tokens['input_ids']  # [1, seq_len]

# Forward pass
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        return_contrastive_features=True
    )
    
    # Get vision-language embeddings
    embeddings = outputs['embeddings']  # [1, seq_len, 128]
    
    # Get contrastive features for similarity
    text_features = outputs['contrastive_features']['text_features']  # [1, 128]
    image_features = outputs['contrastive_features']['image_features']  # [1, 128]
    
    # Compute similarity
    similarity = (text_features @ image_features.T).item()
    print(f"Image-Text Similarity: {similarity:.4f}")
```

### Computing Image-Text Similarity

```python
def compute_image_text_similarity(model, image, caption, tokenizer):
    """Compute similarity between image and text caption"""
    # Prepare inputs
    image_tensor = transform(image).unsqueeze(0)
    tokens = tokenizer(caption, return_tensors='pt', padding=True, truncation=True, max_length=256)
    
    # Get features
    with torch.no_grad():
        outputs = model(
            input_ids=tokens['input_ids'],
            images=image_tensor,
            return_contrastive_features=True
        )
        
        text_feat = outputs['contrastive_features']['text_features']
        image_feat = outputs['contrastive_features']['image_features']
        
        # Normalize and compute cosine similarity
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
        image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
        similarity = (text_feat @ image_feat.T).item()
    
    return similarity
```

## Training Details

### Hyperparameters

- **Batch Size**: 128 (effective: 256 with gradient accumulation)
- **Learning Rate**: 2e-4 (linear warmup for 1000 steps, then cosine decay)
- **Optimizer**: AdamW (weight_decay=0.02)
- **Gradient Accumulation**: 2 steps
- **Max Gradient Norm**: 1.0
- **Mixed Precision**: AMP (torch.cuda.amp)
- **Temperature**: 0.07 (contrastive learning)
- **Queue Size**: 4096 (hard negatives)

### Training Schedule

- **Warmup Steps**: 1000 (linear warmup)
- **Scheduler**: Cosine decay with min LR = 0.1 × initial LR
- **Early Stopping**: Patience = 5 epochs
- **Validation Split**: 5% of dataset

### Metrics to Monitor

**Vision-Language Alignment:**
- `acc_t2i`: Text-to-Image retrieval accuracy (should increase)
- `acc_i2t`: Image-to-Text retrieval accuracy (should increase)
- `contrastive_loss`: ITC loss (should decrease steadily)
- `itm_loss`: ITM loss (should decrease)
- `itm_acc`: ITM accuracy (should increase to ~0.8-0.9)

**Language Understanding:**
- `text_loss`: Text reconstruction loss
- `perplexity`: Language model perplexity
- `token_accuracy`: Token prediction accuracy

**Memory:**
- `memory_kl_loss`: Episodic memory regularization
- `memory_utilization`: Percentage of memory slots used
- `memory_quality`: Average quality of stored memories

## Evaluation Results

### Zero-Shot Image-Text Retrieval (COCO Validation)

| Metric | Score |
|--------|-------|
| Text→Image Recall@1 | TBD |
| Image→Text Recall@1 | TBD |
| Text→Image Recall@5 | TBD |
| Image→Text Recall@5 | TBD |

*Evaluation in progress. Results will be updated after training completes.*

## Limitations and Biases

### Limitations

1. **Tiny Model**: Designed for efficiency, not SOTA performance
2. **English Only**: Trained on English captions (GPT-2 tokenizer)
3. **Stage 1 Only**: This is pre-training; reasoning module comes in Stage 2
4. **Limited Context**: Max sequence length of 256 tokens
5. **COCO-Centric**: Training data from MS-COCO (natural images, everyday objects)

### Biases

- **Dataset Bias**: Inherits biases from MS-COCO (Western-centric, object-focused)
- **Vision Bias**: DINOv2 trained on general vision data (may not generalize to specialized domains)
- **Language Bias**: GPT-2 tokenizer and text decoder have known biases

## Intended Use

### Primary Use Cases

✅ **Edge Deployment**: Raspberry Pi, mobile devices, IoT  
✅ **Vision-Language Tasks**: Image captioning, VQA, image-text retrieval  
✅ **Educational**: Learning about vision-language models and efficient AI  
✅ **Research**: Baseline for tiny vision-language models  

### Out-of-Scope Use Cases

❌ **Safety-Critical Applications**: Not tested for high-stakes decisions  
❌ **Multilingual Tasks**: English only  
❌ **Complex Reasoning**: Stage 1 is pre-training only (reasoning in Stage 2)  
❌ **Production Systems**: This is a research prototype  

## Citation

If you use BitGen in your research, please cite:

```bibtex
@software{bitgen2025,
  title={BitGen: Tiny Vision-Language Model for Edge Devices},
  author={BitGen Team},
  year={2025},
  url={https://huggingface.co/babylm-ntust/BitGen-PreReasoning}
}
```

### Related Work

**BitNet**: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)  
**FIBER**: [Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone](https://arxiv.org/abs/2206.07643)  
**Larimar**: [Larimar: Large Language Models with Episodic Memory Control](https://arxiv.org/abs/2403.11901)  
**BitMar**: BitMar 100M Token Vision-Language Model  

## Model Card Authors

BitGen Team, 2025

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/euhidaman/BitGen).

---

**License**: MIT  
**Model Version**: Stage 1 (Vision-Language Pre-training)  
**Last Updated**: 2025-01-XX
