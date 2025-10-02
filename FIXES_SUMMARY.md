# BitGen Bug Fixes - Summary

## Issue Fixed
**Error:** `BitGenModel.forward() got an unexpected keyword argument 'attention_mask'`

## Root Cause
The training script (`train_bitgen.py`) was passing an `attention_mask` parameter to the model's forward method, but the `BitGenModel.forward()` method didn't accept this parameter.

## Changes Made

### 1. **src/bitgen_model.py** - Added attention_mask support

#### Change 1: Updated forward method signature (Line 628)
```python
# Before:
def forward(self, input_ids, images=None, return_robot_selection=False, attention_cache=None, return_analysis_data=False):

# After:
def forward(self, input_ids, images=None, attention_mask=None, return_robot_selection=False, attention_cache=None, return_analysis_data=False):
```

#### Change 2: Updated _forward_attention_with_weights method (Line 759)
```python
# Before:
def _forward_attention_with_weights(self, attention_layer, x, cache):

# After:
def _forward_attention_with_weights(self, attention_layer, x, cache, attention_mask=None):
```

#### Change 3: Added attention mask application logic (Lines 795-809)
```python
# Apply attention mask if provided
if attention_mask is not None:
    # Expand attention_mask to match scores dimensions [batch, heads, seq_len, key_len]
    if attention_mask.dim() == 2:
        # [batch, seq_len] -> [batch, 1, 1, seq_len]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    elif attention_mask.dim() == 3:
        # [batch, 1, seq_len] -> [batch, 1, 1, seq_len]
        attention_mask = attention_mask.unsqueeze(1)
    
    # Convert attention_mask (0s and 1s) to additive mask (0s and -inf)
    attention_mask = (1.0 - attention_mask) * -10000.0
    scores = scores + attention_mask
```

#### Change 4: Updated forward method to pass attention_mask (Line 690)
```python
# Before:
layer_output, cache, attention_weights = self._forward_attention_with_weights(
    attention_layer, x, layer_cache
)

# After:
layer_output, cache, attention_weights = self._forward_attention_with_weights(
    attention_layer, x, layer_cache, attention_mask
)
```

### 2. **src/train_bitgen.py** - Fixed PyTorch import compatibility

#### Change: Added fallback import for older PyTorch versions (Lines 10-13)
```python
# Before:
from torch.amp import GradScaler, autocast

# After:
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
```

## How the Fix Works

1. **attention_mask parameter**: Now optional in the model's forward method, maintaining backward compatibility
2. **Mask application**: The attention mask is properly broadcast to match the attention scores dimensions and converted to an additive mask (-inf for masked positions)
3. **PyTorch compatibility**: The code now works with both older and newer versions of PyTorch

## Testing Recommendations

1. Run training with the fixed code:
   ```bash
   python bitgen_cli.py train --coco_data data/coco --model_size tiny --num_epochs 10
   ```

2. Verify that:
   - Training starts without the `attention_mask` error
   - Attention masking works correctly for padded sequences
   - Model produces valid outputs

## Additional Notes

- The attention_mask parameter is optional, so existing code that doesn't pass it will continue to work
- The mask implementation follows standard transformer attention masking practices
- All changes are backward compatible with existing inference code
