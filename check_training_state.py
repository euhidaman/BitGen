"""
Quick diagnostic to check why loss isn't decreasing
"""

import torch
import sys

print("="*80)
print("🔍 TRAINING STATE DIAGNOSTIC")
print("="*80)

# Check 1: Are there any training logs we can analyze?
print("\n📊 CHECKING LOSS VALUES FROM OUTPUT:")
print("-" * 80)
loss_values = [4.5651, 4.5640, 4.5652, 4.5657, 4.5643, 4.5649, 4.5639, 4.5647]
print(f"Loss values (steps 180-240): {loss_values}")
print(f"Mean: {sum(loss_values)/len(loss_values):.6f}")
print(f"Std dev: {torch.tensor(loss_values).std().item():.6f}")
print(f"Range: {max(loss_values) - min(loss_values):.6f}")
print("\n❌ PROBLEM: Loss variance is tiny (0.0018), stuck at 4.56")
print("   Expected: Should drop to ~3.5 within 100 steps, ~2.5 within 500 steps")

# Check 2: What's the robot loss doing?
print("\n🤖 CHECKING ROBOT LOSS:")
print("-" * 80)
robot_losses = [0.631346, 0.611447, 0.621439, 0.627794, 0.583589, 0.602006]
print(f"Robot loss values: {robot_losses}")
print(f"Mean: {sum(robot_losses)/len(robot_losses):.6f}")
print(f"✓ Robot loss is reasonable (0.58-0.63), but it's only 10% of batches")

# Check 3: Calculate expected global_step
print("\n🔢 CHECKING GLOBAL STEP COUNTER:")
print("-" * 80)
displayed_step = 240
grad_accum = 2
actual_optimizer_steps = displayed_step // grad_accum  # 120 optimizer steps
print(f"Displayed step: {displayed_step}")
print(f"Gradient accumulation: {grad_accum}")
print(f"Actual optimizer steps: {actual_optimizer_steps}")
print(f"\n⚠️  BUGS DETECTED:")
print(f"   1. global_step incremented 2x in train_bitgen.py (lines 1196, 1207)")
print(f"   2. global_step incremented 1x in adaptive_loss.py (line 345)")
print(f"   3. Total: 3x increment per optimizer step!")
print(f"   4. Loss calculation thinks step = {actual_optimizer_steps * 3} = {actual_optimizer_steps * 3}")
print(f"      (Should be {actual_optimizer_steps})")

# Check 4: What stage should we be in?
print("\n🎯 CHECKING TRAINING STAGE:")
print("-" * 80)
wrong_step = actual_optimizer_steps * 3
correct_step = actual_optimizer_steps
print(f"Loss calculation uses step: {wrong_step}")
print(f"Correct step should be: {correct_step}")
print(f"Stage thresholds: 2000 (Stage 1→2), 5000 (Stage 2→3)")
print(f"Current stage (wrong): Stage {1 if wrong_step < 2000 else (2 if wrong_step < 5000 else 3)}")
print(f"Current stage (correct): Stage {1 if correct_step < 2000 else (2 if correct_step < 5000 else 3)}")
print(f"✓ Both show Stage 1, so stage is correct for now")

# Check 5: What about learning rate?
print("\n📈 CHECKING LEARNING RATE:")
print("-" * 80)
print(f"Current LR from logs: 0.001000")
print(f"Expected LR at step 120: ~0.00099 (cosine decay just starting)")
print(f"✓ LR is correct!")

# Check 6: Main hypothesis
print("\n💡 MAIN HYPOTHESIS:")
print("="*80)
print("The contrastive loss of 4.56 is TOO HIGH for CLIP-style learning.")
print("This suggests one of:")
print("  1. ❌ Features are random (model not learning)")
print("  2. ❌ Gradients not flowing to text/vision encoders")
print("  3. ❌ Temperature too high (making similarities too weak)")
print("  4. ❌ Features not normalized properly")
print("  5. ❌ Batch size too small (CLIP uses 32K!)")
print("\nRECOMMENDED ACTIONS:")
print("  1. Add gradient flow check (print grad_norm for each module)")
print("  2. Lower temperature from 0.07 to 0.01")
print("  3. Print feature norms to check if they're being learned")
print("  4. Check if text/vision encoders are frozen")
print("="*80)
