"""
Adaptive Loss System for BitGen
Dynamically adjusts training objectives based on performance across modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from collections import defaultdict, deque

class AdaptiveLossManager:
    """Manages adaptive loss weighting for multi-modal training"""

    def __init__(self,
                 initial_weights: Dict[str, float] = None,
                 adaptation_rate: float = 0.01,
                 window_size: int = 100,
                 min_weight: float = 0.1,
                 max_weight: float = 2.0):

        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Default loss weights
        if initial_weights is None:
            initial_weights = {
                'language_modeling': 1.0,
                'vision_text_alignment': 1.0,
                'reconstruction': 0.5,
                'reasoning_consistency': 1.0,
                'episodic_memory': 0.8,
                'robot_selection': 0.6
            }

        self.current_weights = initial_weights.copy()

        # Performance tracking
        self.loss_history = defaultdict(lambda: deque(maxlen=window_size))
        self.performance_history = defaultdict(lambda: deque(maxlen=window_size))

        # Gradient tracking for stability
        self.gradient_history = defaultdict(lambda: deque(maxlen=window_size))

    def update_loss_history(self, losses: Dict[str, float]):
        """Update loss history for each component"""
        for loss_name, loss_value in losses.items():
            if not torch.isnan(torch.tensor(loss_value)) and not torch.isinf(torch.tensor(loss_value)):
                self.loss_history[loss_name].append(loss_value)

    def update_performance_history(self, metrics: Dict[str, float]):
        """Update performance metrics history"""
        for metric_name, metric_value in metrics.items():
            if not torch.isnan(torch.tensor(metric_value)) and not torch.isinf(torch.tensor(metric_value)):
                self.performance_history[metric_name].append(metric_value)

    def compute_loss_trends(self) -> Dict[str, float]:
        """Compute loss trends to determine adaptation direction"""
        trends = {}

        for loss_name, history in self.loss_history.items():
            if len(history) < 10:  # Need enough data points
                trends[loss_name] = 0.0
                continue

            # Simple linear trend
            x = np.arange(len(history))
            y = np.array(list(history))

            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                trends[loss_name] = slope
            else:
                trends[loss_name] = 0.0

        return trends

    def compute_relative_difficulty(self) -> Dict[str, float]:
        """Compute relative difficulty of each loss component"""
        difficulties = {}

        for loss_name, history in self.loss_history.items():
            if len(history) < 5:
                difficulties[loss_name] = 1.0
                continue

            recent_losses = list(history)[-10:]  # Last 10 values

            # Compute coefficient of variation (std/mean) as difficulty measure
            mean_loss = np.mean(recent_losses)
            std_loss = np.std(recent_losses)

            if mean_loss > 0:
                cv = std_loss / mean_loss
                difficulties[loss_name] = cv
            else:
                difficulties[loss_name] = 1.0

        return difficulties

    def adapt_weights(self, current_losses: Dict[str, float]) -> Dict[str, float]:
        """Adapt loss weights based on performance and trends"""

        # Update history
        self.update_loss_history(current_losses)

        # Compute adaptation signals
        trends = self.compute_loss_trends()
        difficulties = self.compute_relative_difficulty()

        # Adapt weights
        new_weights = self.current_weights.copy()

        for loss_name in self.current_weights.keys():
            if loss_name not in current_losses:
                continue

            current_weight = self.current_weights[loss_name]
            trend = trends.get(loss_name, 0.0)
            difficulty = difficulties.get(loss_name, 1.0)

            # Adaptation logic:
            # - If loss is increasing (trend > 0), increase weight
            # - If loss is decreasing but difficulty is high, maintain weight
            # - If loss is stable and low difficulty, decrease weight slightly

            weight_adjustment = 0.0

            if trend > 0:  # Loss increasing - need more attention
                weight_adjustment = self.adaptation_rate * (1 + difficulty)
            elif trend < 0 and difficulty < 0.5:  # Loss decreasing and stable
                weight_adjustment = -self.adaptation_rate * 0.5
            elif difficulty > 1.5:  # High difficulty - increase weight
                weight_adjustment = self.adaptation_rate * difficulty

            # Apply adjustment
            new_weight = current_weight + weight_adjustment

            # Clamp to bounds
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            new_weights[loss_name] = new_weight

        # Normalize weights to prevent total loss explosion
        total_weight = sum(new_weights.values())
        if total_weight > len(new_weights) * 1.5:  # If total weight too high
            scale_factor = (len(new_weights) * 1.2) / total_weight
            for key in new_weights:
                new_weights[key] *= scale_factor

        self.current_weights = new_weights
        return new_weights

    def get_weighted_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted total loss"""
        total_loss = 0.0
        loss_dict = {}

        # Convert tensor losses to float for adaptation
        float_losses = {k: v.item() if isinstance(v, torch.Tensor) else v
                       for k, v in losses.items()}

        # Adapt weights based on current losses
        current_weights = self.adapt_weights(float_losses)

        # Compute weighted loss
        for loss_name, loss_tensor in losses.items():
            if loss_name in current_weights:
                weight = current_weights[loss_name]
                weighted_loss = weight * loss_tensor
                total_loss = total_loss + weighted_loss
                loss_dict[f"weighted_{loss_name}"] = weighted_loss

        return total_loss, loss_dict, current_weights

class BitGenLoss(nn.Module):
    """Complete loss function for BitGen training"""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Initialize adaptive loss manager
        self.adaptive_loss = AdaptiveLossManager()

        # Loss functions with label smoothing for stability
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

        # Temperature parameters with safer initialization
        self.vision_text_temperature = nn.Parameter(torch.tensor(0.2))  # Increased from 0.07
        self.reasoning_temperature = nn.Parameter(torch.tensor(1.0))

        # Debug step counter for reduced logging
        self.debug_step_counter = 0
        
        # Global step for multi-stage training
        self.global_step = 0

    def contrastive_loss(self, text_features: torch.Tensor, vision_features: torch.Tensor, temperature: float = 0.01) -> torch.Tensor:
        """FIBER/CLIP-style image-text contrastive loss with numerical stability"""
        if text_features is None or vision_features is None:
            return torch.tensor(0.0, device=text_features.device if text_features is not None else vision_features.device, requires_grad=True)
        
        # STABILITY: Check for NaN/Inf in input features
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            return torch.tensor(0.0, device=text_features.device, requires_grad=True)
        if torch.isnan(vision_features).any() or torch.isinf(vision_features).any():
            return torch.tensor(0.0, device=vision_features.device, requires_grad=True)
        
        # Normalize features with epsilon for stability
        text_features = F.normalize(text_features, p=2, dim=-1, eps=1e-8)
        vision_features = F.normalize(vision_features, p=2, dim=-1, eps=1e-8)
        
        # STABILITY: Clamp temperature to prevent division by very small numbers
        temperature = max(temperature, 0.01)  # Minimum temp = 0.01
        
        # Compute similarity matrices
        logits_per_text = torch.matmul(text_features, vision_features.t()) / temperature
        logits_per_vision = logits_per_text.t()
        
        # STABILITY: Clamp logits to prevent exp overflow in cross_entropy
        logits_per_text = torch.clamp(logits_per_text, min=-100.0, max=100.0)
        logits_per_vision = torch.clamp(logits_per_vision, min=-100.0, max=100.0)
        
        # Create labels (diagonal is positive pairs)
        batch_size = text_features.shape[0]
        labels = torch.arange(batch_size, device=text_features.device)
        
        # Symmetric cross-entropy loss
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss_vision = F.cross_entropy(logits_per_vision, labels)
        
        # STABILITY: Check for NaN in losses
        if torch.isnan(loss_text) or torch.isinf(loss_text):
            loss_text = torch.tensor(0.0, device=text_features.device, requires_grad=True)
        if torch.isnan(loss_vision) or torch.isinf(loss_vision):
            loss_vision = torch.tensor(0.0, device=vision_features.device, requires_grad=True)
        
        loss = (loss_text + loss_vision) / 2.0
        
        return loss

    def vision_text_alignment_loss(self, text_features: torch.Tensor,
                                 vision_features: torch.Tensor) -> torch.Tensor:
        """CLIP-style vision-text alignment loss with stability checks"""
        if vision_features is None:
            return torch.tensor(0.0, device=text_features.device, requires_grad=True)

        # Check for invalid features
        if torch.isnan(text_features).any() or torch.isnan(vision_features).any():
            return torch.tensor(0.0, device=text_features.device, requires_grad=True)

        # Normalize features with epsilon to prevent division by zero
        text_features = F.normalize(text_features.mean(dim=1), dim=-1, eps=1e-8)
        vision_features = F.normalize(vision_features.squeeze(1), dim=-1, eps=1e-8)

        # Clamp temperature to prevent division by very small numbers
        temperature = torch.clamp(self.vision_text_temperature, min=0.1, max=5.0)

        # Compute similarity matrix
        logits_per_text = torch.matmul(text_features, vision_features.T) / temperature
        logits_per_vision = logits_per_text.T

        # Clamp logits for stability
        logits_per_text = torch.clamp(logits_per_text, min=-50, max=50)
        logits_per_vision = torch.clamp(logits_per_vision, min=-50, max=50)

        # Symmetric loss
        batch_size = text_features.size(0)
        labels = torch.arange(batch_size, device=text_features.device)

        text_loss = F.cross_entropy(logits_per_text, labels)
        vision_loss = F.cross_entropy(logits_per_vision, labels)

        return (text_loss + vision_loss) / 2

    def reconstruction_loss(self, predicted_embeddings: torch.Tensor,
                           target_embeddings: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss for episodic memory - allow gradients to flow"""
        return self.mse_loss(predicted_embeddings, target_embeddings)

    def reasoning_consistency_loss(self, reasoning_logits: torch.Tensor,
                                 base_logits: torch.Tensor) -> torch.Tensor:
        """Ensure reasoning steps are consistent with final output - numerically stable"""
        # Check for invalid logits
        if torch.isnan(reasoning_logits).any() or torch.isnan(base_logits).any():
            return torch.tensor(0.0, device=reasoning_logits.device, requires_grad=True)

        # Clamp logits to prevent extreme values
        reasoning_logits = torch.clamp(reasoning_logits, min=-50, max=50)
        base_logits = torch.clamp(base_logits, min=-50, max=50)

        # Clamp temperature
        temperature = torch.clamp(self.reasoning_temperature, min=0.5, max=2.0)

        # Use MSE loss instead of KL divergence for better stability
        reasoning_probs = F.softmax(reasoning_logits / temperature, dim=-1)
        base_probs = F.softmax(base_logits / temperature, dim=-1)

        # MSE loss is more stable than KL divergence
        mse_loss = F.mse_loss(reasoning_probs, base_probs)

        return mse_loss

    def episodic_memory_consistency_loss(self, memory_retrieved: torch.Tensor,
                                       current_context: torch.Tensor) -> torch.Tensor:
        """Ensure retrieved memories are relevant to current context - numerically stable"""
        # Check for invalid tensors
        if torch.isnan(memory_retrieved).any() or torch.isnan(current_context).any():
            return torch.tensor(0.0, device=memory_retrieved.device, requires_grad=True)

        # Use MSE loss instead of cosine similarity for better stability
        retrieved_mean = memory_retrieved.mean(dim=1)
        context_mean = current_context.mean(dim=1)

        # Normalize for stability
        retrieved_norm = F.normalize(retrieved_mean, dim=-1, eps=1e-8)
        context_norm = F.normalize(context_mean, dim=-1, eps=1e-8)

        # MSE loss on normalized features
        return F.mse_loss(retrieved_norm, context_norm)

    def robot_selection_accuracy_loss(self, robot_probs: torch.Tensor,
                                    target_robot: torch.Tensor) -> torch.Tensor:
        """Robot selection accuracy loss"""
        if target_robot is None:
            return torch.tensor(0.0, device=robot_probs.device)

        return F.cross_entropy(robot_probs, target_robot)

    def forward(self,
                model_outputs: Dict,
                labels: torch.Tensor,
                images: Optional[torch.Tensor] = None,
                target_robot: Optional[torch.Tensor] = None,
                global_step: int = 0) -> Tuple[torch.Tensor, Dict]:
        """Complete forward pass with adaptive loss - FIXED TO ACTUALLY LEARN
        
        Args:
            global_step: Current training step (managed by training loop)
        """

        # Increment counters
        self.debug_step_counter += 1
        # Use externally managed global_step (from training loop)
        self.global_step = global_step
        should_debug = (self.debug_step_counter % 5000 == 0)

        # CRITICAL DEBUG: Check labels
        if self.debug_step_counter % 100 == 0:
            valid_labels = (labels != -100).sum().item()
            print(f"[LOSS FORWARD] Step {self.debug_step_counter}: labels shape={labels.shape}, valid={valid_labels}, has_logits={'logits' in model_outputs}")

        losses = {}
        
        # RADICAL SIMPLIFICATION: Use ONLY contrastive loss (like CLIP)
        # This is proven to work and actually learns meaningful representations
        # NO language modeling (too complex, may not converge)
        # NO multi-objective confusion
        
        # PRIMARY LOSS: Image-Text Contrastive Learning (CLIP-style)
        if images is not None and 'contrastive_features' in model_outputs:
            contrastive_feats = model_outputs['contrastive_features']
            if 'text_features' in contrastive_feats and 'image_features' in contrastive_feats:
                # Use CLIP's proven approach: symmetric contrastive loss
                # DIAGNOSTIC: Check feature norms (every 100 steps)
                if self.debug_step_counter % 100 == 0:
                    text_norm = contrastive_feats['text_features'].norm(dim=-1).mean().item()
                    image_norm = contrastive_feats['image_features'].norm(dim=-1).mean().item()
                    print(f"🔍 Feature Norms (Step {self.debug_step_counter}): "
                          f"Text={text_norm:.4f}, Image={image_norm:.4f}")
                    if text_norm < 0.1 or image_norm < 0.1:
                        print(f"⚠️  WARNING: Very small feature norms detected! Features may be collapsed.")
                
                contrastive_loss = self.contrastive_loss(
                    contrastive_feats['text_features'],
                    contrastive_feats['image_features'],
                    temperature=contrastive_feats.get('temperature', 0.01)
                )
                losses['contrastive'] = contrastive_loss
                
                if should_debug:
                    print(f"DEBUG: Contrastive loss (CLIP-style) = {contrastive_loss.item():.6f}")

        # FIXED: Proper vision-text contrastive loss (like CLIP/FIBER)
        if images is not None and 'text_features' in model_outputs and 'vision_features' in model_outputs:
            try:
                text_features = model_outputs['text_features']
                vision_features = model_outputs['vision_features']

                # Compute proper contrastive loss
                vision_text_loss = self.vision_text_alignment_loss(text_features, vision_features)

                if not torch.isnan(vision_text_loss) and not torch.isinf(vision_text_loss):
                    losses['vision_text_alignment'] = vision_text_loss
                    if should_debug:
                        print(f"DEBUG: Vision-text alignment loss = {vision_text_loss.item():.6f}")
            except Exception as e:
                if should_debug:
                    print(f"WARNING: Vision-text loss computation failed: {e}")

        # Robot selection loss
        if 'robot_selection' in model_outputs and model_outputs['robot_selection'] is not None and target_robot is not None:
            try:
                robot_loss = self.robot_selection_accuracy_loss(model_outputs['robot_selection'], target_robot)
                if not torch.isnan(robot_loss) and not torch.isinf(robot_loss):
                    losses['robot_selection'] = robot_loss
                    if should_debug:
                        print(f"DEBUG: Robot selection loss = {robot_loss.item():.6f}")
            except Exception as e:
                if should_debug:
                    print(f"WARNING: Robot selection loss computation failed: {e}")

        # ADAPTIVE MULTI-STAGE TRAINING STRATEGY:
        # Stage 1 (steps 0-2000): Focus on contrastive learning (get model to learn SOMETHING)
        # Stage 2 (steps 2000-5000): Add robot selection (reasoning-based task selection)
        # Stage 3 (steps 5000+): Full training with all objectives
        
        # Determine training stage based on global step counter
        current_step = self.global_step
        
        # Track stage transitions
        if not hasattr(self, 'current_stage'):
            self.current_stage = 1
        
        if current_step < 2000:
            # STAGE 1: Pure contrastive learning (like CLIP warm-start)
            contrastive_weight = 1.0
            robot_weight = 0.0  # Disabled
            stage_name = "Stage 1: Contrastive Warm-start"
            new_stage = 1
        elif current_step < 5000:
            # STAGE 2: Add robot selection (reasoning starts learning)
            contrastive_weight = 0.7
            robot_weight = 0.5  # Now active
            stage_name = "Stage 2: Add Robot Reasoning"
            new_stage = 2
        else:
            # STAGE 3: Full multi-task learning
            contrastive_weight = 0.5
            robot_weight = 1.0  # Full importance
            stage_name = "Stage 3: Full Multi-Task"
            new_stage = 3
        
        # Log stage transitions
        if new_stage != self.current_stage:
            print(f"\n{'='*80}")
            print(f"🚀 TRAINING STAGE TRANSITION: Stage {self.current_stage} → Stage {new_stage}")
            print(f"   {stage_name}")
            print(f"   Step: {current_step}")
            print(f"   Contrastive weight: {contrastive_weight}")
            print(f"   Robot weight: {robot_weight}")
            print(f"{'='*80}\n")
            self.current_stage = new_stage
        
        # Build total loss based on current stage
        if 'contrastive' in losses:
            total_loss = losses['contrastive'] * contrastive_weight
            
            if should_debug:
                print(f"DEBUG: {stage_name} (step {current_step})")
                print(f"DEBUG: Contrastive loss = {losses['contrastive'].item():.6f} (weight={contrastive_weight})")
            
            # Add robot selection if available and weight > 0
            if 'robot_selection' in losses and robot_weight > 0:
                total_loss = total_loss + losses['robot_selection'] * robot_weight
                if should_debug:
                    print(f"DEBUG: Robot selection loss = {losses['robot_selection'].item():.6f} (weight={robot_weight})")
            elif should_debug and 'robot_selection' in losses:
                print(f"DEBUG: Robot selection loss = {losses['robot_selection'].item():.6f} (DISABLED in this stage)")
                    
        elif 'robot_selection' in losses:
            # Pure robot selection training (when no images available)
            total_loss = losses['robot_selection']
            if should_debug:
                print(f"DEBUG: Pure robot selection loss = {total_loss.item():.6f}")
        else:
            # No valid loss - should not happen
            total_loss = torch.tensor(1.0, device=labels.device, requires_grad=True)
            if should_debug:
                print("ERROR: No losses computed! Using dummy loss = 1.0")

        # FIXED: Only check for NaN/Inf, don't artificially inflate loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if should_debug:
                print(f"ERROR: Total loss became NaN/Inf! Falling back to language modeling loss only")
            total_loss = losses.get('language_modeling', torch.tensor(0.0, device=labels.device, requires_grad=True))

            # If even language modeling loss is NaN, use zero
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
                if should_debug:
                    print("ERROR: Even language modeling loss is NaN! Using zero loss")

        if should_debug:
            print(f"DEBUG: Final total loss = {total_loss.item():.6f}")

        # Combine all information
        loss_dict = {
            'total_loss': total_loss,
            **losses
        }

        return total_loss, loss_dict

    def compute_fiber_contrastive_loss(self, text_features: torch.Tensor,
                                     image_features: torch.Tensor,
                                     temperature: torch.Tensor) -> torch.Tensor:
        """FIBER-style Image-Text Contrastive (ITC) loss"""
        batch_size = text_features.size(0)

        # Compute similarity matrix
        logits_per_text = torch.matmul(text_features, image_features.T) / temperature
        logits_per_image = logits_per_text.T

        # Labels for contrastive learning (diagonal matching)
        labels = torch.arange(batch_size, device=text_features.device)

        # Compute symmetric contrastive loss
        text_loss = F.cross_entropy(logits_per_text, labels)
        image_loss = F.cross_entropy(logits_per_image, labels)

        return (text_loss + image_loss) / 2

class PerformanceTracker:
    """Track training performance across different modalities"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.step_count = 0

    def update(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        self.step_count += 1

        for metric_name, value in metrics.items():
            if not (torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value))):
                self.metrics_history[metric_name].append(value)

    def get_recent_average(self, metric_name: str, steps: int = 100) -> float:
        """Get recent average of a metric"""
        if metric_name not in self.metrics_history:
            return 0.0

        history = list(self.metrics_history[metric_name])
        if len(history) == 0:
            return 0.0

        recent_history = history[-steps:] if len(history) >= steps else history
        return sum(recent_history) / len(recent_history)

    def get_trend(self, metric_name: str, steps: int = 100) -> str:
        """Get trend direction for a metric"""
        if len(self.metrics_history[metric_name]) < 20:
            return "stable"

        history = list(self.metrics_history[metric_name])
        recent = history[-steps//2:] if len(history) >= steps else history[len(history)//2:]
        older = history[-steps:-steps//2] if len(history) >= steps else history[:len(history)//2]

        if not recent or not older:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        relative_change = abs(recent_avg - older_avg) / (older_avg + 1e-8)

        if relative_change < 0.05:
            return "stable"
        elif recent_avg > older_avg:
            return "increasing"
        else:
            return "decreasing"

    def get_summary(self) -> Dict:
        """Get performance summary"""
        summary = {}

        for metric_name in self.metrics_history.keys():
            summary[metric_name] = {
                'current': self.get_recent_average(metric_name, 10),
                'recent_100': self.get_recent_average(metric_name, 100),
                'trend': self.get_trend(metric_name),
                'history_length': len(self.metrics_history[metric_name])
            }

        return summary

# Memory-efficient training utilities for embedded systems
class EmbeddedTrainingUtils:
    """Utilities for memory-efficient training on resource-constrained systems"""

    @staticmethod
    def compute_memory_usage():
        """Compute current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024)
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                'rss_mb': 0.0,
                'vms_mb': 0.0
            }

    @staticmethod
    def gradient_accumulation_steps(target_batch_size: int, max_memory_mb: int = 512) -> int:
        """Calculate optimal gradient accumulation steps for memory constraints"""
        # Rough heuristic: each sample uses ~1MB during training
        max_batch_size = max(1, max_memory_mb // 4)  # Conservative estimate
        return max(1, target_batch_size // max_batch_size)

    @staticmethod
    def should_checkpoint_gradients(model_size_mb: float, available_memory_mb: float) -> bool:
        """Determine if gradient checkpointing should be used"""
        return model_size_mb * 2 > available_memory_mb * 0.7  # Use 70% threshold
