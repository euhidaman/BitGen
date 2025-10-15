"""
Loss Stabilizer for BitGen - Ensures consistent loss decrease
Uses sliding window to detect loss increases and applies corrective measures
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Deque
from collections import deque
import logging

logger = logging.getLogger(__name__)


class LossStabilizer:
    """
    Monitors training loss with sliding window and prevents sudden increases.
    Applies gentle interventions to keep training stable.
    """

    def __init__(
        self,
        window_size: int = 50,              # Steps to track for trend analysis
        increase_threshold: float = 0.05,    # Relative increase to trigger intervention (5%)
        smoothing_alpha: float = 0.1,        # EMA smoothing factor
        min_steps_between_checks: int = 10,  # Check every N steps
        lr_reduction_factor: float = 0.5,    # Reduce LR by this factor on increase
        min_lr: float = 1e-6                 # Minimum learning rate
    ):
        self.window_size = window_size
        self.increase_threshold = increase_threshold
        self.smoothing_alpha = smoothing_alpha
        self.min_steps_between_checks = min_steps_between_checks
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr

        # Tracking variables
        self.loss_history: Deque[float] = deque(maxlen=window_size)
        self.loss_ema: Optional[float] = None
        self.current_step = 0
        self.last_check_step = 0

        # Statistics
        self.interventions_count = 0
        self.loss_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0.0,
            'trend': 'stable'
        }

    def update_loss(
        self,
        loss: float,
        step: int,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, any]:
        """
        Update loss tracker and check for interventions

        Args:
            loss: Current training loss
            step: Current training step
            optimizer: Optimizer to adjust if needed

        Returns:
            Dict with intervention decisions and statistics
        """
        self.current_step = step
        self.loss_history.append(loss)

        # Update EMA
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = (
                self.smoothing_alpha * loss +
                (1 - self.smoothing_alpha) * self.loss_ema
            )

        # Update statistics
        self._update_statistics()

        # Check for loss increases periodically
        should_check = (step - self.last_check_step >= self.min_steps_between_checks)
        intervention_applied = False
        intervention_reason = None

        if should_check and len(self.loss_history) >= self.window_size:
            intervention_applied, intervention_reason = self._check_and_intervene(loss, optimizer)
            self.last_check_step = step

        return {
            'loss_ema': self.loss_ema,
            'intervention_applied': intervention_applied,
            'intervention_reason': intervention_reason,
            'loss_stats': self.loss_stats.copy(),
            'current_lr': optimizer.param_groups[0]['lr']
        }

    def _check_and_intervene(
        self,
        current_loss: float,
        optimizer: torch.optim.Optimizer
    ) -> tuple:
        """Check if loss is increasing and apply intervention"""

        # Calculate recent vs older trend
        recent_window = int(self.window_size * 0.3)  # Last 30% of window
        recent_losses = list(self.loss_history)[-recent_window:]
        older_losses = list(self.loss_history)[:recent_window]

        recent_mean = np.mean(recent_losses)
        older_mean = np.mean(older_losses)

        # Calculate relative increase
        if older_mean > 0:
            relative_change = (recent_mean - older_mean) / older_mean
        else:
            relative_change = 0.0

        # Check if loss increased beyond threshold
        if relative_change > self.increase_threshold:
            # Loss is increasing - apply intervention
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * self.lr_reduction_factor, self.min_lr)
            
            # Only reduce if we're above minimum
            if new_lr > self.min_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                self.interventions_count += 1
                reason = f"Loss increased by {relative_change:.2%} ({older_mean:.4f} → {recent_mean:.4f}). Reduced LR: {current_lr:.2e} → {new_lr:.2e}"
                logger.warning(f"🔧 Loss Stabilizer Intervention #{self.interventions_count}: {reason}")
                
                return True, reason
            else:
                reason = f"Loss increased by {relative_change:.2%}, but LR already at minimum ({self.min_lr:.2e})"
                logger.warning(f"⚠️  {reason}")
                return False, reason

        return False, None

    def _update_statistics(self):
        """Update loss statistics"""
        if len(self.loss_history) == 0:
            return

        losses = list(self.loss_history)
        self.loss_stats['min'] = min(losses)
        self.loss_stats['max'] = max(losses)
        self.loss_stats['mean'] = np.mean(losses)

        # Determine trend
        if len(losses) >= 10:
            first_half = np.mean(losses[:len(losses)//2])
            second_half = np.mean(losses[len(losses)//2:])
            
            if second_half < first_half * 0.95:
                self.loss_stats['trend'] = 'decreasing'
            elif second_half > first_half * 1.05:
                self.loss_stats['trend'] = 'increasing'
            else:
                self.loss_stats['trend'] = 'stable'

    def should_continue_training(self) -> bool:
        """
        Check if training should continue based on loss behavior
        
        Returns:
            True if training is healthy, False if loss is exploding/stuck
        """
        if len(self.loss_history) < self.window_size:
            return True

        # Check for exploding loss
        if self.loss_ema > 100.0:
            logger.error(f"❌ Loss exploded: {self.loss_ema:.2f} > 100")
            return False

        # Check for stuck training (no change in 50 steps)
        if len(self.loss_history) >= 50:
            recent_50 = list(self.loss_history)[-50:]
            std = np.std(recent_50)
            if std < 1e-6:
                logger.error(f"❌ Loss stuck at {np.mean(recent_50):.4f} (std={std:.2e})")
                return False

        return True

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_interventions': self.interventions_count,
            'loss_ema': self.loss_ema,
            'loss_stats': self.loss_stats.copy(),
            'window_size': len(self.loss_history)
        }
