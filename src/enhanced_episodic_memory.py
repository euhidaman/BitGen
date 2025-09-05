"""
Enhanced Episodic Memory for BitGen Model
Based on Larimar's Episodic Memory Control with Multi-Modal Extensions
Supports external storage, compression, and advanced memory management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import time
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

EPSILON = 1e-6

@dataclass
class EpisodicMemoryConfig:
    """Configuration for enhanced episodic memory"""
    memory_size: int = 64
    episode_dim: int = 192
    alpha: float = 0.1
    direct_writing: bool = True
    observation_noise_std: float = 1e-6
    external_storage: bool = True
    memory_storage_path: str = "./episodic_memory"
    compression_enabled: bool = True
    lazy_loading: bool = True
    max_memory_age: int = 10000
    memory_consolidation_threshold: float = 0.8
    cross_modal_fusion: bool = True
    vision_memory_weight: float = 0.3
    text_memory_weight: float = 0.7
    pseudoinverse_approx_steps: int = 3
    memory_decay_rate: float = 0.999
    importance_threshold: float = 0.1
    async_save: bool = True

class LarimarInspiredEpisodicMemory(nn.Module):
    """
    Enhanced Episodic Memory based on Larimar's architecture
    Extended for multi-modal (text + vision) representations with external storage
    """
    
    def __init__(self, config: EpisodicMemoryConfig):
        super().__init__()
        self.config = config
        
        # Core memory parameters with safe defaults
        self.memory_size = getattr(config, 'memory_size', 1000)
        self.episode_dim = getattr(config, 'episode_dim', 512)
        self.alpha = getattr(config, 'alpha', 0.1)
        self.direct_writing = getattr(config, 'direct_writing', True)
        self.observation_noise_std = getattr(config, 'observation_noise_std', 0.1)
        
        # External storage configuration
        self.external_storage = getattr(config, 'external_storage', False)
        self.memory_storage_path = Path(getattr(config, 'memory_storage_path', './external_memory'))
        self.memory_storage_path.mkdir(parents=True, exist_ok=True)
        self.compression_enabled = getattr(config, 'compression_enabled', True)
        self.lazy_loading = getattr(config, 'lazy_loading', True)
        
        # Memory state tracking
        self._memory_loaded = False
        self._memory_version = 1
        self._last_save_time = time.time()
        self._save_queue = queue.Queue() if getattr(config, 'async_save', False) else None
        self._save_executor = ThreadPoolExecutor(max_workers=1) if getattr(config, 'async_save', False) else None
        
        # Larimar-inspired memory parameters (learnable)
        self.register_parameter('memory_mean_prior', 
                              nn.Parameter(torch.randn(self.memory_size, self.episode_dim) * 0.02))
        self.register_parameter('memory_logvar_prior', 
                              nn.Parameter(torch.zeros(1)))
        
        # Multi-modal memory components
        if getattr(config, 'cross_modal_fusion', False):
            # Separate memory banks for different modalities
            self.text_memory_weight = getattr(config, 'text_memory_weight', 0.7)
            self.vision_memory_weight = getattr(config, 'vision_memory_weight', 0.3)
            
            # Cross-modal fusion networks
            self.text_to_memory = nn.Sequential(
                nn.Linear(self.episode_dim, self.episode_dim),
                nn.LayerNorm(self.episode_dim),
                nn.GELU(),
                nn.Linear(self.episode_dim, self.episode_dim)
            )
            
            self.vision_to_memory = nn.Sequential(
                nn.Linear(self.episode_dim, self.episode_dim),
                nn.LayerNorm(self.episode_dim),
                nn.GELU(),
                nn.Linear(self.episode_dim, self.episode_dim)
            )
            
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=self.episode_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # Enhanced memory access networks (Larimar-style)
        self.query_encoder = nn.Sequential(
            nn.Linear(self.episode_dim, self.episode_dim * 2),
            nn.LayerNorm(self.episode_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.episode_dim * 2, self.episode_dim)
        )
        
        self.key_encoder = nn.Sequential(
            nn.Linear(self.episode_dim, self.episode_dim * 2),
            nn.LayerNorm(self.episode_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.episode_dim * 2, self.episode_dim)
        )
        
        self.value_encoder = nn.Sequential(
            nn.Linear(self.episode_dim, self.episode_dim * 2),
            nn.LayerNorm(self.episode_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.episode_dim * 2, self.episode_dim)
        )
        
        # Larimar-style weight computation networks
        self.w_logvar_network = nn.Sequential(
            nn.Linear(self.episode_dim + self.memory_size * self.episode_dim, self.memory_size),
            nn.LayerNorm(self.memory_size),
            nn.Tanh()
        )
        
        # Memory consolidation network (for memory compression)
        self.consolidation_network = nn.Sequential(
            nn.Linear(self.episode_dim, self.episode_dim),
            nn.LayerNorm(self.episode_dim),
            nn.GELU(),
            nn.Linear(self.episode_dim, self.episode_dim)
        )
        
        # Memory state buffers (will be loaded from external storage if available)
        if not (self.external_storage and self.lazy_loading):
            self._initialize_memory_buffers()
        else:
            self._memory_data = None
            self._memory_metadata = None
        
        # Temperature parameter for attention sharpening
        self.register_parameter('attention_temperature', nn.Parameter(torch.tensor(1.0)))
        
        # Memory importance scoring
        self.importance_network = nn.Sequential(
            nn.Linear(self.episode_dim, self.episode_dim // 2),
            nn.ReLU(),
            nn.Linear(self.episode_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"🧠 Enhanced Episodic Memory initialized")
        logger.info(f"  • Memory size: {self.memory_size}")
        logger.info(f"  • Episode dimension: {self.episode_dim}")
        logger.info(f"  • External storage: {self.external_storage}")
        logger.info(f"  • Storage path: {self.memory_storage_path}")
        logger.info(f"  • Cross-modal fusion: {getattr(config, 'cross_modal_fusion', False)}")
        logger.info(f"  • Lazy loading: {self.lazy_loading}")

    def _initialize_memory_buffers(self):
        """Initialize memory buffers in device memory"""
        # Check if already initialized to prevent double registration
        if hasattr(self, 'memory_mean') and self.memory_mean is not None:
            logger.info("Memory buffers already initialized, skipping...")
            self._memory_loaded = True
            return
        
        # Determine device from existing parameters
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        
        # Larimar-style memory initialization on correct device
        self.register_buffer('memory_mean', 
                           torch.randn(self.memory_size, self.episode_dim, device=device) * 0.02)
        self.register_buffer('memory_logvar', 
                           torch.zeros(self.memory_size, self.episode_dim, device=device))
        self.register_buffer('memory_cov', 
                           torch.eye(self.memory_size, device=device).unsqueeze(0))  # [1, memory_size, memory_size]
        
        # Additional tracking buffers on correct device
        self.register_buffer('memory_age', torch.zeros(self.memory_size, device=device))
        self.register_buffer('memory_usage', torch.zeros(self.memory_size, device=device))
        self.register_buffer('memory_importance', torch.ones(self.memory_size, device=device))
        self.register_buffer('memory_quality', torch.zeros(self.memory_size, device=device))
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long, device=device))
        
        # Cross-modal memory tracking on correct device
        if getattr(self.config, 'cross_modal_fusion', False):
            self.register_buffer('text_memory_contributions', torch.zeros(self.memory_size, device=device))
            self.register_buffer('vision_memory_contributions', torch.zeros(self.memory_size, device=device))
        
        # Mark as loaded to prevent re-initialization
        self._memory_loaded = True
        logger.info(f"Memory buffers initialized successfully on {device}: {self.memory_size} slots, dim={self.episode_dim}")

    def _ensure_memory_loaded(self):
        """Ensure memory is loaded into device memory"""
        # If already loaded, skip
        if self._memory_loaded and hasattr(self, 'memory_mean') and self.memory_mean is not None:
            return
            
        if self.external_storage and self.lazy_loading and not self._memory_loaded:
            # Try to load external memory, if fails, initialize buffers
            if not self.load_external_memory():
                logger.info("No external memory found, initializing new memory buffers")
                self._initialize_memory_buffers()
        elif not hasattr(self, 'memory_mean') or self.memory_mean is None:
            self._initialize_memory_buffers()

    def _get_prior_memory_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prior memory state distribution (Larimar-style)"""
        # Prior mean: [batch_size, memory_size, episode_dim]
        prior_mean = self.memory_mean_prior.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Prior covariance: [batch_size, memory_size, memory_size]
        prior_var = torch.exp(self.memory_logvar_prior) + EPSILON
        prior_cov = torch.diag(torch.ones(self.memory_size, device=self.memory_mean_prior.device) * prior_var)
        prior_cov = prior_cov.unsqueeze(0).expand(batch_size, -1, -1)
        
        return prior_mean, prior_cov

    def _sample_memory(self, memory_state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Sample memory from the current state distribution"""
        memory_mean, memory_cov = memory_state
        return memory_mean  # Use mean for deterministic sampling

    def _solve_w_mean(self, z: torch.Tensor, M: torch.Tensor, pseudoinverse: bool = True) -> torch.Tensor:
        """
        Solve for attention weights (Larimar-style)
        
        Args:
            z: Input encoding [episode_size, batch_size, episode_dim]
            M: Memory matrix [batch_size, memory_size, episode_dim]
            pseudoinverse: Whether to use pseudoinverse approximation
        
        Returns:
            w: Attention weights [episode_size, batch_size, memory_size]
        """
        episode_size, batch_size, episode_dim = z.shape
        
        if pseudoinverse:
            # Use pseudoinverse for more stable computation
            # M: [batch_size, memory_size, episode_dim]
            # We want to solve: z * M^T = w, so w = z * pinv(M^T)
            # M^T: [batch_size, episode_dim, memory_size]
            M_T = M.transpose(1, 2)  # [batch_size, episode_dim, memory_size]
            
            # z_transposed: [batch_size, episode_size, episode_dim]
            z_transposed = z.transpose(0, 1)
            
            # Compute pseudoinverse for each batch element separately
            w_list = []
            for i in range(batch_size):
                # M_T_i: [episode_dim, memory_size], z_i: [episode_size, episode_dim]
                M_T_i = M_T[i]  # [episode_dim, memory_size]
                z_i = z_transposed[i]  # [episode_size, episode_dim]
                
                # Compute pseudoinverse: pinv(M_T_i) has shape [memory_size, episode_dim]
                try:
                    M_T_i_pinv = torch.pinverse(M_T_i)  # [memory_size, episode_dim]
                    # z_i @ M_T_i_pinv.T = [episode_size, episode_dim] @ [episode_dim, memory_size] = [episode_size, memory_size]
                    w_i = torch.mm(z_i, M_T_i_pinv.T)  # [episode_size, memory_size]
                except RuntimeError:
                    # Fallback to least squares if pinverse fails
                    w_i = torch.linalg.lstsq(M_T_i.T, z_i.T).solution.T  # [episode_size, memory_size]
                
                w_list.append(w_i)
            
            w = torch.stack(w_list, dim=1)  # [episode_size, batch_size, memory_size]
        else:
            # Direct least squares solution
            M_flat = M.view(batch_size, -1)  # [batch_size, memory_size * episode_dim]
            z_flat = z.view(episode_size * batch_size, episode_dim)
            
            # Compute weights using learned network
            combined_input = torch.cat([
                z_flat.unsqueeze(1).expand(-1, self.memory_size, -1).reshape(-1, episode_dim),
                M_flat.unsqueeze(0).expand(episode_size, -1, -1).reshape(-1, self.memory_size * episode_dim)
            ], dim=1)
            
            w_logvar = self.w_logvar_network(combined_input)
            w = torch.softmax(w_logvar / self.attention_temperature, dim=-1)
            w = w.view(episode_size, batch_size, self.memory_size)
        
        return w

    def _update_memory_larimar_style(
        self, 
        old_memory_state: Tuple[torch.Tensor, torch.Tensor],
        w: torch.Tensor, 
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update memory using Larimar's Gaussian Process Memory approach
        
        Args:
            old_memory_state: (memory_mean, memory_cov)
            w: Attention weights [episode_size, batch_size, memory_size]
            z: Input encodings [episode_size, batch_size, episode_dim]
        """
        old_mean, old_cov = old_memory_state
        episode_size, batch_size = w.shape[:2]
        
        # Process each timestep
        new_mean, new_cov = old_mean, old_cov
        
        for t in range(episode_size):
            w_t = w[t:t+1]  # [1, batch_size, memory_size]
            z_t = z[t:t+1]  # [1, batch_size, episode_dim]
            
            # Compute prediction error
            predicted = torch.bmm(w_t.transpose(0, 1), new_mean)  # [batch_size, 1, episode_dim]
            delta = z_t.transpose(0, 1) - predicted  # [batch_size, 1, episode_dim]
            
            # Compute Kalman gain-like update
            wU = torch.bmm(w_t.transpose(0, 1), new_cov)  # [batch_size, 1, memory_size]
            wUw = torch.bmm(wU, w_t.transpose(0, 1).transpose(1, 2))  # [batch_size, 1, 1]
            
            sigma_z = wUw + self.observation_noise_std**2 * torch.eye(1, device=wUw.device)
            
            # Avoid division by zero
            sigma_z = sigma_z + EPSILON
            
            c_z = wU / sigma_z  # [batch_size, 1, memory_size]
            
            # Update memory mean
            update_term = torch.bmm(c_z.transpose(1, 2), delta)  # [batch_size, memory_size, episode_dim]
            new_mean = new_mean + update_term
            
            # Update memory covariance
            cov_update = torch.bmm(c_z.transpose(1, 2), wU)  # [batch_size, memory_size, memory_size]
            new_cov = new_cov - cov_update
        
        return new_mean, new_cov

    def _compute_kl_divergence(
        self, 
        prior_state: Tuple[torch.Tensor, torch.Tensor],
        posterior_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence between prior and posterior memory distributions"""
        prior_mean, prior_cov = prior_state
        posterior_mean, posterior_cov = posterior_state
        
        # Simplified KL divergence computation for Gaussian distributions
        mean_diff = posterior_mean - prior_mean
        
        # Compute trace terms (simplified)
        trace_term = torch.trace(torch.bmm(torch.inverse(prior_cov + EPSILON * torch.eye(self.memory_size, device=prior_cov.device)), posterior_cov)).mean()
        
        # Mean difference term
        mean_term = torch.bmm(
            mean_diff.unsqueeze(-2), 
            torch.bmm(torch.inverse(prior_cov + EPSILON * torch.eye(self.memory_size, device=prior_cov.device)), 
                     mean_diff.unsqueeze(-1))
        ).mean()
        
        # Log determinant term (simplified)
        logdet_term = torch.logdet(prior_cov + EPSILON).mean() - torch.logdet(posterior_cov + EPSILON).mean()
        
        kl_div = 0.5 * (trace_term + mean_term + logdet_term - self.memory_size)
        
        return kl_div

    def write_to_memory(
        self, 
        episodes: torch.Tensor,
        text_features: torch.Tensor = None,
        vision_features: torch.Tensor = None,
        importance_scores: torch.Tensor = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Write episodes to memory with multi-modal support
        
        Args:
            episodes: Episode encodings [episode_size, batch_size, episode_dim]
            text_features: Text component features [episode_size, batch_size, episode_dim]
            vision_features: Vision component features [episode_size, batch_size, episode_dim]
            importance_scores: Importance scores for episodes [episode_size, batch_size]
        
        Returns:
            (updated_memory_state, kl_divergence)
        """
        self._ensure_memory_loaded()
        
        episode_size, batch_size = episodes.shape[:2]
        
        # Get prior memory state
        prior_state = self._get_prior_memory_state(batch_size)
        
        # Current memory state
        current_state = (
            self.memory_mean.unsqueeze(0).expand(batch_size, -1, -1),
            self.memory_cov.expand(batch_size, -1, -1)
        )
        
        # Process multi-modal features if provided
        if getattr(self.config, 'cross_modal_fusion', False) and text_features is not None and vision_features is not None:
            # Encode modality-specific features
            text_encoded = self.text_to_memory(text_features)
            vision_encoded = self.vision_to_memory(vision_features)
            
            # Cross-modal attention fusion
            fused_features, cross_attn_weights = self.cross_modal_attention(
                text_encoded.view(-1, self.episode_dim).unsqueeze(0),
                vision_encoded.view(-1, self.episode_dim).unsqueeze(0),
                vision_encoded.view(-1, self.episode_dim).unsqueeze(0)
            )
            
            # Combine with original episodes
            fused_features = fused_features.view(episode_size, batch_size, self.episode_dim)
            episodes = (self.text_memory_weight * text_encoded + 
                       self.vision_memory_weight * vision_encoded + 
                       0.1 * fused_features)
        
        # Add noise for robustness
        noise = torch.randn_like(episodes) * self.observation_noise_std
        episodes_noisy = episodes + noise
        
        if self.direct_writing:
            # Direct writing approach (faster)
            w = self._solve_w_mean(episodes_noisy, current_state[0], pseudoinverse=True)
            
            # Approximate pseudoinverse for direct update
            w_pseudo_inv = self._approx_pseudoinverse(
                w.transpose(0, 1), 
                iterative_steps=getattr(self.config, 'pseudoinverse_approx_steps', 10)
            )
            
            # Direct memory update
            new_memory_mean = torch.bmm(w_pseudo_inv, episodes_noisy.transpose(0, 1))
            new_memory_state = (new_memory_mean, current_state[1])
        else:
            # Iterative Larimar-style update
            new_memory_state = self._update_memory_larimar_style(current_state, None, episodes_noisy)
        
        # Update memory buffers
        self.memory_mean.data = new_memory_state[0].mean(dim=0)
        if new_memory_state[1].dim() == 3:
            self.memory_cov.data = new_memory_state[1].mean(dim=0).unsqueeze(0)
        
        # Update tracking information
        self.update_count += 1
        self.memory_age += 1
        
        # Update importance scores if provided
        if importance_scores is not None:
            avg_importance = importance_scores.mean(dim=0)  # [batch_size]
            self.memory_importance.data = (self.memory_importance * 0.9 + 
                                         avg_importance.mean() * 0.1)
        
        # Compute KL divergence
        kl_div = self._compute_kl_divergence(prior_state, new_memory_state)
        
        # Schedule asynchronous save if enabled
        if self.external_storage and getattr(self.config, 'async_save', False):
            self._schedule_async_save()
        
        return new_memory_state, kl_div

    def read_from_memory(
        self, 
        query: torch.Tensor,
        text_query: torch.Tensor = None,
        vision_query: torch.Tensor = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Read from memory with multi-modal query support
        
        Args:
            query: Query encoding [episode_size, batch_size, episode_dim]
            text_query: Text component query [episode_size, batch_size, episode_dim]
            vision_query: Vision component query [episode_size, batch_size, episode_dim]
            return_attention: Whether to return attention weights
        
        Returns:
            (retrieved_content, attention_weights, metadata)
        """
        self._ensure_memory_loaded()
        
        episode_size, batch_size = query.shape[:2]
        
        # Process multi-modal queries if provided
        if getattr(self.config, 'cross_modal_fusion', False) and text_query is not None and vision_query is not None:
            # Encode modality-specific queries
            text_encoded = self.text_to_memory(text_query)
            vision_encoded = self.vision_to_memory(vision_query)
            
            # Fuse multi-modal queries
            query = (self.text_memory_weight * text_encoded + 
                    self.vision_memory_weight * vision_encoded + 
                    query) / 3.0
        
        # Current memory state
        memory_mean = self.memory_mean.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Encode query, key, and value
        query_encoded = self.query_encoder(query)  # [episode_size, batch_size, episode_dim]
        key_encoded = self.key_encoder(memory_mean.transpose(0, 1))  # [memory_size, batch_size, episode_dim]
        value_encoded = self.value_encoder(memory_mean.transpose(0, 1))  # [memory_size, batch_size, episode_dim]
        
        # Compute attention weights
        attention_scores = torch.bmm(
            query_encoded.transpose(0, 1),  # [batch_size, episode_size, episode_dim]
            key_encoded.transpose(0, 1).transpose(1, 2)  # [batch_size, episode_dim, memory_size]
        ) / np.sqrt(self.episode_dim)  # [batch_size, episode_size, memory_size]
        
        # Apply temperature scaling
        attention_weights = F.softmax(attention_scores / self.attention_temperature, dim=-1)
        
        # Retrieve content
        retrieved_content = torch.bmm(
            attention_weights,  # [batch_size, episode_size, memory_size]
            value_encoded.transpose(0, 1)  # [batch_size, memory_size, episode_dim]
        ).transpose(0, 1)  # [episode_size, batch_size, episode_dim]
        
        # Update memory usage statistics
        usage_update = attention_weights.mean(dim=1).mean(dim=0)  # [memory_size]
        self.memory_usage.data = self.memory_usage * getattr(self.config, 'memory_decay_rate', 0.99) + usage_update * (1 - getattr(self.config, 'memory_decay_rate', 0.99))
        
        # Create metadata
        metadata = {
            'attention_entropy': -(attention_weights * torch.log(attention_weights + EPSILON)).sum(dim=-1).mean().item(),
            'memory_usage': self.memory_usage.cpu().numpy(),
            'memory_age': self.memory_age.cpu().numpy(),
            'update_count': self.update_count.item(),
            'most_accessed_indices': self.memory_usage.topk(5).indices.cpu().numpy()
        }
        
        return retrieved_content, attention_weights.transpose(0, 1) if return_attention else None, metadata

    def _approx_pseudoinverse(self, matrix: torch.Tensor, iterative_steps: int = 3) -> torch.Tensor:
        """Approximate pseudoinverse using iterative method"""
        # Use SVD-based pseudoinverse for stability
        U, S, V = torch.svd(matrix)
        
        # Regularize singular values
        S_reg = S / (S**2 + EPSILON)
        
        # Reconstruct pseudoinverse
        pseudoinv = torch.bmm(torch.bmm(V, torch.diag_embed(S_reg)), U.transpose(-2, -1))
        
        return pseudoinv

    def consolidate_memory(self, threshold: float = None) -> Dict[str, Any]:
        """
        Consolidate memory by merging similar or less important memories
        """
        self._ensure_memory_loaded()
        
        threshold = threshold or getattr(self.config, 'memory_consolidation_threshold', 0.8)
        
        # Compute memory similarities
        memory_similarities = torch.mm(self.memory_mean, self.memory_mean.t())
        memory_similarities = F.softmax(memory_similarities, dim=-1)
        
        # Find memories to consolidate (high similarity, low importance)
        consolidation_candidates = []
        for i in range(self.memory_size):
            for j in range(i + 1, self.memory_size):
                if (memory_similarities[i, j] > threshold and 
                    self.memory_importance[i] < getattr(self.config, 'importance_threshold', 0.1) and
                    self.memory_importance[j] < self.config.importance_threshold):
                    consolidation_candidates.append((i, j, memory_similarities[i, j].item()))
        
        # Perform consolidation
        consolidated_count = 0
        for i, j, similarity in sorted(consolidation_candidates, key=lambda x: x[2], reverse=True):
            if self.memory_importance[i] > 0 and self.memory_importance[j] > 0:  # Both still active
                # Merge memories
                weight_i = self.memory_importance[i] / (self.memory_importance[i] + self.memory_importance[j])
                weight_j = 1 - weight_i
                
                self.memory_mean[i] = weight_i * self.memory_mean[i] + weight_j * self.memory_mean[j]
                self.memory_importance[i] = self.memory_importance[i] + self.memory_importance[j]
                
                # Mark the second memory as inactive
                self.memory_importance[j] = 0
                self.memory_mean[j] = torch.randn_like(self.memory_mean[j]) * 0.02
                self.memory_age[j] = 0
                self.memory_usage[j] = 0
                
                consolidated_count += 1
        
        consolidation_stats = {
            'consolidated_pairs': consolidated_count,
            'candidates_found': len(consolidation_candidates),
            'active_memories': (self.memory_importance > self.config.importance_threshold).sum().item(),
            'avg_memory_similarity': memory_similarities.mean().item()
        }
        
        logger.info(f"Memory consolidation completed: {consolidation_stats}")
        return consolidation_stats

    def save_external_memory(self, path: str = None, compress: bool = None) -> str:
        """Save memory to external storage with enhanced metadata"""
        save_path = Path(path) if path else self.memory_storage_path / f"episodic_memory_v{self._memory_version}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        use_compression = compress if compress is not None else self.compression_enabled
        
        # Ensure memory is loaded
        self._ensure_memory_loaded()
        
        # Prepare comprehensive memory data
        memory_data = {
            # Core memory state
            'memory_mean': self.memory_mean.cpu(),
            'memory_cov': self.memory_cov.cpu(),
            'memory_mean_prior': self.memory_mean_prior.cpu(),
            'memory_logvar_prior': self.memory_logvar_prior.cpu(),
            
            # Tracking information
            'memory_age': self.memory_age.cpu(),
            'memory_usage': self.memory_usage.cpu(),
            'memory_importance': self.memory_importance.cpu(),
            'memory_quality': self.memory_quality.cpu(),
            'update_count': self.update_count.cpu(),
            'attention_temperature': self.attention_temperature.cpu(),
            
            # Multi-modal tracking
            'text_memory_contributions': getattr(self, 'text_memory_contributions', torch.zeros(self.memory_size)).cpu(),
            'vision_memory_contributions': getattr(self, 'vision_memory_contributions', torch.zeros(self.memory_size)).cpu(),
            
            # Configuration and metadata
            'config': {
                'memory_size': self.memory_size,
                'episode_dim': self.episode_dim,
                'alpha': self.alpha,
                'direct_writing': self.direct_writing,
                'cross_modal_fusion': getattr(self.config, 'cross_modal_fusion', False),
                'text_memory_weight': self.text_memory_weight if hasattr(self, 'text_memory_weight') else 0.7,
                'vision_memory_weight': self.vision_memory_weight if hasattr(self, 'vision_memory_weight') else 0.3,
            },
            'metadata': {
                'version': self._memory_version,
                'save_timestamp': time.time(),
                'total_updates': self.update_count.item() if hasattr(self, 'update_count') else 0,
                'compression_enabled': use_compression,
                'device_info': str(self.memory_mean.device) if hasattr(self, 'memory_mean') else 'unknown'
            }
        }
        
        # Save with optional compression
        if use_compression:
            compressed_path = save_path.with_suffix('.pt.gz')
            with gzip.open(compressed_path, 'wb') as f:
                torch.save(memory_data, f)
            final_path = compressed_path
        else:
            torch.save(memory_data, save_path)
            final_path = save_path
        
        # Save human-readable metadata
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'config': memory_data['config'],
                'metadata': memory_data['metadata'],
                'statistics': {
                    'active_memories': (memory_data['memory_importance'] > self.config.importance_threshold).sum().item(),
                    'total_memory_usage': memory_data['memory_usage'].sum().item(),
                    'average_memory_age': memory_data['memory_age'].mean().item(),
                    'memory_utilization': (memory_data['memory_usage'] > 0).sum().item() / self.memory_size
                }
            }, f, indent=2)
        
        self._last_save_time = time.time()
        logger.info(f"✅ Episodic memory saved to: {final_path}")
        logger.info(f"📊 Memory statistics saved to: {metadata_path}")
        
        return str(final_path)

    def load_external_memory(self, path: str = None) -> bool:
        """Load memory from external storage"""
        if path:
            load_path = Path(path)
        else:
            # Find the latest version
            pattern = "episodic_memory_v*.pt*"
            candidates = list(self.memory_storage_path.glob(pattern))
            if not candidates:
                logger.warning("No external memory files found")
                return False
            load_path = max(candidates, key=lambda p: p.stat().st_mtime)
        
        if not load_path.exists():
            logger.warning(f"Memory file not found: {load_path}")
            return False
        
        try:
            # Load with compression support
            if load_path.suffix == '.gz':
                with gzip.open(load_path, 'rb') as f:
                    memory_data = torch.load(f, map_location='cpu')
            else:
                memory_data = torch.load(load_path, map_location='cpu')
            
            # Restore memory state
            device = next(self.parameters()).device
            
            self.register_buffer('memory_mean', memory_data['memory_mean'].to(device))
            self.register_buffer('memory_cov', memory_data['memory_cov'].to(device))
            self.memory_mean_prior.data = memory_data['memory_mean_prior'].to(device)
            self.memory_logvar_prior.data = memory_data['memory_logvar_prior'].to(device)
            
            # Restore tracking information
            self.register_buffer('memory_age', memory_data['memory_age'].to(device))
            self.register_buffer('memory_usage', memory_data['memory_usage'].to(device))
            self.register_buffer('memory_importance', memory_data['memory_importance'].to(device))
            self.register_buffer('memory_quality', memory_data['memory_quality'].to(device))
            self.register_buffer('update_count', memory_data['update_count'].to(device))
            self.attention_temperature.data = memory_data['attention_temperature'].to(device)
            
            # Restore multi-modal tracking if available
            if 'text_memory_contributions' in memory_data:
                self.register_buffer('text_memory_contributions', 
                                   memory_data['text_memory_contributions'].to(device))
            if 'vision_memory_contributions' in memory_data:
                self.register_buffer('vision_memory_contributions', 
                                   memory_data['vision_memory_contributions'].to(device))
            
            self._memory_loaded = True
            self._memory_version = memory_data['metadata']['version']
            
            logger.info(f"✅ Episodic memory loaded from: {load_path}")
            logger.info(f"📊 Memory version: {self._memory_version}")
            logger.info(f"📊 Total updates: {memory_data['metadata'].get('total_updates', 0)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load memory from {load_path}: {e}")
            return False

    def _schedule_async_save(self):
        """Schedule asynchronous memory save"""
        if self._save_executor and time.time() - self._last_save_time > 60:  # Save at most once per minute
            self._save_executor.submit(self.save_external_memory)

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        self._ensure_memory_loaded()
        
        stats = {
            'memory_size': self.memory_size,
            'episode_dim': self.episode_dim,
            'active_memories': (self.memory_importance > self.config.importance_threshold).sum().item(),
            'total_updates': self.update_count.item(),
            'average_memory_age': self.memory_age.mean().item(),
            'memory_utilization': (self.memory_usage > 0).sum().item() / self.memory_size,
            'most_important_memories': self.memory_importance.topk(5).indices.cpu().numpy().tolist(),
            'most_used_memories': self.memory_usage.topk(5).indices.cpu().numpy().tolist(),
            'memory_distribution': {
                'importance_mean': self.memory_importance.mean().item(),
                'importance_std': self.memory_importance.std().item(),
                'usage_mean': self.memory_usage.mean().item(),
                'usage_std': self.memory_usage.std().item(),
                'age_mean': self.memory_age.mean().item(),
                'age_std': self.memory_age.std().item(),
            }
        }
        
        if hasattr(self, 'text_memory_contributions'):
            stats['modality_contributions'] = {
                'text_total': self.text_memory_contributions.sum().item(),
                'vision_total': self.vision_memory_contributions.sum().item(),
                'text_avg': self.text_memory_contributions.mean().item(),
                'vision_avg': self.vision_memory_contributions.mean().item(),
            }
        
        return stats

    def reset_memory(self, keep_important: bool = True):
        """Reset memory state"""
        if keep_important and hasattr(self, 'memory_importance'):
            # Keep only important memories
            important_mask = self.memory_importance > self.config.importance_threshold
            if important_mask.any():
                # Reset only unimportant memories
                self.memory_mean[~important_mask] = torch.randn_like(self.memory_mean[~important_mask]) * 0.02
                self.memory_age[~important_mask] = 0
                self.memory_usage[~important_mask] = 0
                self.memory_quality[~important_mask] = 0
            else:
                # If no important memories, reset all
                self._initialize_memory_buffers()
        else:
            # Complete reset
            self._initialize_memory_buffers()
        
        logger.info(f"Memory reset completed (keep_important={keep_important})")

def create_enhanced_episodic_memory(config: Dict = None) -> LarimarInspiredEpisodicMemory:
    """Factory function to create enhanced episodic memory"""
    memory_config = EpisodicMemoryConfig()
    if config:
        for key, value in config.items():
            if hasattr(memory_config, key):
                setattr(memory_config, key, value)
    
    return LarimarInspiredEpisodicMemory(memory_config)
