"""
Larimar GPM (Generative Parametric Memory) for BitGen
Adapted from Larimar's episodic memory module for vision-language tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class GPM(nn.Module):
    """
    Generative Parametric Memory (GPM) from Larimar
    Implements Bayesian episodic memory with trainable parameters
    """
    
    def __init__(
        self,
        code_size: int,
        memory_size: int,
        direct_write: bool = True,
        ordering: bool = False,
        pseudoinverse_approx_step: int = 3,
        observation_noise_std: float = 0.0,
        identity: bool = False,
        w_logvar_setting: float = 0.0,
        deterministic: bool = False
    ):
        super().__init__()
        
        self._code_size = code_size  # C (embedding dimension)
        self._memory_size = memory_size  # K (number of memory slots)
        self._direct_write = direct_write
        self._ordering = ordering
        self._pseudoinverse_approx_step = pseudoinverse_approx_step
        self._observation_noise_std = observation_noise_std
        self._identity = identity
        self._w_logvar_setting = w_logvar_setting
        self._deterministic = deterministic
        
        # Trainable memory parameters
        self.memory_mean = nn.Parameter(torch.randn(memory_size, code_size))
        self.memory_logvar = nn.Parameter(torch.zeros(1) + w_logvar_setting)
        
        # Initialize memory with Xavier uniform
        nn.init.xavier_uniform_(self.memory_mean)
        
        # For tracking memory statistics
        self.register_buffer('memory_read_count', torch.zeros(memory_size))
        self.register_buffer('memory_write_count', torch.zeros(memory_size))
        self.register_buffer('memory_quality', torch.ones(memory_size))  # Quality scores for memories
        self.register_buffer('memory_age', torch.zeros(memory_size))  # Age of memories (for decay)
    
    def _get_prior_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prior mean and covariance for Bayesian inference
        
        Returns:
            prior_mean: [memory_size, code_size]
            prior_cov: [code_size, code_size]
        """
        prior_mean = self.memory_mean  # [K, C]
        
        # Compute covariance from logvar
        if self._identity:
            prior_cov = torch.eye(self._code_size, device=self.memory_mean.device)
        else:
            var = torch.exp(self.memory_logvar)
            prior_cov = var * torch.eye(self._code_size, device=self.memory_mean.device)
        
        return prior_mean, prior_cov
    
    def _sample_M(self, memory_state: Dict) -> torch.Tensor:
        """
        Sample memory matrix from distribution
        
        Args:
            memory_state: Dictionary containing memory statistics
        
        Returns:
            sampled_M: [memory_size, code_size]
        """
        if self._deterministic:
            return self.memory_mean
        
        # Sample from Gaussian distribution
        prior_mean, prior_cov = self._get_prior_params()
        
        # Efficient sampling using Cholesky decomposition
        try:
            L = torch.linalg.cholesky(prior_cov)
            eps = torch.randn(self._memory_size, self._code_size, device=self.memory_mean.device)
            sampled = prior_mean + eps @ L.T
        except RuntimeError:
            # Fallback to diagonal sampling if Cholesky fails
            std = torch.sqrt(torch.diag(prior_cov))
            eps = torch.randn(self._memory_size, self._code_size, device=self.memory_mean.device)
            sampled = prior_mean + eps * std.unsqueeze(0)
        
        return sampled
    
    def read(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, Dict]:
        """
        Read from memory using query
        
        Args:
            query: [batch_size, seq_len, code_size]
            top_k: Number of top memories to retrieve
        
        Returns:
            retrieved: [batch_size, seq_len, code_size]
            read_info: Dictionary with read statistics
        """
        batch_size, seq_len, code_size = query.shape
        
        # Sample memory matrix
        M = self._sample_M({})  # [K, C]
        
        # Compute similarity scores (cosine similarity)
        query_flat = query.view(-1, code_size)  # [B*L, C]
        query_norm = F.normalize(query_flat, p=2, dim=-1)
        memory_norm = F.normalize(M, p=2, dim=-1)
        
        similarity = torch.matmul(query_norm, memory_norm.T)  # [B*L, K]
        
        # Weight similarity by memory quality (Larimar enhancement)
        quality_weighted_similarity = similarity * self.memory_quality.unsqueeze(0)
        
        # Get top-k memories
        top_k = min(top_k, self._memory_size)
        top_k_scores, top_k_indices = torch.topk(quality_weighted_similarity, k=top_k, dim=-1)
        
        # Weighted retrieval
        weights = F.softmax(top_k_scores, dim=-1)  # [B*L, top_k]
        
        # Gather top-k memories
        top_k_memories = M[top_k_indices]  # [B*L, top_k, C]
        
        # Weighted combination
        retrieved = torch.sum(weights.unsqueeze(-1) * top_k_memories, dim=1)  # [B*L, C]
        retrieved = retrieved.view(batch_size, seq_len, code_size)
        
        # Update read counts
        with torch.no_grad():
            for idx in top_k_indices.view(-1).unique():
                self.memory_read_count[idx] += 1
        
        read_info = {
            'top_k_indices': top_k_indices.view(batch_size, seq_len, top_k),
            'top_k_scores': top_k_scores.view(batch_size, seq_len, top_k),
            'weights': weights.view(batch_size, seq_len, top_k)
        }
        
        return retrieved, read_info
    
    def write(self, value: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Dict:
        """
        Write to memory (direct write mode)
        
        Args:
            value: [batch_size, seq_len, code_size]
            importance: Optional importance weights [batch_size, seq_len]
        
        Returns:
            write_info: Dictionary with write statistics
        """
        if not self._direct_write:
            return {}
        
        batch_size, seq_len, code_size = value.shape
        value_flat = value.view(-1, code_size)  # [B*L, C]
        
        # Compute importance if not provided
        if importance is None:
            importance = torch.ones(batch_size, seq_len, device=value.device)
        importance_flat = importance.view(-1)  # [B*L]
        
        # Select top important values to write
        num_writes = min(batch_size * seq_len, self._memory_size)
        top_importance, write_indices = torch.topk(importance_flat, k=num_writes)
        
        # Get values to write
        values_to_write = value_flat[write_indices]  # [num_writes, C]
        
        # Find memory slots to update (least recently used)
        with torch.no_grad():
            lru_indices = torch.argsort(self.memory_write_count)[:num_writes]
        
        # Update memory (gradient flows through this!)
        for i, slot_idx in enumerate(lru_indices):
            # Exponential moving average update
            alpha = 0.1  # Learning rate for memory updates
            self.memory_mean.data[slot_idx] = (
                (1 - alpha) * self.memory_mean.data[slot_idx] +
                alpha * values_to_write[i].detach()
            )
            self.memory_write_count[slot_idx] += 1
        
        write_info = {
            'num_writes': num_writes,
            'write_slots': lru_indices,
            'importance_scores': top_importance
        }
        
        return write_info
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass: read from memory and optionally write
        
        Args:
            x: [batch_size, seq_len, code_size]
        
        Returns:
            output: [batch_size, seq_len, code_size]
            memory_info: Dictionary with memory statistics
        """
        # Read from memory
        retrieved, read_info = self.read(x, top_k=5)
        
        # Combine input with retrieved memory
        output = x + 0.5 * retrieved  # Residual connection with retrieved memory
        
        # Write to memory (if direct write enabled)
        write_info = {}
        if self._direct_write and self.training:
            # Compute importance based on L2 norm
            importance = torch.norm(x, p=2, dim=-1)  # [batch_size, seq_len]
            write_info = self.write(x, importance)
        
        memory_info = {
            **read_info,
            **write_info,
            'memory_mean_norm': torch.norm(self.memory_mean, p=2, dim=-1).mean().item(),
            'memory_logvar': self.memory_logvar.item()
        }
        
        return output, memory_info
    
    def get_memory_kl_loss(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior (for regularization)
        
        Returns:
            kl_loss: Scalar KL divergence
        """
        # Simple KL loss: regularize memory variance
        # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
        
        var = torch.exp(self.memory_logvar)
        mean_sq = torch.mean(self.memory_mean ** 2)
        
        kl_loss = 0.5 * (var + mean_sq - 1.0 - self.memory_logvar)
        
        return kl_loss.mean()
    
    def save_memory(self, path: str):
        """
        Save memory state to disk (Larimar-style external storage)
        
        Args:
            path: Path to save memory state
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        memory_state = {
            'memory_mean': self.memory_mean.cpu(),
            'memory_logvar': self.memory_logvar.cpu(),
            'memory_read_count': self.memory_read_count.cpu(),
            'memory_write_count': self.memory_write_count.cpu(),
            'memory_quality': self.memory_quality.cpu(),
            'memory_age': self.memory_age.cpu(),
            'config': {
                'code_size': self._code_size,
                'memory_size': self._memory_size,
                'direct_write': self._direct_write,
                'ordering': self._ordering,
                'deterministic': self._deterministic
            }
        }
        
        torch.save(memory_state, path)
        print(f"Memory state saved to {path}")
    
    def load_memory(self, path: str):
        """
        Load memory state from disk (Larimar-style external storage)
        
        Args:
            path: Path to load memory state from
        """
        import os
        if not os.path.exists(path):
            print(f"Warning: Memory state file not found at {path}")
            return
        
        memory_state = torch.load(path, map_location=self.memory_mean.device)
        
        # Restore memory parameters
        self.memory_mean.data = memory_state['memory_mean'].to(self.memory_mean.device)
        self.memory_logvar.data = memory_state['memory_logvar'].to(self.memory_logvar.device)
        self.memory_read_count.data = memory_state['memory_read_count'].to(self.memory_read_count.device)
        self.memory_write_count.data = memory_state['memory_write_count'].to(self.memory_write_count.device)
        self.memory_quality.data = memory_state['memory_quality'].to(self.memory_quality.device)
        self.memory_age.data = memory_state['memory_age'].to(self.memory_age.device)
        
        print(f"Memory state loaded from {path}")
        print(f"Memory size: {self._memory_size}, Code size: {self._code_size}")
        print(f"Total reads: {self.memory_read_count.sum().item()}, Total writes: {self.memory_write_count.sum().item()}")
    
    def update_memory_quality(self, indices: torch.Tensor, success_rate: float):
        """
        Update quality scores for memories based on usage success
        
        Args:
            indices: Indices of accessed memories
            success_rate: Success rate (0.0 to 1.0) for this access
        """
        with torch.no_grad():
            # Exponential moving average of quality
            alpha = 0.1
            for idx in indices.view(-1).unique():
                old_quality = self.memory_quality[idx]
                self.memory_quality[idx] = (1 - alpha) * old_quality + alpha * success_rate
    
    def decay_memory_age(self, decay_rate: float = 0.01):
        """
        Increment memory age and apply decay to old memories
        
        Args:
            decay_rate: Rate at which old memories decay
        """
        with torch.no_grad():
            self.memory_age += 1
            
            # Apply quality decay for old memories
            age_factor = torch.exp(-decay_rate * self.memory_age)
            self.memory_quality *= age_factor


class BitGenMemory(nn.Module):
    """
    BitGen-specific wrapper for Larimar GPM
    Adds projection layers for integration with BitGen architecture
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # GPM core
        self.gpm = GPM(
            code_size=config.memory_dim,
            memory_size=config.memory_size,
            direct_write=config.direct_writing,
            ordering=False,
            deterministic=False
        )
        
        # Projection layers
        self.input_proj = nn.Linear(config.embed_dim, config.memory_dim)
        self.output_proj = nn.Linear(config.memory_dim, config.embed_dim)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with projection
        
        Args:
            x: [batch_size, seq_len, embed_dim]
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
            memory_info: Dictionary with memory statistics
        """
        # Project to memory dimension
        x_proj = self.input_proj(x)  # [B, L, memory_dim]
        
        # GPM forward
        memory_output, memory_info = self.gpm(x_proj)
        
        # Project back to embedding dimension
        output = self.output_proj(memory_output)  # [B, L, embed_dim]
        
        # Residual connection with layer norm
        output = self.layer_norm(x + output)
        
        return output, memory_info
    
    def get_memory_kl_loss(self) -> torch.Tensor:
        """Get KL loss from GPM"""
        return self.gpm.get_memory_kl_loss()
    
    def save_memory(self, path: str):
        """Save memory state to disk"""
        self.gpm.save_memory(path)
    
    def load_memory(self, path: str):
        """Load memory state from disk"""
        self.gpm.load_memory(path)
    
    def update_memory_quality(self, indices: torch.Tensor, success_rate: float):
        """Update memory quality scores"""
        self.gpm.update_memory_quality(indices, success_rate)
    
    def decay_memory_age(self, decay_rate: float = 0.01):
        """Apply age-based decay to memory quality"""
        self.gpm.decay_memory_age(decay_rate)
