"""
Comprehensive WandB Monitoring for BitGen
Tracks all model components: attention, memory, quantization, cross-modal fusion, etc.
"""

import torch
import torch.nn as nn
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import io
from PIL import Image

class WandbMonitor:
    """Comprehensive monitoring system for BitGen training"""

    def __init__(self, log_freq: int = 100, log_attention_freq: int = 100):
        self.log_freq = log_freq
        self.log_attention_freq = log_attention_freq
        self.step_count = 0

        # Track statistics
        self.attention_stats = []
        self.memory_stats = []
        self.gradient_norms = []

    def log_training_step(self, metrics: Dict, model: nn.Module, step: int):
        """Log basic training metrics every step"""
        log_dict = {
            'train/loss': metrics.get('total_loss', 0),
            'train/learning_rate': metrics.get('learning_rate', 0),
            'train/gradient_norm': metrics.get('gradient_norm', 0),
            'step': step
        }

        # Loss breakdown
        if 'language_modeling' in metrics:
            log_dict['train/loss_lm'] = metrics['language_modeling']
        if 'contrastive_itc' in metrics:
            log_dict['train/loss_contrastive'] = metrics['contrastive_itc']
        if 'vision_alignment' in metrics:
            log_dict['train/loss_vision'] = metrics['vision_alignment']
        if 'robot_selection' in metrics:
            log_dict['train/loss_robot'] = metrics['robot_selection']

        # GPU metrics
        if torch.cuda.is_available():
            log_dict['system/gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            log_dict['system/gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            log_dict['system/gpu_memory_utilization'] = metrics.get('gpu_util', 0)

        # CPU metrics
        import psutil
        log_dict['system/cpu_memory_gb'] = psutil.virtual_memory().used / 1e9
        log_dict['system/cpu_percent'] = psutil.cpu_percent()

        # Training throughput
        if 'samples_per_sec' in metrics:
            log_dict['performance/samples_per_sec'] = metrics['samples_per_sec']
        if 'tokens_per_sec' in metrics:
            log_dict['performance/tokens_per_sec'] = metrics['tokens_per_sec']

        wandb.log(log_dict, step=step)

    def log_attention_patterns(self, attention_weights: torch.Tensor, step: int, layer_idx: int = 0):
        """Log attention heatmaps and statistics"""
        if step % self.log_attention_freq != 0:
            return

        with torch.no_grad():
            # attention_weights: [batch, num_heads, seq_len, seq_len]
            if attention_weights is None or attention_weights.numel() == 0:
                return

            # Take first sample in batch
            attn = attention_weights[0].cpu().numpy()  # [num_heads, seq_len, seq_len]

            # Compute attention entropy (measure of focus vs spread)
            attn_flat = attn.reshape(-1, attn.shape[-1])
            epsilon = 1e-10
            entropy = -np.sum(attn_flat * np.log(attn_flat + epsilon), axis=-1)
            avg_entropy = np.mean(entropy)

            # Compute attention concentration (max attention per query)
            max_attention = np.max(attn, axis=-1).mean()

            # Log metrics
            log_dict = {
                f'attention/layer_{layer_idx}_entropy': avg_entropy,
                f'attention/layer_{layer_idx}_max_attention': max_attention,
                f'attention/layer_{layer_idx}_sparsity': np.sum(attn < 0.01) / attn.size,
                'step': step
            }

            # Create attention heatmap for first head
            if step % (self.log_attention_freq * 5) == 0:  # Less frequent for images
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle(f'Layer {layer_idx} Attention Patterns (8 heads)', fontsize=14)

                for head_idx in range(min(8, attn.shape[0])):
                    ax = axes[head_idx // 4, head_idx % 4]
                    sns.heatmap(attn[head_idx], ax=ax, cmap='viridis', cbar=True)
                    ax.set_title(f'Head {head_idx}')
                    ax.set_xlabel('Key Position')
                    ax.set_ylabel('Query Position')

                plt.tight_layout()
                log_dict[f'attention/layer_{layer_idx}_heatmap'] = wandb.Image(fig)
                plt.close(fig)

            wandb.log(log_dict, step=step)

    def log_episodic_memory(self, memory_bank: torch.Tensor, memory_usage: Dict, step: int):
        """Log episodic memory statistics"""
        if step % self.log_freq != 0:
            return

        with torch.no_grad():
            log_dict = {'step': step}

            # Memory utilization
            if 'read_count' in memory_usage:
                log_dict['memory/read_frequency'] = memory_usage['read_count']
            if 'write_count' in memory_usage:
                log_dict['memory/write_frequency'] = memory_usage['write_count']

            # Memory slot utilization
            if memory_bank is not None:
                memory_norms = torch.norm(memory_bank, dim=-1).cpu().numpy()
                log_dict['memory/avg_slot_magnitude'] = np.mean(memory_norms)
                log_dict['memory/max_slot_magnitude'] = np.max(memory_norms)
                log_dict['memory/active_slots'] = np.sum(memory_norms > 0.1)
                log_dict['memory/slot_sparsity'] = np.sum(memory_norms < 0.01) / memory_norms.size

                # Memory distribution histogram
                if step % (self.log_freq * 5) == 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(memory_norms, bins=50, alpha=0.7, color='blue')
                    ax.set_xlabel('Memory Slot Magnitude')
                    ax.set_ylabel('Count')
                    ax.set_title('Episodic Memory Slot Distribution')
                    ax.grid(True, alpha=0.3)
                    log_dict['memory/slot_distribution'] = wandb.Image(fig)
                    plt.close(fig)

            # Memory access patterns
            if 'top_k_indices' in memory_usage:
                indices = memory_usage['top_k_indices']
                if len(indices) > 0:
                    unique_indices = len(set(indices))
                    log_dict['memory/unique_slots_accessed'] = unique_indices
                    log_dict['memory/access_diversity'] = unique_indices / len(indices)

            wandb.log(log_dict, step=step)

    def log_weight_distributions(self, model: nn.Module, step: int):
        """Log weight distributions and statistics"""
        if step % self.log_freq != 0:
            return

        log_dict = {'step': step}

        for name, param in model.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    weights = param.data.cpu().numpy().flatten()

                    # Basic statistics
                    log_dict[f'weights/{name}_mean'] = np.mean(weights)
                    log_dict[f'weights/{name}_std'] = np.std(weights)
                    log_dict[f'weights/{name}_min'] = np.min(weights)
                    log_dict[f'weights/{name}_max'] = np.max(weights)

                    # Sparsity (percentage of near-zero weights)
                    sparsity = np.sum(np.abs(weights) < 0.01) / weights.size
                    log_dict[f'weights/{name}_sparsity'] = sparsity

                    # Weight histograms (less frequent)
                    if step % (self.log_freq * 10) == 0:
                        log_dict[f'weights/{name}_histogram'] = wandb.Histogram(weights)

        wandb.log(log_dict, step=step)

    def log_gradient_flow(self, model: nn.Module, step: int):
        """Log gradient magnitudes and flow across layers"""
        log_dict = {'step': step}

        total_norm = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                with torch.no_grad():
                    grad = param.grad.data.cpu()
                    param_norm = grad.norm(2).item()
                    total_norm += param_norm ** 2

                    # Store layer-wise gradient norms
                    layer_name = name.split('.')[0]
                    if layer_name not in layer_norms:
                        layer_norms[layer_name] = 0.0
                    layer_norms[layer_name] += param_norm ** 2

                    # Log individual parameter gradients
                    log_dict[f'gradients/{name}_norm'] = param_norm
                    log_dict[f'gradients/{name}_mean'] = grad.mean().item()
                    log_dict[f'gradients/{name}_max'] = grad.max().item()

        total_norm = total_norm ** 0.5
        log_dict['gradients/total_norm'] = total_norm

        # Log layer-wise gradient norms
        for layer_name, norm_sq in layer_norms.items():
            log_dict[f'gradients/layer_{layer_name}_norm'] = norm_sq ** 0.5

        wandb.log(log_dict, step=step)

    def log_activations(self, activations: Dict[str, torch.Tensor], step: int):
        """Log activation statistics"""
        if step % self.log_freq != 0:
            return

        log_dict = {'step': step}

        for name, activation in activations.items():
            with torch.no_grad():
                act = activation.cpu().numpy().flatten()

                log_dict[f'activations/{name}_mean'] = np.mean(act)
                log_dict[f'activations/{name}_std'] = np.std(act)
                log_dict[f'activations/{name}_min'] = np.min(act)
                log_dict[f'activations/{name}_max'] = np.max(act)

                # Dead neurons (activations near zero)
                dead_neurons = np.sum(np.abs(act) < 1e-6) / act.size
                log_dict[f'activations/{name}_dead_neurons'] = dead_neurons

        wandb.log(log_dict, step=step)

    def log_bitnet_quantization(self, layer_name: str, weights: torch.Tensor,
                                quantized_weights: torch.Tensor, step: int):
        """Log BitNet quantization metrics"""
        if step % self.log_freq != 0:
            return

        with torch.no_grad():
            # Quantization error
            quant_error = (weights - quantized_weights).abs().mean().item()

            # Weight distribution after quantization
            q_weights = quantized_weights.cpu().numpy().flatten()
            unique, counts = np.unique(q_weights, return_counts=True)

            log_dict = {
                f'quantization/{layer_name}_error': quant_error,
                f'quantization/{layer_name}_num_values': len(unique),
                f'quantization/{layer_name}_zeros_pct': np.sum(q_weights == 0) / q_weights.size,
                f'quantization/{layer_name}_positive_pct': np.sum(q_weights > 0) / q_weights.size,
                f'quantization/{layer_name}_negative_pct': np.sum(q_weights < 0) / q_weights.size,
                'step': step
            }

            # Quantization distribution plot
            if step % (self.log_freq * 5) == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Original weights
                ax1.hist(weights.cpu().numpy().flatten(), bins=50, alpha=0.7, color='blue')
                ax1.set_title('Original Weights')
                ax1.set_xlabel('Weight Value')
                ax1.set_ylabel('Count')
                ax1.grid(True, alpha=0.3)

                # Quantized weights
                ax2.bar(unique, counts, alpha=0.7, color='red')
                ax2.set_title('Quantized Weights (1.58-bit)')
                ax2.set_xlabel('Quantized Value')
                ax2.set_ylabel('Count')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                log_dict[f'quantization/{layer_name}_distribution'] = wandb.Image(fig)
                plt.close(fig)

            wandb.log(log_dict, step=step)

    def log_cross_modal_fusion(self, text_features: torch.Tensor,
                               image_features: torch.Tensor,
                               similarity_matrix: Optional[torch.Tensor],
                               step: int):
        """Log cross-modal fusion metrics (FIBER-style)"""
        if step % self.log_freq != 0:
            return

        with torch.no_grad():
            log_dict = {'step': step}

            # Feature statistics
            log_dict['fusion/text_feature_mean'] = text_features.mean().item()
            log_dict['fusion/text_feature_std'] = text_features.std().item()
            log_dict['fusion/image_feature_mean'] = image_features.mean().item()
            log_dict['fusion/image_feature_std'] = image_features.std().item()

            # Similarity matrix statistics
            if similarity_matrix is not None:
                sim = similarity_matrix.cpu().numpy()
                log_dict['fusion/similarity_mean'] = np.mean(sim)
                log_dict['fusion/similarity_max'] = np.max(sim)
                log_dict['fusion/similarity_diagonal'] = np.mean(np.diag(sim))

                # Similarity matrix heatmap
                if step % (self.log_freq * 5) == 0:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(sim, ax=ax, cmap='coolwarm', center=0,
                               xticklabels=False, yticklabels=False)
                    ax.set_title('Text-Image Similarity Matrix')
                    ax.set_xlabel('Image Features')
                    ax.set_ylabel('Text Features')
                    log_dict['fusion/similarity_heatmap'] = wandb.Image(fig)
                    plt.close(fig)

            # Alignment score
            if text_features.shape[0] == image_features.shape[0]:
                alignment = F.cosine_similarity(text_features, image_features).mean().item()
                log_dict['fusion/alignment_score'] = alignment

            wandb.log(log_dict, step=step)

    def log_embeddings_visualization(self, embeddings: torch.Tensor,
                                     labels: List[str], step: int, name: str = "embeddings"):
        """Create t-SNE visualization of embeddings"""
        if step % (self.log_freq * 10) != 0:
            return

        with torch.no_grad():
            # Subsample if too many points
            max_points = 1000
            if embeddings.shape[0] > max_points:
                indices = np.random.choice(embeddings.shape[0], max_points, replace=False)
                embeddings = embeddings[indices]
                labels = [labels[i] for i in indices]

            # Compute t-SNE
            emb_np = embeddings.cpu().numpy()
            if emb_np.shape[0] > 2:
                tsne = TSNE(n_components=2, random_state=42)
                emb_2d = tsne.fit_transform(emb_np)

                # Create scatter plot
                fig, ax = plt.subplots(figsize=(12, 10))
                scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                                    c=range(len(emb_2d)), cmap='viridis', alpha=0.6)
                ax.set_title(f'{name} t-SNE Visualization')
                ax.set_xlabel('t-SNE Dimension 1')
                ax.set_ylabel('t-SNE Dimension 2')
                plt.colorbar(scatter, ax=ax)

                wandb.log({f'embeddings/{name}_tsne': wandb.Image(fig), 'step': step})
                plt.close(fig)

    def log_codecarbon(self, emissions_data: Dict, step: int):
        """Log CodeCarbon energy consumption metrics"""
        log_dict = {
            'carbon/energy_consumed_kwh': emissions_data.get('energy_consumed', 0),
            'carbon/co2_emissions_kg': emissions_data.get('emissions', 0),
            'carbon/power_watts': emissions_data.get('power', 0),
            'carbon/cpu_energy_kwh': emissions_data.get('cpu_energy', 0),
            'carbon/gpu_energy_kwh': emissions_data.get('gpu_energy', 0),
            'carbon/ram_energy_kwh': emissions_data.get('ram_energy', 0),
            'step': step
        }
        wandb.log(log_dict, step=step)

    def log_generation_quality(self, generated_texts: List[str],
                              reference_texts: Optional[List[str]],
                              step: int):
        """Log text generation quality metrics"""
        log_dict = {'step': step}

        # Text diversity
        all_tokens = []
        for text in generated_texts:
            all_tokens.extend(text.split())

        if len(all_tokens) > 0:
            unique_tokens = len(set(all_tokens))
            log_dict['generation/unique_tokens'] = unique_tokens
            log_dict['generation/diversity_ratio'] = unique_tokens / len(all_tokens)

        # Average length
        avg_length = np.mean([len(text.split()) for text in generated_texts])
        log_dict['generation/avg_length'] = avg_length

        # Log sample generations
        if step % (self.log_freq * 5) == 0:
            table = wandb.Table(columns=["Generated Text"])
            for text in generated_texts[:5]:  # Log first 5 samples
                table.add_data(text)
            log_dict['generation/samples'] = table

        wandb.log(log_dict, step=step)

    def log_data_pipeline(self, loading_time: float, preprocessing_time: float, step: int):
        """Log data loading and preprocessing metrics"""
        log_dict = {
            'data/loading_time_ms': loading_time * 1000,
            'data/preprocessing_time_ms': preprocessing_time * 1000,
            'data/total_time_ms': (loading_time + preprocessing_time) * 1000,
            'step': step
        }
        wandb.log(log_dict, step=step)

