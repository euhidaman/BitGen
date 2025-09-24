"""
Advanced BitGen Metrics: Episodic Memory, Attention Heatmaps, and Reasoning Matrices
Comprehensive monitoring of internal model states and decision processes
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wandb
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

class EpisodicMemoryAnalyzer:
    """Analyze episodic memory performance and visualization"""

    def __init__(self, memory_size: int, memory_dim: int):
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Memory tracking
        self.memory_usage_history = []
        self.memory_retrieval_patterns = defaultdict(list)
        self.memory_update_frequency = np.zeros(memory_size)
        self.memory_similarity_matrix = np.zeros((memory_size, memory_size))

    def analyze_memory_state(self, memory_keys, memory_values, attention_weights, step: int):
        """Analyze current episodic memory state"""

        # Memory utilization analysis
        memory_norms = torch.norm(memory_values, dim=-1).cpu().numpy()
        active_memories = (memory_norms > 0.1).sum()
        memory_utilization = active_memories / self.memory_size

        # Attention pattern analysis
        attention_entropy = self._calculate_attention_entropy(attention_weights)
        most_accessed_memories = torch.argmax(attention_weights, dim=-1).cpu().numpy()

        # Memory diversity (how different are the stored memories)
        memory_diversity = self._calculate_memory_diversity(memory_values)

        # Update tracking
        self.memory_usage_history.append({
            'step': step,
            'utilization': memory_utilization,
            'active_memories': active_memories,
            'attention_entropy': attention_entropy,
            'memory_diversity': memory_diversity,
            'most_accessed': most_accessed_memories.tolist()
        })

        # Update memory access frequency
        for mem_idx in most_accessed_memories:
            self.memory_update_frequency[mem_idx] += 1

        # Update similarity matrix
        self._update_memory_similarity_matrix(memory_values)

        return {
            'memory_utilization': memory_utilization,
            'active_memories': active_memories,
            'attention_entropy': attention_entropy,
            'memory_diversity': memory_diversity,
            'memory_access_frequency': self.memory_update_frequency.copy()
        }

    def _calculate_attention_entropy(self, attention_weights):
        """Calculate entropy of attention distribution"""
        # attention_weights: [batch_size, seq_len, memory_size]
        avg_attention = attention_weights.mean(dim=(0, 1))  # Average over batch and sequence
        entropy = -torch.sum(avg_attention * torch.log(avg_attention + 1e-8))
        return entropy.item()

    def _calculate_memory_diversity(self, memory_values):
        """Calculate diversity of stored memories"""
        # Compute pairwise cosine similarities
        normalized_memories = F.normalize(memory_values, dim=-1)
        similarities = torch.mm(normalized_memories, normalized_memories.T)

        # Calculate average similarity (lower = more diverse)
        mask = torch.triu(torch.ones_like(similarities), diagonal=1).bool()
        avg_similarity = similarities[mask].mean().item()

        # Diversity = 1 - similarity
        return 1.0 - avg_similarity

    def _update_memory_similarity_matrix(self, memory_values):
        """Update memory similarity matrix"""
        normalized_memories = F.normalize(memory_values, dim=-1)
        similarities = torch.mm(normalized_memories, normalized_memories.T)
        self.memory_similarity_matrix = similarities.cpu().numpy()

    def create_memory_heatmaps(self, save_path: str = None):
        """Create comprehensive memory heatmaps"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Episodic Memory Analysis', fontsize=16, fontweight='bold')

        # Memory access frequency heatmap
        access_freq_2d = self.memory_update_frequency.reshape(-1, int(np.sqrt(self.memory_size)))
        sns.heatmap(access_freq_2d, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 0])
        axes[0, 0].set_title('Memory Access Frequency')
        axes[0, 0].set_xlabel('Memory Index (reshaped)')
        axes[0, 0].set_ylabel('Memory Index (reshaped)')

        # Memory similarity matrix
        sns.heatmap(self.memory_similarity_matrix, cmap='RdBu_r', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Memory Similarity Matrix')
        axes[0, 1].set_xlabel('Memory Slot')
        axes[0, 1].set_ylabel('Memory Slot')

        # Memory utilization over time
        if self.memory_usage_history:
            steps = [h['step'] for h in self.memory_usage_history]
            utilization = [h['utilization'] for h in self.memory_usage_history]
            diversity = [h['memory_diversity'] for h in self.memory_usage_history]

            axes[1, 0].plot(steps, utilization, 'b-', label='Utilization', linewidth=2)
            axes[1, 0].plot(steps, diversity, 'r-', label='Diversity', linewidth=2)
            axes[1, 0].set_title('Memory Metrics Over Time')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Metric Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Attention entropy over time
        if self.memory_usage_history:
            entropy_values = [h['attention_entropy'] for h in self.memory_usage_history]
            axes[1, 1].plot(steps, entropy_values, 'g-', linewidth=2)
            axes[1, 1].set_title('Attention Entropy Over Time')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig

class AttentionHeatmapAnalyzer:
    """Analyze and visualize attention patterns with focus on important tokens"""

    def __init__(self, num_heads: int, max_seq_len: int):
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Attention tracking
        self.attention_history = []
        self.important_token_patterns = defaultdict(list)
        self.head_specialization = np.zeros((num_heads, 4))  # Track what each head focuses on

    def analyze_attention_patterns(self, attention_weights, input_tokens, output_tokens, step: int):
        """Analyze attention patterns and identify important tokens"""

        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Calculate attention statistics
        attention_stats = {}

        # 1. Attention concentration (how focused is attention)
        attention_entropy = self._calculate_attention_entropy_per_head(attention_weights)
        attention_stats['attention_entropy'] = attention_entropy

        # 2. Important tokens identification
        important_tokens = self._identify_important_tokens(attention_weights, input_tokens)
        attention_stats['important_tokens'] = important_tokens

        # 3. Head specialization analysis
        head_roles = self._analyze_head_specialization(attention_weights, step)
        attention_stats['head_specialization'] = head_roles

        # 4. Attention sink analysis
        sink_attention = self._analyze_attention_sinks(attention_weights)
        attention_stats['attention_sinks'] = sink_attention

        # Store for historical analysis
        self.attention_history.append({
            'step': step,
            'attention_weights': attention_weights.cpu().numpy(),
            'input_tokens': input_tokens.cpu().numpy() if torch.is_tensor(input_tokens) else input_tokens,
            'stats': attention_stats
        })

        return attention_stats

    def _calculate_attention_entropy_per_head(self, attention_weights):
        """Calculate attention entropy for each head"""
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        entropies = []

        for head in range(self.num_heads):
            head_attention = attention_weights[:, head, :, :]  # [batch_size, seq_len, seq_len]

            # Calculate entropy for each position
            head_entropy = -torch.sum(head_attention * torch.log(head_attention + 1e-8), dim=-1)
            avg_entropy = head_entropy.mean().item()
            entropies.append(avg_entropy)

        return entropies

    def _identify_important_tokens(self, attention_weights, input_tokens):
        """Identify tokens that receive the most attention"""
        # Sum attention across all heads and positions
        total_attention = attention_weights.sum(dim=(1, 2))  # [batch_size, seq_len]
        avg_attention = total_attention.mean(dim=0)  # [seq_len]

        # Get top attended positions
        top_positions = torch.topk(avg_attention, k=min(5, len(avg_attention)))[1]

        important_info = []
        for pos in top_positions:
            attention_score = avg_attention[pos].item()
            token_id = input_tokens[0][pos].item() if torch.is_tensor(input_tokens) else input_tokens[0][pos]

            important_info.append({
                'position': pos.item(),
                'token_id': token_id,
                'attention_score': attention_score
            })

        return important_info

    def _analyze_head_specialization(self, attention_weights, step):
        """Analyze what each attention head specializes in"""
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]

        head_roles = []

        for head in range(self.num_heads):
            head_attention = attention_weights[:, head, :, :].mean(dim=0)  # [seq_len, seq_len]

            # Analyze attention patterns
            # 1. Local vs Global attention
            local_attention = torch.triu(head_attention, diagonal=-2).sum() - torch.triu(head_attention, diagonal=3).sum()
            global_attention = head_attention.sum() - local_attention

            # 2. Beginning vs End focus
            begin_focus = head_attention[:, :len(head_attention)//3].sum()
            end_focus = head_attention[:, 2*len(head_attention)//3:].sum()

            # 3. Self vs Cross attention
            self_attention = torch.diag(head_attention).sum()
            cross_attention = head_attention.sum() - self_attention

            # 4. Attention spreading vs concentration
            attention_std = head_attention.std().item()

            role_vector = np.array([
                local_attention.item() / (local_attention.item() + global_attention.item() + 1e-8),
                begin_focus.item() / (begin_focus.item() + end_focus.item() + 1e-8),
                self_attention.item() / (self_attention.item() + cross_attention.item() + 1e-8),
                attention_std
            ])

            # Update head specialization tracking
            self.head_specialization[head] = 0.9 * self.head_specialization[head] + 0.1 * role_vector

            head_roles.append({
                'head_id': head,
                'local_focus': role_vector[0],
                'begin_focus': role_vector[1],
                'self_focus': role_vector[2],
                'concentration': role_vector[3],
                'specialization_vector': self.head_specialization[head].tolist()
            })

        return head_roles

    def _analyze_attention_sinks(self, attention_weights):
        """Analyze attention sink behavior"""
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]

        # Sum attention received by each position across all queries
        attention_received = attention_weights.sum(dim=(0, 1, 2))  # [seq_len]

        # Identify attention sinks (positions that receive disproportionate attention)
        attention_threshold = attention_received.mean() + 2 * attention_received.std()
        sink_positions = (attention_received > attention_threshold).nonzero().squeeze()

        sink_info = []
        for pos in sink_positions:
            sink_info.append({
                'position': pos.item(),
                'attention_received': attention_received[pos].item(),
                'relative_importance': attention_received[pos].item() / attention_received.sum().item()
            })

        return sink_info

    def create_attention_heatmaps(self, attention_weights, input_tokens, tokenizer=None, save_path: str = None):
        """Create comprehensive attention heatmaps"""

        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Take first sample for visualization
        sample_attention = attention_weights[0].cpu().numpy()  # [num_heads, seq_len, seq_len]
        sample_tokens = input_tokens[0].cpu().numpy() if torch.is_tensor(input_tokens) else input_tokens[0]

        # Create token labels if tokenizer available
        if tokenizer:
            token_labels = [tokenizer.decode([token_id]) for token_id in sample_tokens[:seq_len]]
        else:
            token_labels = [f"T{i}" for i in range(seq_len)]

        # Create subplots for multiple heads
        n_cols = min(4, num_heads)
        n_rows = (num_heads + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        fig.suptitle('Attention Heatmaps - Focus on Important Tokens', fontsize=16, fontweight='bold')

        if num_heads == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for head in range(num_heads):
            if head < len(axes):
                ax = axes[head]

                # Create heatmap for this head
                sns.heatmap(
                    sample_attention[head],
                    xticklabels=token_labels,
                    yticklabels=token_labels,
                    cmap='Reds',
                    ax=ax,
                    cbar_kws={'label': 'Attention Weight'}
                )

                ax.set_title(f'Head {head} - Attention Pattern')
                ax.set_xlabel('Key Tokens')
                ax.set_ylabel('Query Tokens')

                # Rotate labels for readability
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)

        # Hide unused subplots
        for head in range(num_heads, len(axes)):
            axes[head].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig

    def create_interactive_attention_heatmap(self, attention_weights, input_tokens, tokenizer=None):
        """Create interactive attention heatmap using Plotly"""

        # Take first sample and average across heads for overview
        sample_attention = attention_weights[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
        sample_tokens = input_tokens[0].cpu().numpy() if torch.is_tensor(input_tokens) else input_tokens[0]

        # Create token labels
        if tokenizer:
            token_labels = [tokenizer.decode([token_id]) for token_id in sample_tokens]
        else:
            token_labels = [f"Token_{i}" for i in range(len(sample_tokens))]

        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sample_attention,
            x=token_labels,
            y=token_labels,
            colorscale='Reds',
            hoverongaps=False,
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title='Interactive Attention Heatmap - Average Across Heads',
            xaxis_title='Key Tokens (Attended To)',
            yaxis_title='Query Tokens (Attending)',
            width=800,
            height=800
        )

        return fig

class ReasoningMatrixAnalyzer:
    """Analyze reasoning patterns and robot selection matrices"""

    def __init__(self, num_robots: int, robot_types: List[str] = None):
        self.num_robots = num_robots
        self.robot_types = robot_types or [f"Robot_{i}" for i in range(num_robots)]

        # Reasoning tracking
        self.confusion_matrices = []  # Store confusion matrix for each epoch
        self.reasoning_accuracy_history = []
        self.robot_selection_patterns = defaultdict(list)
        self.task_robot_associations = defaultdict(Counter)

    def analyze_reasoning_step(self,
                             robot_predictions,
                             robot_targets,
                             task_descriptions,
                             reasoning_outputs,
                             epoch: int):
        """Analyze reasoning step and robot selection accuracy"""

        # Convert to numpy for analysis
        predictions = robot_predictions.cpu().numpy()
        targets = robot_targets.cpu().numpy()

        # Calculate confusion matrix for this batch
        batch_confusion = self._calculate_confusion_matrix(predictions, targets)

        # Analyze reasoning patterns
        reasoning_stats = {
            'epoch': epoch,
            'batch_accuracy': (predictions == targets).mean(),
            'confusion_matrix': batch_confusion,
            'robot_distribution': np.bincount(predictions, minlength=self.num_robots),
            'target_distribution': np.bincount(targets, minlength=self.num_robots)
        }

        # Update task-robot associations
        for i, (pred, target, task) in enumerate(zip(predictions, targets, task_descriptions)):
            self.task_robot_associations[task].update([self.robot_types[pred]])

        # Store reasoning accuracy
        self.reasoning_accuracy_history.append({
            'epoch': epoch,
            'accuracy': reasoning_stats['batch_accuracy'],
            'step': len(self.reasoning_accuracy_history)
        })

        return reasoning_stats

    def update_epoch_confusion_matrix(self, epoch_predictions, epoch_targets, epoch: int):
        """Update confusion matrix for entire epoch"""

        # Calculate epoch confusion matrix
        epoch_confusion = self._calculate_confusion_matrix(epoch_predictions, epoch_targets)

        # Store with metadata
        confusion_data = {
            'epoch': epoch,
            'confusion_matrix': epoch_confusion,
            'accuracy': (epoch_predictions == epoch_targets).mean(),
            'per_class_accuracy': self._calculate_per_class_accuracy(epoch_confusion),
            'timestamp': datetime.now().isoformat()
        }

        self.confusion_matrices.append(confusion_data)

        return confusion_data

    def _calculate_confusion_matrix(self, predictions, targets):
        """Calculate confusion matrix"""
        confusion = np.zeros((self.num_robots, self.num_robots))

        for pred, target in zip(predictions.flatten(), targets.flatten()):
            if 0 <= pred < self.num_robots and 0 <= target < self.num_robots:
                confusion[target, pred] += 1

        return confusion

    def _calculate_per_class_accuracy(self, confusion_matrix):
        """Calculate per-class accuracy from confusion matrix"""
        diagonal = np.diag(confusion_matrix)
        row_sums = confusion_matrix.sum(axis=1)

        # Avoid division by zero
        accuracies = np.divide(diagonal, row_sums, out=np.zeros_like(diagonal), where=row_sums!=0)

        return accuracies.tolist()

    def create_confusion_matrix_evolution(self, save_path: str = None):
        """Create visualization showing confusion matrix evolution across epochs"""

        if not self.confusion_matrices:
            return None

        n_epochs = len(self.confusion_matrices)
        n_cols = min(4, n_epochs)
        n_rows = (n_epochs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('Robot Selection Confusion Matrix Evolution', fontsize=16, fontweight='bold')

        if n_epochs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_epochs == 1 else axes
        else:
            axes = axes.flatten()

        for i, cm_data in enumerate(self.confusion_matrices):
            if i < len(axes):
                ax = axes[i]

                # Normalize confusion matrix for better visualization
                cm = cm_data['confusion_matrix']
                cm_normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

                # Create heatmap
                sns.heatmap(
                    cm_normalized,
                    xticklabels=self.robot_types,
                    yticklabels=self.robot_types,
                    cmap='Blues',
                    ax=ax,
                    annot=True,
                    fmt='.2f',
                    cbar_kws={'label': 'Normalized Frequency'}
                )

                accuracy = cm_data['accuracy']
                ax.set_title(f'Epoch {cm_data["epoch"]} (Acc: {accuracy:.3f})')
                ax.set_xlabel('Predicted Robot')
                ax.set_ylabel('Actual Robot')

        # Hide unused subplots
        for i in range(n_epochs, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig

    def create_reasoning_improvement_chart(self, save_path: str = None):
        """Create chart showing reasoning improvement over training"""

        if not self.reasoning_accuracy_history:
            return None

        # Extract data
        epochs = [h['epoch'] for h in self.reasoning_accuracy_history]
        accuracies = [h['accuracy'] for h in self.reasoning_accuracy_history]

        # Create improvement chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy over time
        ax1.plot(epochs, accuracies, 'b-', linewidth=2, marker='o')
        ax1.set_title('Robot Selection Accuracy Over Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True)
        ax1.set_ylim(0, 1)

        # Add trend line
        z = np.polyfit(epochs, accuracies, 1)
        p = np.poly1d(z)
        ax1.plot(epochs, p(epochs), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
        ax1.legend()

        # Accuracy distribution per robot type
        if self.confusion_matrices:
            latest_cm = self.confusion_matrices[-1]
            per_class_acc = latest_cm['per_class_accuracy']

            bars = ax2.bar(self.robot_types, per_class_acc, color='skyblue', edgecolor='navy')
            ax2.set_title('Per-Robot Selection Accuracy (Latest)')
            ax2.set_xlabel('Robot Type')
            ax2.set_ylabel('Accuracy')
            ax2.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, acc in zip(bars, per_class_acc):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig

    def create_interactive_reasoning_dashboard(self):
        """Create interactive reasoning analysis dashboard"""

        if not self.confusion_matrices:
            return None

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Evolution', 'Latest Confusion Matrix',
                          'Robot Selection Distribution', 'Reasoning Improvement Rate'),
            specs=[[{"secondary_y": False}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"secondary_y": False}]]
        )

        # 1. Accuracy evolution
        if self.reasoning_accuracy_history:
            epochs = [h['epoch'] for h in self.reasoning_accuracy_history]
            accuracies = [h['accuracy'] for h in self.reasoning_accuracy_history]

            fig.add_trace(
                go.Scatter(x=epochs, y=accuracies, mode='lines+markers',
                          name='Accuracy', line=dict(width=3)),
                row=1, col=1
            )

        # 2. Latest confusion matrix
        if self.confusion_matrices:
            latest_cm = self.confusion_matrices[-1]['confusion_matrix']
            cm_normalized = latest_cm / (latest_cm.sum(axis=1, keepdims=True) + 1e-8)

            fig.add_trace(
                go.Heatmap(z=cm_normalized,
                          x=self.robot_types,
                          y=self.robot_types,
                          colorscale='Blues',
                          showscale=True),
                row=1, col=2
            )

        # 3. Robot selection distribution
        if self.confusion_matrices:
            latest_pred_dist = self.confusion_matrices[-1]['confusion_matrix'].sum(axis=0)

            fig.add_trace(
                go.Bar(x=self.robot_types, y=latest_pred_dist, name='Predictions'),
                row=2, col=1
            )

        # 4. Improvement rate
        if len(self.reasoning_accuracy_history) > 5:
            # Calculate moving average
            window_size = min(5, len(self.reasoning_accuracy_history))
            moving_avg = []
            for i in range(len(self.reasoning_accuracy_history)):
                start_idx = max(0, i - window_size + 1)
                window_data = self.reasoning_accuracy_history[start_idx:i+1]
                avg_acc = sum(h['accuracy'] for h in window_data) / len(window_data)
                moving_avg.append(avg_acc)

            fig.add_trace(
                go.Scatter(x=epochs, y=moving_avg, mode='lines',
                          name='Moving Average', line=dict(width=2, dash='dash')),
                row=2, col=2
            )

        fig.update_layout(height=800, title_text="Reasoning and Robot Selection Analysis Dashboard")

        return fig

class AdvancedMetricsLogger:
    """Integrate advanced metrics with WandB logging"""

    def __init__(self, wandb_integration, output_dir: str):
        self.wandb_integration = wandb_integration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize analyzers
        self.memory_analyzer = None
        self.attention_analyzer = None
        self.reasoning_analyzer = None

    def initialize_analyzers(self, config):
        """Initialize all analyzers based on model configuration"""

        self.memory_analyzer = EpisodicMemoryAnalyzer(
            memory_size=config.memory_size,
            memory_dim=config.memory_dim
        )

        self.attention_analyzer = AttentionHeatmapAnalyzer(
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len
        )

        # Get robot types from config or use defaults
        robot_types = getattr(config, 'robot_types', None)
        if robot_types is None:
            robot_types = ['manipulator', 'mobile_base', 'quadruped', 'humanoid',
                          'aerial_drone', 'ground_vehicle', 'gripper_robot', 'inspection_robot']

        self.reasoning_analyzer = ReasoningMatrixAnalyzer(
            num_robots=config.num_robots,
            robot_types=robot_types[:config.num_robots]
        )

    def log_step_metrics(self, model_outputs, batch_data, step: int):
        """Log metrics for a single training step"""

        if not all([self.memory_analyzer, self.attention_analyzer, self.reasoning_analyzer]):
            return

        metrics = {}

        # 1. Episodic Memory Analysis
        if hasattr(model_outputs, 'memory_keys') and hasattr(model_outputs, 'memory_values'):
            memory_stats = self.memory_analyzer.analyze_memory_state(
                memory_keys=model_outputs.memory_keys,
                memory_values=model_outputs.memory_values,
                attention_weights=model_outputs.get('memory_attention', None),
                step=step
            )

            metrics.update({
                'episodic_memory/utilization': memory_stats['memory_utilization'],
                'episodic_memory/active_memories': memory_stats['active_memories'],
                'episodic_memory/attention_entropy': memory_stats['attention_entropy'],
                'episodic_memory/diversity': memory_stats['memory_diversity']
            })

        # 2. Attention Pattern Analysis
        if hasattr(model_outputs, 'attention_weights'):
            attention_stats = self.attention_analyzer.analyze_attention_patterns(
                attention_weights=model_outputs.attention_weights,
                input_tokens=batch_data['input_ids'],
                output_tokens=model_outputs.get('output_tokens', None),
                step=step
            )

            # Log attention entropy per head
            for head_idx, entropy in enumerate(attention_stats['attention_entropy']):
                metrics[f'attention/head_{head_idx}_entropy'] = entropy

            # Log important tokens
            if attention_stats['important_tokens']:
                avg_importance = sum(t['attention_score'] for t in attention_stats['important_tokens']) / len(attention_stats['important_tokens'])
                metrics['attention/avg_token_importance'] = avg_importance

            # Log attention sink information
            if attention_stats['attention_sinks']:
                metrics['attention/num_attention_sinks'] = len(attention_stats['attention_sinks'])

        # 3. Robot Selection Analysis (if available)
        if 'robot_selection' in model_outputs and model_outputs['robot_selection'] is not None:
            robot_probs = model_outputs['robot_selection']
            if 'target_robot' in batch_data:
                targets = batch_data['target_robot']

                # Get predictions
                predictions = robot_probs.argmax(dim=-1)

                reasoning_stats = self.reasoning_analyzer.analyze_reasoning_step(
                    robot_predictions=predictions,
                    robot_targets=targets,
                    task_descriptions=batch_data.get('task_description', [''] * len(targets)),
                    reasoning_outputs=model_outputs.get('reasoning_outputs', None),
                    epoch=step // 100  # Approximate epoch from step
                )

                metrics.update({
                    'reasoning/batch_accuracy': reasoning_stats['batch_accuracy'],
                    'reasoning/prediction_confidence': robot_probs.max(dim=-1)[0].mean().item(),
                    'reasoning/prediction_entropy': -torch.sum(robot_probs * torch.log(robot_probs + 1e-8), dim=-1).mean().item()
                })

        # Log to WandB
        if metrics:
            self.wandb_integration.log_training_metrics(metrics, step=step)

    def log_epoch_analysis(self, epoch: int, model, epoch_data):
        """Log comprehensive epoch analysis with visualizations"""

        # 1. Create and log episodic memory heatmaps
        if self.memory_analyzer and self.memory_analyzer.memory_usage_history:
            memory_heatmap_path = self.output_dir / f"memory_heatmaps_epoch_{epoch}.png"
            memory_fig = self.memory_analyzer.create_memory_heatmaps(str(memory_heatmap_path))

            # Log to WandB
            self.wandb_integration.run.log({
                f"episodic_memory/heatmaps_epoch_{epoch}": wandb.Image(str(memory_heatmap_path))
            })

        # 2. Create and log attention heatmaps
        if self.attention_analyzer and self.attention_analyzer.attention_history:
            # Get latest attention data
            latest_attention = self.attention_analyzer.attention_history[-1]

            # Create static heatmap
            attention_heatmap_path = self.output_dir / f"attention_heatmaps_epoch_{epoch}.png"
            attention_fig = self.attention_analyzer.create_attention_heatmaps(
                attention_weights=torch.tensor(latest_attention['attention_weights']),
                input_tokens=torch.tensor(latest_attention['input_tokens']),
                save_path=str(attention_heatmap_path)
            )

            # Create interactive heatmap
            interactive_attention = self.attention_analyzer.create_interactive_attention_heatmap(
                attention_weights=torch.tensor(latest_attention['attention_weights']),
                input_tokens=torch.tensor(latest_attention['input_tokens'])
            )

            # Log to WandB
            self.wandb_integration.run.log({
                f"attention/heatmaps_epoch_{epoch}": wandb.Image(str(attention_heatmap_path)),
                f"attention/interactive_epoch_{epoch}": wandb.Plotly(interactive_attention)
            })

        # 3. Create and log reasoning matrices
        if self.reasoning_analyzer and self.reasoning_analyzer.confusion_matrices:
            # Create confusion matrix evolution
            confusion_evolution_path = self.output_dir / f"reasoning_matrix_epoch_{epoch}.png"
            confusion_fig = self.reasoning_analyzer.create_confusion_matrix_evolution(str(confusion_evolution_path))

            # Create reasoning improvement chart
            improvement_chart_path = self.output_dir / f"reasoning_improvement_epoch_{epoch}.png"
            improvement_fig = self.reasoning_analyzer.create_reasoning_improvement_chart(str(improvement_chart_path))

            # Create interactive reasoning dashboard
            reasoning_dashboard = self.reasoning_analyzer.create_interactive_reasoning_dashboard()

            # Log to WandB
            wandb_logs = {
                f"reasoning/confusion_matrix_epoch_{epoch}": wandb.Image(str(confusion_evolution_path)),
                f"reasoning/improvement_chart_epoch_{epoch}": wandb.Image(str(improvement_chart_path))
            }

            if reasoning_dashboard:
                wandb_logs[f"reasoning/dashboard_epoch_{epoch}"] = wandb.Plotly(reasoning_dashboard)

            self.wandb_integration.run.log(wandb_logs)

        # 4. Log comprehensive epoch metrics
        epoch_metrics = {
            'epoch_analysis/epoch': epoch,
            'epoch_analysis/timestamp': datetime.now().timestamp()
        }

        # Add memory metrics
        if self.memory_analyzer and self.memory_analyzer.memory_usage_history:
            latest_memory = self.memory_analyzer.memory_usage_history[-1]
            epoch_metrics.update({
                'epoch_analysis/memory_utilization': latest_memory['utilization'],
                'epoch_analysis/memory_diversity': latest_memory['memory_diversity'],
                'epoch_analysis/memory_attention_entropy': latest_memory['attention_entropy']
            })

        # Add reasoning metrics
        if self.reasoning_analyzer and self.reasoning_analyzer.confusion_matrices:
            latest_reasoning = self.reasoning_analyzer.confusion_matrices[-1]
            epoch_metrics.update({
                'epoch_analysis/reasoning_accuracy': latest_reasoning['accuracy'],
                'epoch_analysis/reasoning_improvement': self._calculate_improvement_rate()
            })

        self.wandb_integration.run.log(epoch_metrics)

    def _calculate_improvement_rate(self):
        """Calculate reasoning improvement rate"""
        if len(self.reasoning_analyzer.reasoning_accuracy_history) < 2:
            return 0.0

        # Calculate improvement over last few epochs
        recent_accuracies = [h['accuracy'] for h in self.reasoning_analyzer.reasoning_accuracy_history[-5:]]

        if len(recent_accuracies) >= 2:
            improvement = recent_accuracies[-1] - recent_accuracies[0]
            return improvement / len(recent_accuracies)

        return 0.0

def create_advanced_metrics_logger(wandb_integration, config, output_dir: str = "advanced_metrics"):
    """Factory function to create advanced metrics logger"""

    logger = AdvancedMetricsLogger(wandb_integration, output_dir)
    logger.initialize_analyzers(config)

    return logger
