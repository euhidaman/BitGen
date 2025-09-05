"""
Advanced Attention Visualization for BitGen Model
Inspired by BertViz and KDnuggets attention visualization techniques
Supports multi-modal attention patterns, episodic memory attention, and cross-modal fusion
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wandb

try:
    from bertviz import head_view, model_view, neuron_view
    from bertviz.util import format_attention
    BERTVIZ_AVAILABLE = True
except ImportError:
    BERTVIZ_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AttentionVisualizationConfig:
    """Configuration for attention visualization"""
    save_dir: str = "./attention_visualizations"
    max_heads_per_plot: int = 12
    use_wandb: bool = True
    save_interactive: bool = True
    save_static: bool = True
    attention_threshold: float = 0.1
    memory_attention_threshold: float = 0.05
    color_scheme: str = "viridis"
    figsize: Tuple[int, int] = (15, 10)
    dpi: int = 300

class AdvancedAttentionVisualizer:
    """
    Advanced attention visualization system for BitGen model
    Supports text, vision, cross-modal, and episodic memory attention patterns
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: AttentionVisualizationConfig = None,
        wandb_logger = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AttentionVisualizationConfig()
        self.wandb_logger = wandb_logger
        
        # Create save directory
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Attention pattern storage
        self.attention_patterns = {
            'text_encoder': [],
            'text_decoder': [],
            'cross_modal': [],
            'episodic_memory': [],
            'fiber_fusion': []
        }
        
        # Memory access patterns for analysis
        self.memory_access_history = []
        self.cross_modal_alignments = []
        
        logger.info(f"🎯 Advanced Attention Visualizer initialized")
        logger.info(f"  • Save directory: {self.save_dir}")
        logger.info(f"  • BertViz available: {BERTVIZ_AVAILABLE}")
        logger.info(f"  • Wandb integration: {self.config.use_wandb and self.wandb_logger is not None}")

    def extract_attention_patterns(
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        extract_memory: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all attention patterns from the model during forward pass
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass with attention extraction
            outputs = self.model(
                input_ids=input_ids,
                vision_features=vision_features,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            attention_patterns = {}
            
            # Extract text encoder attention patterns
            if hasattr(outputs, 'text_attentions') and outputs.text_attentions is not None:
                attention_patterns['text_encoder'] = [
                    attn.cpu() for attn in outputs.text_attentions
                ]
            
            # Extract text decoder attention patterns
            if hasattr(outputs, 'decoder_attentions') and outputs.decoder_attentions is not None:
                attention_patterns['text_decoder'] = [
                    attn.cpu() for attn in outputs.decoder_attentions
                ]
            
            # Extract cross-modal attention patterns
            if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions is not None:
                attention_patterns['cross_modal'] = [
                    attn.cpu() for attn in outputs.cross_attentions
                ]
            
            # Extract FIBER fusion attention patterns
            if hasattr(outputs, 'fiber_attentions') and outputs.fiber_attentions is not None:
                attention_patterns['fiber_fusion'] = [
                    attn.cpu() for attn in outputs.fiber_attentions
                ]
            
            # Extract episodic memory attention patterns
            if extract_memory and hasattr(outputs, 'memory_attention') and outputs.memory_attention is not None:
                attention_patterns['episodic_memory'] = outputs.memory_attention.cpu()
                
                # Store memory access pattern for analysis
                if hasattr(outputs, 'memory_usage') and outputs.memory_usage is not None:
                    self.memory_access_history.append({
                        'attention': outputs.memory_attention.cpu().numpy(),
                        'usage': outputs.memory_usage.cpu().numpy(),
                        'timestamp': len(self.memory_access_history)
                    })
            
            # Extract cross-modal alignment scores
            if hasattr(outputs, 'cross_modal_similarity') and outputs.cross_modal_similarity is not None:
                self.cross_modal_alignments.append({
                    'similarity': outputs.cross_modal_similarity.cpu().numpy(),
                    'timestamp': len(self.cross_modal_alignments)
                })
        
        self.model.train()
        return attention_patterns

    def visualize_text_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_idx: int,
        head_idx: int,
        title_prefix: str = "Text Attention"
    ) -> go.Figure:
        """
        Create interactive heatmap for text attention patterns
        """
        # Get single head attention [seq_len, seq_len]
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            attn_matrix = attention_weights[0, head_idx].numpy()
        elif attention_weights.dim() == 3:  # [heads, seq, seq]
            attn_matrix = attention_weights[head_idx].numpy()
        else:
            attn_matrix = attention_weights.numpy()
        
        # Ensure tokens match attention matrix size
        seq_len = attn_matrix.shape[0]
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            tokens.extend([f"<pad_{i}>" for i in range(len(tokens), seq_len)])
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attn_matrix,
            x=tokens,
            y=tokens,
            colorscale=self.config.color_scheme,
            colorbar=dict(title="Attention Weight"),
            hoverongaps=False,
            hovertemplate="From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"{title_prefix} - Layer {layer_idx}, Head {head_idx}",
            xaxis_title="To Token",
            yaxis_title="From Token",
            width=800,
            height=800,
            font=dict(size=10)
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig

    def visualize_multi_head_attention_grid(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer_idx: int,
        title_prefix: str = "Multi-Head Attention",
        max_heads: int = None
    ) -> go.Figure:
        """
        Create grid visualization for multiple attention heads
        """
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            attention_weights = attention_weights[0]  # Take first batch
        
        num_heads = attention_weights.shape[0]
        max_heads = max_heads or min(num_heads, self.config.max_heads_per_plot)
        
        # Calculate grid dimensions
        cols = 4
        rows = (max_heads + cols - 1) // cols
        
        # Create subplot grid
        subplot_titles = [f"Head {i+1}" for i in range(max_heads)]
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for head_idx in range(max_heads):
            row = head_idx // cols + 1
            col = head_idx % cols + 1
            
            attn_matrix = attention_weights[head_idx].numpy()
            
            # Ensure tokens match attention matrix size
            seq_len = attn_matrix.shape[0]
            display_tokens = tokens[:seq_len] if len(tokens) >= seq_len else tokens + [f"<pad_{i}>" for i in range(len(tokens), seq_len)]
            
            fig.add_trace(
                go.Heatmap(
                    z=attn_matrix,
                    x=display_tokens,
                    y=display_tokens,
                    colorscale=self.config.color_scheme,
                    showscale=(head_idx == 0),  # Only show colorbar for first subplot
                    hovertemplate=f"Head {head_idx+1}<br>From: %{{y}}<br>To: %{{x}}<br>Attention: %{{z:.3f}}<extra></extra>"
                ),
                row=row,
                col=col
            )
        
        fig.update_layout(
            title=f"{title_prefix} - Layer {layer_idx} (First {max_heads} heads)",
            height=300 * rows,
            width=1200,
            font=dict(size=8)
        )
        
        return fig

    def visualize_cross_modal_attention(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        cross_attention: torch.Tensor,
        text_tokens: List[str],
        vision_patch_info: List[str] = None
    ) -> go.Figure:
        """
        Visualize cross-modal attention between text and vision
        """
        if cross_attention.dim() == 4:  # [batch, heads, text_seq, vision_seq]
            # Average across heads for overall pattern
            cross_attn = cross_attention[0].mean(dim=0).numpy()
        else:
            cross_attn = cross_attention.numpy()
        
        # Create vision patch labels
        if vision_patch_info is None:
            vision_patches = [f"Vision_Patch_{i}" for i in range(cross_attn.shape[1])]
        else:
            vision_patches = vision_patch_info
        
        # Ensure tokens match attention dimensions
        text_len = cross_attn.shape[0]
        vision_len = cross_attn.shape[1]
        
        if len(text_tokens) > text_len:
            text_tokens = text_tokens[:text_len]
        elif len(text_tokens) < text_len:
            text_tokens.extend([f"<pad_{i}>" for i in range(len(text_tokens), text_len)])
        
        if len(vision_patches) > vision_len:
            vision_patches = vision_patches[:vision_len]
        elif len(vision_patches) < vision_len:
            vision_patches.extend([f"V_Patch_{i}" for i in range(len(vision_patches), vision_len)])
        
        fig = go.Figure(data=go.Heatmap(
            z=cross_attn,
            x=vision_patches,
            y=text_tokens,
            colorscale=self.config.color_scheme,
            colorbar=dict(title="Cross-Modal Attention"),
            hovertemplate="Text: %{y}<br>Vision: %{x}<br>Attention: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Cross-Modal Attention (Text → Vision)",
            xaxis_title="Vision Patches",
            yaxis_title="Text Tokens",
            width=1000,
            height=600,
            font=dict(size=10)
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig

    def visualize_episodic_memory_attention(
        self,
        memory_attention: torch.Tensor,
        query_tokens: List[str],
        memory_indices: List[int] = None,
        memory_descriptions: List[str] = None
    ) -> go.Figure:
        """
        Visualize attention patterns to episodic memory
        """
        if memory_attention.dim() == 3:  # [batch, seq_len, memory_size]
            mem_attn = memory_attention[0].numpy()
        else:
            mem_attn = memory_attention.numpy()
        
        # Create memory slot labels
        memory_size = mem_attn.shape[1]
        if memory_descriptions is not None:
            memory_labels = memory_descriptions[:memory_size]
        elif memory_indices is not None:
            memory_labels = [f"Memory_{idx}" for idx in memory_indices[:memory_size]]
        else:
            memory_labels = [f"Memory_Slot_{i}" for i in range(memory_size)]
        
        # Ensure query tokens match attention dimensions
        query_len = mem_attn.shape[0]
        if len(query_tokens) > query_len:
            query_tokens = query_tokens[:query_len]
        elif len(query_tokens) < query_len:
            query_tokens.extend([f"<pad_{i}>" for i in range(len(query_tokens), query_len)])
        
        fig = go.Figure(data=go.Heatmap(
            z=mem_attn,
            x=memory_labels,
            y=query_tokens,
            colorscale=self.config.color_scheme,
            colorbar=dict(title="Memory Attention"),
            hovertemplate="Query: %{y}<br>Memory: %{x}<br>Attention: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Episodic Memory Attention Patterns",
            xaxis_title="Memory Slots",
            yaxis_title="Query Tokens",
            width=1200,
            height=600,
            font=dict(size=10)
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig

    def visualize_attention_evolution(
        self,
        token_idx: int,
        token: str,
        attention_layers: List[torch.Tensor],
        component_name: str = "Text Encoder"
    ) -> go.Figure:
        """
        Visualize how attention for a specific token evolves across layers
        """
        num_layers = len(attention_layers)
        
        # Extract attention weights for the specific token across layers
        token_attention_evolution = []
        layer_numbers = []
        
        for layer_idx, layer_attn in enumerate(attention_layers):
            if layer_attn.dim() == 4:  # [batch, heads, seq, seq]
                # Average across batch and heads, get attention TO the token
                attn_to_token = layer_attn[0, :, :, token_idx].mean(dim=0)
                token_attention_evolution.append(attn_to_token.numpy())
            elif layer_attn.dim() == 3:  # [heads, seq, seq]
                attn_to_token = layer_attn[:, :, token_idx].mean(dim=0)
                token_attention_evolution.append(attn_to_token.numpy())
            
            layer_numbers.append(layer_idx + 1)
        
        # Create evolution plot
        evolution_matrix = np.array(token_attention_evolution).T  # [seq_len, num_layers]
        
        # Create tokens for y-axis (assuming same sequence length)
        seq_len = evolution_matrix.shape[0]
        y_tokens = [f"Token_{i}" for i in range(seq_len)]
        y_tokens[token_idx] = f"**{token}**"  # Highlight the target token
        
        fig = go.Figure(data=go.Heatmap(
            z=evolution_matrix,
            x=layer_numbers,
            y=y_tokens,
            colorscale=self.config.color_scheme,
            colorbar=dict(title="Attention Weight"),
            hovertemplate="Layer: %{x}<br>Token: %{y}<br>Attention: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Attention Evolution for Token '{token}' in {component_name}",
            xaxis_title="Layer",
            yaxis_title="Tokens Attending To Target",
            width=800,
            height=600,
            font=dict(size=10)
        )
        
        return fig

    def visualize_memory_access_patterns(self) -> go.Figure:
        """
        Visualize episodic memory access patterns over time
        """
        if not self.memory_access_history:
            logger.warning("No memory access history available for visualization")
            return None
        
        # Extract memory usage over time
        memory_usage_over_time = []
        timestamps = []
        
        for entry in self.memory_access_history:
            memory_usage_over_time.append(entry['usage'])
            timestamps.append(entry['timestamp'])
        
        memory_usage_matrix = np.array(memory_usage_over_time).T  # [memory_size, time_steps]
        
        fig = go.Figure(data=go.Heatmap(
            z=memory_usage_matrix,
            x=timestamps,
            y=[f"Memory_Slot_{i}" for i in range(memory_usage_matrix.shape[0])],
            colorscale=self.config.color_scheme,
            colorbar=dict(title="Memory Usage"),
            hovertemplate="Time: %{x}<br>Memory Slot: %{y}<br>Usage: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Episodic Memory Access Patterns Over Time",
            xaxis_title="Time Step",
            yaxis_title="Memory Slots",
            width=1000,
            height=600,
            font=dict(size=10)
        )
        
        return fig

    def generate_bertviz_visualization(
        self,
        attention_weights: List[torch.Tensor],
        tokens: List[str],
        component_name: str = "text_encoder"
    ):
        """
        Generate BertViz-style visualizations if available
        """
        if not BERTVIZ_AVAILABLE:
            logger.warning("BertViz not available for advanced visualization")
            return None
        
        try:
            # Convert attention weights to the format expected by BertViz
            formatted_attention = format_attention(attention_weights)
            
            # Generate head view
            head_view_html = head_view(formatted_attention, tokens)
            
            # Save HTML file
            html_path = self.save_dir / f"bertviz_{component_name}_head_view.html"
            with open(html_path, 'w') as f:
                f.write(head_view_html.data)
            
            logger.info(f"BertViz head view saved to: {html_path}")
            
            # Generate model view
            model_view_html = model_view(formatted_attention, tokens)
            html_path = self.save_dir / f"bertviz_{component_name}_model_view.html"
            with open(html_path, 'w') as f:
                f.write(model_view_html.data)
            
            logger.info(f"BertViz model view saved to: {html_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate BertViz visualization: {e}")

    def create_comprehensive_attention_report(
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        text_tokens: List[str] = None,
        save_name: str = "attention_report"
    ) -> Dict[str, str]:
        """
        Create a comprehensive attention visualization report
        """
        logger.info("🎯 Generating comprehensive attention visualization report...")
        
        # Extract attention patterns
        attention_patterns = self.extract_attention_patterns(
            input_ids, vision_features, attention_mask
        )
        
        # Get text tokens if not provided
        if text_tokens is None:
            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        saved_files = []
        
        # 1. Text encoder attention visualizations
        if 'text_encoder' in attention_patterns and attention_patterns['text_encoder']:
            for layer_idx, layer_attn in enumerate(attention_patterns['text_encoder']):
                # Multi-head grid visualization
                fig = self.visualize_multi_head_attention_grid(
                    layer_attn, text_tokens, layer_idx, "Text Encoder"
                )
                
                if self.config.save_interactive:
                    html_path = self.save_dir / f"{save_name}_text_encoder_layer_{layer_idx}.html"
                    fig.write_html(str(html_path))
                    saved_files.append(str(html_path))
                
                if self.config.save_static:
                    png_path = self.save_dir / f"{save_name}_text_encoder_layer_{layer_idx}.png"
                    fig.write_image(str(png_path), width=1200, height=800, scale=2)
                    saved_files.append(str(png_path))
                
                # Log to wandb if available
                if self.config.use_wandb and self.wandb_logger:
                    wandb.log({f"attention/text_encoder_layer_{layer_idx}": wandb.Html(fig.to_html())})
        
        # 2. Cross-modal attention visualization
        if 'cross_modal' in attention_patterns and attention_patterns['cross_modal']:
            cross_attn = attention_patterns['cross_modal'][0]  # Use first layer
            fig = self.visualize_cross_modal_attention(
                None, None, cross_attn, text_tokens
            )
            
            if self.config.save_interactive:
                html_path = self.save_dir / f"{save_name}_cross_modal_attention.html"
                fig.write_html(str(html_path))
                saved_files.append(str(html_path))
            
            if self.config.save_static:
                png_path = self.save_dir / f"{save_name}_cross_modal_attention.png"
                fig.write_image(str(png_path), width=1000, height=600, scale=2)
                saved_files.append(str(png_path))
            
            if self.config.use_wandb and self.wandb_logger:
                wandb.log({"attention/cross_modal": wandb.Html(fig.to_html())})
        
        # 3. Episodic memory attention visualization
        if 'episodic_memory' in attention_patterns and attention_patterns['episodic_memory'] is not None:
            fig = self.visualize_episodic_memory_attention(
                attention_patterns['episodic_memory'], text_tokens
            )
            
            if self.config.save_interactive:
                html_path = self.save_dir / f"{save_name}_episodic_memory_attention.html"
                fig.write_html(str(html_path))
                saved_files.append(str(html_path))
            
            if self.config.save_static:
                png_path = self.save_dir / f"{save_name}_episodic_memory_attention.png"
                fig.write_image(str(png_path), width=1200, height=600, scale=2)
                saved_files.append(str(png_path))
            
            if self.config.use_wandb and self.wandb_logger:
                wandb.log({"attention/episodic_memory": wandb.Html(fig.to_html())})
        
        # 4. Memory access patterns over time
        if self.memory_access_history:
            fig = self.visualize_memory_access_patterns()
            if fig:
                if self.config.save_interactive:
                    html_path = self.save_dir / f"{save_name}_memory_access_patterns.html"
                    fig.write_html(str(html_path))
                    saved_files.append(str(html_path))
                
                if self.config.save_static:
                    png_path = self.save_dir / f"{save_name}_memory_access_patterns.png"
                    fig.write_image(str(png_path), width=1000, height=600, scale=2)
                    saved_files.append(str(png_path))
                
                if self.config.use_wandb and self.wandb_logger:
                    wandb.log({"attention/memory_access_patterns": wandb.Html(fig.to_html())})
        
        # 5. Generate BertViz visualizations if available
        if 'text_encoder' in attention_patterns and attention_patterns['text_encoder']:
            self.generate_bertviz_visualization(
                attention_patterns['text_encoder'], text_tokens, "text_encoder"
            )
        
        # 6. Attention evolution for important tokens
        if 'text_encoder' in attention_patterns and attention_patterns['text_encoder']:
            # Find important tokens (non-special tokens)
            important_tokens = [(i, token) for i, token in enumerate(text_tokens) 
                              if not token.startswith('[') and len(token) > 1][:3]
            
            for token_idx, token in important_tokens:
                fig = self.visualize_attention_evolution(
                    token_idx, token, attention_patterns['text_encoder'], "Text Encoder"
                )
                
                if self.config.save_interactive:
                    html_path = self.save_dir / f"{save_name}_attention_evolution_{token}.html"
                    fig.write_html(str(html_path))
                    saved_files.append(str(html_path))
                
                if self.config.save_static:
                    png_path = self.save_dir / f"{save_name}_attention_evolution_{token}.png"
                    fig.write_image(str(png_path), width=800, height=600, scale=2)
                    saved_files.append(str(png_path))
        
        # Save metadata
        metadata = {
            'save_name': save_name,
            'num_text_tokens': len(text_tokens),
            'attention_components': list(attention_patterns.keys()),
            'saved_files': saved_files,
            'config': {
                'color_scheme': self.config.color_scheme,
                'max_heads_per_plot': self.config.max_heads_per_plot,
                'attention_threshold': self.config.attention_threshold
            }
        }
        
        metadata_path = self.save_dir / f"{save_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Comprehensive attention report generated")
        logger.info(f"  • Total files saved: {len(saved_files)}")
        logger.info(f"  • Report directory: {self.save_dir}")
        
        return {
            'saved_files': saved_files,
            'metadata_path': str(metadata_path),
            'save_directory': str(self.save_dir)
        }

    def analyze_attention_patterns(self) -> Dict[str, float]:
        """
        Analyze attention patterns for insights
        """
        if not self.attention_patterns['text_encoder']:
            logger.warning("No attention patterns stored for analysis")
            return {}
        
        analysis = {}
        
        # Analyze attention sparsity
        for component, patterns in self.attention_patterns.items():
            if patterns:
                if isinstance(patterns, list):
                    # Average sparsity across layers
                    sparsities = []
                    for pattern in patterns:
                        if pattern is not None:
                            sparsity = (pattern < self.config.attention_threshold).float().mean().item()
                            sparsities.append(sparsity)
                    if sparsities:
                        analysis[f'{component}_sparsity'] = np.mean(sparsities)
                else:
                    # Single pattern
                    sparsity = (patterns < self.config.attention_threshold).float().mean().item()
                    analysis[f'{component}_sparsity'] = sparsity
        
        # Analyze memory usage efficiency
        if self.memory_access_history:
            memory_usage_values = [entry['usage'] for entry in self.memory_access_history]
            if memory_usage_values:
                avg_memory_usage = np.mean([usage.mean() for usage in memory_usage_values])
                memory_efficiency = np.mean([usage.max() - usage.min() for usage in memory_usage_values])
                analysis['memory_usage_avg'] = avg_memory_usage
                analysis['memory_efficiency'] = memory_efficiency
        
        return analysis

def create_attention_visualizer(
    model: nn.Module,
    tokenizer,
    config: Dict = None,
    wandb_logger = None
) -> AdvancedAttentionVisualizer:
    """
    Factory function to create an attention visualizer
    """
    viz_config = AttentionVisualizationConfig()
    if config:
        for key, value in config.items():
            if hasattr(viz_config, key):
                setattr(viz_config, key, value)
    
    return AdvancedAttentionVisualizer(model, tokenizer, viz_config, wandb_logger)
