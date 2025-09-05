"""
Memory-Attention Integration Module
Combines advanced attention visualization with enhanced episodic memory
Provides comprehensive insights into memory-attention dynamics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import json
from dataclasses import dataclass

from .advanced_attention_visualizer import AdvancedAttentionVisualizer, AttentionVisualizationConfig
from .enhanced_episodic_memory import LarimarInspiredEpisodicMemory, EpisodicMemoryConfig

logger = logging.getLogger(__name__)

@dataclass
class MemoryAttentionConfig:
    """Configuration for memory-attention integration"""
    visualization_config: AttentionVisualizationConfig = None
    memory_config: EpisodicMemoryConfig = None
    save_dir: str = "./memory_attention_analysis"
    analysis_frequency: int = 100  # Steps between comprehensive analysis
    memory_attention_threshold: float = 0.1
    cross_modal_threshold: float = 0.15
    importance_decay: float = 0.95
    max_memory_snapshots: int = 50

class MemoryAttentionAnalyzer:
    """
    Comprehensive analyzer for memory-attention interactions
    Visualizes how attention patterns affect memory formation and retrieval
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: MemoryAttentionConfig = None,
        wandb_logger = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or MemoryAttentionConfig()
        self.wandb_logger = wandb_logger
        
        # Initialize components
        self.attention_visualizer = AdvancedAttentionVisualizer(
            model, tokenizer, 
            self.config.visualization_config or AttentionVisualizationConfig(),
            wandb_logger
        )
        
        # Create analysis directory
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis state
        self.memory_snapshots = []
        self.attention_memory_correlations = []
        self.cross_modal_memory_interactions = []
        self.step_count = 0
        
        # Find episodic memory in model
        self.episodic_memory = self._find_episodic_memory()
        
        logger.info(f"🧠🎯 Memory-Attention Analyzer initialized")
        logger.info(f"  • Save directory: {self.save_dir}")
        logger.info(f"  • Episodic memory found: {self.episodic_memory is not None}")
        logger.info(f"  • Analysis frequency: {self.config.analysis_frequency}")

    def _find_episodic_memory(self) -> Optional[LarimarInspiredEpisodicMemory]:
        """Find the episodic memory module in the model"""
        for name, module in self.model.named_modules():
            if isinstance(module, LarimarInspiredEpisodicMemory):
                logger.info(f"Found episodic memory module: {name}")
                return module
        
        # Fallback: look for any module with memory-like attributes
        for name, module in self.model.named_modules():
            if hasattr(module, 'memory_mean') and hasattr(module, 'memory_usage'):
                logger.info(f"Found memory-like module: {name}")
                return module
        
        logger.warning("No episodic memory module found in model")
        return None

    def capture_memory_attention_state(
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        text_tokens: List[str] = None
    ) -> Dict[str, Any]:
        """
        Capture comprehensive memory-attention state during model forward pass
        """
        if text_tokens is None:
            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        
        # Extract attention patterns
        attention_patterns = self.attention_visualizer.extract_attention_patterns(
            input_ids, vision_features, attention_mask, extract_memory=True
        )
        
        # Get memory state if available
        memory_state = {}
        if self.episodic_memory:
            memory_stats = self.episodic_memory.get_memory_statistics()
            memory_state = {
                'memory_usage': memory_stats.get('memory_usage', []),
                'memory_importance': memory_stats.get('most_important_memories', []),
                'memory_utilization': memory_stats.get('memory_utilization', 0.0),
                'active_memories': memory_stats.get('active_memories', 0),
                'total_updates': memory_stats.get('total_updates', 0)
            }
        
        # Combine state
        combined_state = {
            'step': self.step_count,
            'attention_patterns': attention_patterns,
            'memory_state': memory_state,
            'text_tokens': text_tokens,
            'timestamp': torch.tensor(self.step_count, dtype=torch.float32)
        }
        
        # Store snapshot if within limits
        if len(self.memory_snapshots) < self.config.max_memory_snapshots:
            self.memory_snapshots.append(combined_state)
        else:
            # Remove oldest snapshot
            self.memory_snapshots.pop(0)
            self.memory_snapshots.append(combined_state)
        
        self.step_count += 1
        return combined_state

    def analyze_memory_attention_correlation(self) -> Dict[str, float]:
        """
        Analyze correlation between attention patterns and memory usage
        """
        if len(self.memory_snapshots) < 2:
            return {}
        
        correlations = {}
        
        # Extract memory usage and attention patterns over time
        memory_usage_over_time = []
        attention_entropy_over_time = []
        cross_modal_attention_strength = []
        
        for snapshot in self.memory_snapshots:
            if 'memory_state' in snapshot and snapshot['memory_state']:
                memory_usage_over_time.append(snapshot['memory_state'].get('memory_utilization', 0))
            
            # Calculate attention entropy
            if 'attention_patterns' in snapshot:
                patterns = snapshot['attention_patterns']
                if 'text_encoder' in patterns and patterns['text_encoder']:
                    # Average entropy across all attention heads
                    entropy_values = []
                    for layer_attn in patterns['text_encoder']:
                        if layer_attn is not None:
                            attn_probs = torch.softmax(layer_attn, dim=-1)
                            entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()
                            entropy_values.append(entropy.item())
                    
                    if entropy_values:
                        attention_entropy_over_time.append(np.mean(entropy_values))
                
                # Cross-modal attention strength
                if 'cross_modal' in patterns and patterns['cross_modal']:
                    cross_modal_strength = patterns['cross_modal'][0].max().item()
                    cross_modal_attention_strength.append(cross_modal_strength)
        
        # Compute correlations
        if len(memory_usage_over_time) > 1 and len(attention_entropy_over_time) > 1:
            min_len = min(len(memory_usage_over_time), len(attention_entropy_over_time))
            memory_usage = np.array(memory_usage_over_time[:min_len])
            attention_entropy = np.array(attention_entropy_over_time[:min_len])
            
            correlations['memory_attention_correlation'] = np.corrcoef(memory_usage, attention_entropy)[0, 1]
        
        if len(memory_usage_over_time) > 1 and len(cross_modal_attention_strength) > 1:
            min_len = min(len(memory_usage_over_time), len(cross_modal_attention_strength))
            memory_usage = np.array(memory_usage_over_time[:min_len])
            cross_modal_strength = np.array(cross_modal_attention_strength[:min_len])
            
            correlations['memory_crossmodal_correlation'] = np.corrcoef(memory_usage, cross_modal_strength)[0, 1]
        
        # Store correlation for trend analysis
        self.attention_memory_correlations.append({
            'step': self.step_count,
            'correlations': correlations
        })
        
        return correlations

    def visualize_memory_attention_dynamics(self, save_name: str = "memory_attention_dynamics") -> Dict[str, str]:
        """
        Create comprehensive visualizations of memory-attention dynamics
        """
        if len(self.memory_snapshots) < 2:
            logger.warning("Not enough memory snapshots for visualization")
            return {}
        
        saved_files = []
        
        # 1. Memory utilization over time
        fig = self._create_memory_utilization_plot()
        if fig:
            html_path = self.save_dir / f"{save_name}_memory_utilization.html"
            fig.write_html(str(html_path))
            saved_files.append(str(html_path))
            
            png_path = self.save_dir / f"{save_name}_memory_utilization.png"
            fig.write_image(str(png_path), width=1000, height=600, scale=2)
            saved_files.append(str(png_path))
        
        # 2. Attention-Memory correlation heatmap
        fig = self._create_attention_memory_correlation_heatmap()
        if fig:
            html_path = self.save_dir / f"{save_name}_attention_memory_correlation.html"
            fig.write_html(str(html_path))
            saved_files.append(str(html_path))
            
            png_path = self.save_dir / f"{save_name}_attention_memory_correlation.png"
            fig.write_image(str(png_path), width=800, height=600, scale=2)
            saved_files.append(str(png_path))
        
        # 3. Cross-modal memory interaction analysis
        fig = self._create_cross_modal_memory_analysis()
        if fig:
            html_path = self.save_dir / f"{save_name}_cross_modal_memory.html"
            fig.write_html(str(html_path))
            saved_files.append(str(html_path))
            
            png_path = self.save_dir / f"{save_name}_cross_modal_memory.png"
            fig.write_image(str(png_path), width=1200, height=800, scale=2)
            saved_files.append(str(png_path))
        
        # 4. Memory attention flow visualization
        fig = self._create_memory_attention_flow()
        if fig:
            html_path = self.save_dir / f"{save_name}_memory_attention_flow.html"
            fig.write_html(str(html_path))
            saved_files.append(str(html_path))
            
            png_path = self.save_dir / f"{save_name}_memory_attention_flow.png"
            fig.write_image(str(png_path), width=1000, height=800, scale=2)
            saved_files.append(str(png_path))
        
        # 5. Memory importance evolution
        fig = self._create_memory_importance_evolution()
        if fig:
            html_path = self.save_dir / f"{save_name}_memory_importance_evolution.html"
            fig.write_html(str(html_path))
            saved_files.append(str(html_path))
            
            png_path = self.save_dir / f"{save_name}_memory_importance_evolution.png"
            fig.write_image(str(png_path), width=1000, height=600, scale=2)
            saved_files.append(str(png_path))
        
        # Log to wandb if available
        if self.wandb_logger:
            for i, (html_file, png_file) in enumerate(zip(saved_files[::2], saved_files[1::2])):
                chart_name = Path(html_file).stem
                wandb.log({f"memory_attention/{chart_name}": wandb.Html(open(html_file).read())})
        
        logger.info(f"✅ Memory-attention dynamics visualized")
        logger.info(f"  • Files saved: {len(saved_files)}")
        
        return {'saved_files': saved_files}

    def _create_memory_utilization_plot(self) -> Optional[go.Figure]:
        """Create memory utilization over time plot"""
        steps = []
        memory_utilizations = []
        active_memories = []
        total_updates = []
        
        for snapshot in self.memory_snapshots:
            steps.append(snapshot['step'])
            memory_state = snapshot.get('memory_state', {})
            memory_utilizations.append(memory_state.get('memory_utilization', 0))
            active_memories.append(memory_state.get('active_memories', 0))
            total_updates.append(memory_state.get('total_updates', 0))
        
        if not steps:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=memory_utilizations,
            mode='lines+markers',
            name='Memory Utilization',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=[m / max(active_memories) if max(active_memories) > 0 else 0 for m in active_memories],
            mode='lines+markers',
            name='Active Memories (Normalized)',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Memory Utilization and Activity Over Time',
            xaxis_title='Training Step',
            yaxis_title='Memory Utilization',
            yaxis2=dict(
                title='Normalized Active Memories',
                overlaying='y',
                side='right',
                color='red'
            ),
            width=1000,
            height=600,
            hovermode='x unified'
        )
        
        return fig

    def _create_attention_memory_correlation_heatmap(self) -> Optional[go.Figure]:
        """Create correlation heatmap between attention and memory patterns"""
        if len(self.attention_memory_correlations) < 2:
            return None
        
        # Extract correlation values over time
        correlation_matrix = []
        correlation_labels = []
        steps = []
        
        for entry in self.attention_memory_correlations:
            correlations = entry['correlations']
            if correlations:
                correlation_matrix.append(list(correlations.values()))
                if not correlation_labels:
                    correlation_labels = list(correlations.keys())
                steps.append(entry['step'])
        
        if not correlation_matrix:
            return None
        
        correlation_matrix = np.array(correlation_matrix).T  # [n_correlations, n_steps]
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=steps,
            y=correlation_labels,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation"),
            hovertemplate="Step: %{x}<br>Metric: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title='Attention-Memory Correlation Over Time',
            xaxis_title='Training Step',
            yaxis_title='Correlation Metrics',
            width=800,
            height=600
        )
        
        return fig

    def _create_cross_modal_memory_analysis(self) -> Optional[go.Figure]:
        """Create cross-modal memory interaction analysis"""
        if not self.episodic_memory or not hasattr(self.episodic_memory, 'text_memory_contributions'):
            return None
        
        # Extract cross-modal contributions over time
        text_contributions = []
        vision_contributions = []
        cross_modal_ratios = []
        steps = []
        
        for snapshot in self.memory_snapshots:
            if 'memory_state' in snapshot:
                memory_state = snapshot['memory_state']
                # This would need to be added to memory statistics
                text_contrib = 0.7  # Placeholder
                vision_contrib = 0.3  # Placeholder
                
                text_contributions.append(text_contrib)
                vision_contributions.append(vision_contrib)
                cross_modal_ratios.append(text_contrib / (vision_contrib + 1e-8))
                steps.append(snapshot['step'])
        
        if not steps:
            return None
        
        # Create subplot with multiple charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Text vs Vision Contributions', 'Cross-Modal Ratio', 
                          'Contribution Balance', 'Memory Modality Distribution'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Text vs Vision contributions
        fig.add_trace(
            go.Scatter(x=steps, y=text_contributions, name='Text', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=vision_contributions, name='Vision', line=dict(color='red')),
            row=1, col=1
        )
        
        # Cross-modal ratio
        fig.add_trace(
            go.Scatter(x=steps, y=cross_modal_ratios, name='Text/Vision Ratio', line=dict(color='green')),
            row=1, col=2
        )
        
        # Contribution balance
        fig.add_trace(
            go.Scatter(x=steps, y=[abs(t - v) for t, v in zip(text_contributions, vision_contributions)], 
                      name='Balance', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Pie chart of overall distribution
        avg_text = np.mean(text_contributions) if text_contributions else 0
        avg_vision = np.mean(vision_contributions) if vision_contributions else 0
        
        fig.add_trace(
            go.Pie(labels=['Text', 'Vision'], values=[avg_text, avg_vision], 
                  marker_colors=['blue', 'red']),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Cross-Modal Memory Interaction Analysis',
            width=1200,
            height=800,
            showlegend=True
        )
        
        return fig

    def _create_memory_attention_flow(self) -> Optional[go.Figure]:
        """Create memory attention flow visualization"""
        # Extract attention flow patterns
        flow_data = []
        
        for snapshot in self.memory_snapshots:
            attention_patterns = snapshot.get('attention_patterns', {})
            if 'episodic_memory' in attention_patterns and attention_patterns['episodic_memory'] is not None:
                # Analyze attention flow to memory
                memory_attention = attention_patterns['episodic_memory']
                if memory_attention.dim() == 3:  # [batch, seq_len, memory_size]
                    memory_attention = memory_attention[0]  # Take first batch
                
                # Calculate flow metrics
                attention_concentration = memory_attention.max(dim=-1)[0].mean().item()
                attention_dispersion = memory_attention.std(dim=-1).mean().item()
                memory_coverage = (memory_attention > self.config.memory_attention_threshold).float().mean().item()
                
                flow_data.append({
                    'step': snapshot['step'],
                    'concentration': attention_concentration,
                    'dispersion': attention_dispersion,
                    'coverage': memory_coverage
                })
        
        if not flow_data:
            return None
        
        steps = [d['step'] for d in flow_data]
        concentrations = [d['concentration'] for d in flow_data]
        dispersions = [d['dispersion'] for d in flow_data]
        coverages = [d['coverage'] for d in flow_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps, y=concentrations, name='Attention Concentration',
            mode='lines+markers', line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=steps, y=dispersions, name='Attention Dispersion',
            mode='lines+markers', line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=steps, y=coverages, name='Memory Coverage',
            mode='lines+markers', line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Memory Attention Flow Dynamics',
            xaxis_title='Training Step',
            yaxis_title='Attention Metrics',
            width=1000,
            height=800,
            hovermode='x unified'
        )
        
        return fig

    def _create_memory_importance_evolution(self) -> Optional[go.Figure]:
        """Create memory importance evolution visualization"""
        if not self.episodic_memory:
            return None
        
        # Extract memory importance over time
        importance_evolution = []
        
        for snapshot in self.memory_snapshots:
            memory_state = snapshot.get('memory_state', {})
            important_memories = memory_state.get('memory_importance', [])
            if important_memories:
                importance_evolution.append({
                    'step': snapshot['step'],
                    'top_memories': important_memories[:5],  # Top 5 most important
                    'avg_importance': np.mean(important_memories) if isinstance(important_memories, list) else 0
                })
        
        if not importance_evolution:
            return None
        
        steps = [d['step'] for d in importance_evolution]
        avg_importances = [d['avg_importance'] for d in importance_evolution]
        
        fig = go.Figure()
        
        # Average importance over time
        fig.add_trace(go.Scatter(
            x=steps, y=avg_importances, name='Average Memory Importance',
            mode='lines+markers', line=dict(color='purple', width=2)
        ))
        
        # Add trend line
        if len(steps) > 1:
            z = np.polyfit(steps, avg_importances, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=steps, y=p(steps), name='Trend',
                mode='lines', line=dict(color='orange', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Memory Importance Evolution',
            xaxis_title='Training Step',
            yaxis_title='Importance Score',
            width=1000,
            height=600,
            hovermode='x unified'
        )
        
        return fig

    def generate_comprehensive_report(
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        save_name: str = "comprehensive_memory_attention_report"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive memory-attention analysis report
        """
        logger.info("🧠🎯 Generating comprehensive memory-attention report...")
        
        # Capture current state
        current_state = self.capture_memory_attention_state(
            input_ids, vision_features, attention_mask
        )
        
        # Analyze correlations
        correlations = self.analyze_memory_attention_correlation()
        
        # Generate attention visualizations
        attention_report = self.attention_visualizer.create_comprehensive_attention_report(
            input_ids, vision_features, attention_mask, save_name=f"{save_name}_attention"
        )
        
        # Generate memory-attention dynamics visualizations
        dynamics_report = self.visualize_memory_attention_dynamics(save_name=f"{save_name}_dynamics")
        
        # Analyze attention patterns
        attention_analysis = self.attention_visualizer.analyze_attention_patterns()
        
        # Get memory statistics
        memory_stats = {}
        if self.episodic_memory:
            memory_stats = self.episodic_memory.get_memory_statistics()
        
        # Compile comprehensive report
        report = {
            'current_state': current_state,
            'correlations': correlations,
            'attention_analysis': attention_analysis,
            'memory_statistics': memory_stats,
            'attention_visualizations': attention_report,
            'dynamics_visualizations': dynamics_report,
            'analysis_metadata': {
                'total_snapshots': len(self.memory_snapshots),
                'analysis_step': self.step_count,
                'correlation_entries': len(self.attention_memory_correlations),
                'config': {
                    'memory_attention_threshold': self.config.memory_attention_threshold,
                    'cross_modal_threshold': self.config.cross_modal_threshold,
                    'analysis_frequency': self.config.analysis_frequency
                }
            }
        }
        
        # Save report metadata
        report_path = self.save_dir / f"{save_name}_report.json"
        with open(report_path, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_report = self._prepare_report_for_json(report)
            json.dump(json_report, f, indent=2)
        
        logger.info(f"✅ Comprehensive memory-attention report generated")
        logger.info(f"  • Report saved to: {report_path}")
        logger.info(f"  • Total visualizations: {len(attention_report.get('saved_files', [])) + len(dynamics_report.get('saved_files', []))}")
        
        return report

    def _prepare_report_for_json(self, report: Dict) -> Dict:
        """Prepare report for JSON serialization by converting tensors"""
        json_report = {}
        
        for key, value in report.items():
            if isinstance(value, dict):
                json_report[key] = self._prepare_report_for_json(value)
            elif isinstance(value, torch.Tensor):
                json_report[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                json_report[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                json_report[key] = [
                    item.cpu().numpy().tolist() if isinstance(item, torch.Tensor) else item
                    for item in value
                ]
            else:
                json_report[key] = value
        
        return json_report

    def should_run_analysis(self) -> bool:
        """Check if comprehensive analysis should be run"""
        return self.step_count % self.config.analysis_frequency == 0 and self.step_count > 0

def create_memory_attention_analyzer(
    model: nn.Module,
    tokenizer,
    config: Dict = None,
    wandb_logger = None
) -> MemoryAttentionAnalyzer:
    """Factory function to create memory-attention analyzer"""
    analyzer_config = MemoryAttentionConfig()
    if config:
        for key, value in config.items():
            if hasattr(analyzer_config, key):
                setattr(analyzer_config, key, value)
    
    return MemoryAttentionAnalyzer(model, tokenizer, analyzer_config, wandb_logger)
