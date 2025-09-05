"""
Robot Reasoning Visualizations for BitGen
Comprehensive visualization suite for robot selection reasoning capabilities
Integrates with Weights & Biases for real-time monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import torch
import wandb
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from collections import defaultdict, Counter
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class RobotReasoningVisualizer:
    """Advanced visualizer for robot reasoning capabilities with W&B integration"""
    
    def __init__(self, save_dir: str = "./reasoning_visualizations", use_wandb: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Robot types and capabilities
        self.robot_types = ['Drone', 'Underwater Robot', 'Humanoid', 'Robot with Wheels', 'Robot with Legs']
        self.task_categories = ['Aerial', 'Underwater', 'Manipulation', 'Transport', 'Inspection', 'Navigation']
        
        # Tracking data
        self.reasoning_history = []
        self.selection_history = []
        self.accuracy_history = []
        self.confusion_data = defaultdict(lambda: defaultdict(int))
        self.capability_usage = defaultdict(int)
        self.reasoning_quality_scores = []
        
        # Color schemes for consistent visualization
        self.robot_colors = {
            'Drone': '#FF6B6B',
            'Underwater Robot': '#4ECDC4', 
            'Humanoid': '#45B7D1',
            'Robot with Wheels': '#96CEB4',
            'Robot with Legs': '#FFEAA7'
        }
        
        self.task_colors = {
            'Aerial': '#FF6B6B',
            'Underwater': '#4ECDC4',
            'Manipulation': '#45B7D1', 
            'Transport': '#96CEB4',
            'Inspection': '#FFEAA7',
            'Navigation': '#DDA0DD'
        }
        
        logger.info("🎨 Robot Reasoning Visualizer initialized")
    
    def update_reasoning_data(self, 
                            predicted_robots: List[str],
                            actual_robots: List[str], 
                            reasoning_trace: str,
                            task_description: str,
                            confidence_scores: List[float],
                            step: int):
        """Update reasoning data for visualization"""
        
        # Record selection accuracy
        top1_correct = len(predicted_robots) > 0 and predicted_robots[0] in actual_robots
        topk_correct = any(robot in actual_robots for robot in predicted_robots[:3])
        
        self.accuracy_history.append({
            'step': step,
            'top1_accuracy': 1.0 if top1_correct else 0.0,
            'top3_accuracy': 1.0 if topk_correct else 0.0,
            'confidence': max(confidence_scores) if confidence_scores else 0.0
        })
        
        # Update confusion matrix data
        if len(predicted_robots) > 0 and len(actual_robots) > 0:
            predicted = predicted_robots[0]  # Top-1 prediction
            actual = actual_robots[0]  # Ground truth
            self.confusion_data[actual][predicted] += 1
        
        # Track capability usage
        for robot in predicted_robots:
            self.capability_usage[robot] += 1
        
        # Analyze reasoning quality
        reasoning_quality = self._analyze_reasoning_quality(reasoning_trace, task_description)
        self.reasoning_quality_scores.append({
            'step': step,
            'format_score': reasoning_quality['format_score'],
            'logic_score': reasoning_quality['logic_score'],
            'evidence_score': reasoning_quality['evidence_score'],
            'overall_score': reasoning_quality['overall_score']
        })
        
        # Store reasoning trace for XML visualization
        self.reasoning_history.append({
            'step': step,
            'task': task_description,
            'predicted_robots': predicted_robots,
            'actual_robots': actual_robots,
            'reasoning_trace': reasoning_trace,
            'confidence_scores': confidence_scores,
            'timestamp': datetime.now().isoformat()
        })
    
    def _analyze_reasoning_quality(self, reasoning_trace: str, task_description: str) -> Dict[str, float]:
        """Analyze the quality of reasoning trace"""
        scores = {
            'format_score': 0.0,
            'logic_score': 0.0, 
            'evidence_score': 0.0,
            'overall_score': 0.0
        }
        
        # Format score: Check for proper XML structure
        if '<reasoning>' in reasoning_trace and '</reasoning>' in reasoning_trace:
            scores['format_score'] += 0.5
        if '<answer>' in reasoning_trace and '</answer>' in reasoning_trace:
            scores['format_score'] += 0.5
        
        # Logic score: Check for logical reasoning patterns
        logic_keywords = ['because', 'therefore', 'since', 'due to', 'requires', 'suitable', 'capable']
        logic_count = sum(1 for keyword in logic_keywords if keyword.lower() in reasoning_trace.lower())
        scores['logic_score'] = min(1.0, logic_count / 3.0)
        
        # Evidence score: Check for evidence from task description
        task_words = set(task_description.lower().split())
        reasoning_words = set(reasoning_trace.lower().split())
        overlap = len(task_words.intersection(reasoning_words))
        scores['evidence_score'] = min(1.0, overlap / max(len(task_words), 1))
        
        # Overall score
        scores['overall_score'] = np.mean(list(scores.values())[:-1])
        
        return scores
    
    def create_reasoning_evolution_heatmap(self, window_size: int = 100) -> go.Figure:
        """Create reasoning evolution heatmap showing improvement over time"""
        
        if len(self.reasoning_quality_scores) < window_size:
            return None
        
        # Prepare data for heatmap
        steps = [item['step'] for item in self.reasoning_quality_scores[-window_size:]]
        aspects = ['Format', 'Logic', 'Evidence', 'Overall']
        
        # Create rolling averages for smoothness
        window = 10
        heatmap_data = []
        
        for aspect_key in ['format_score', 'logic_score', 'evidence_score', 'overall_score']:
            scores = [item[aspect_key] for item in self.reasoning_quality_scores[-window_size:]]
            smoothed = pd.Series(scores).rolling(window=window, min_periods=1).mean().tolist()
            heatmap_data.append(smoothed)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=steps,
            y=aspects,
            colorscale='RdYlBu',
            zmid=0.5,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Quality Score"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🧠 Reasoning Quality Evolution Over Training",
            xaxis_title="Training Step",
            yaxis_title="Reasoning Aspect",
            font=dict(size=12),
            height=400,
            width=800
        )
        
        return fig
    
    def create_enhanced_confusion_matrix(self) -> go.Figure:
        """Create enhanced confusion matrix with additional metrics"""
        
        if not self.confusion_data:
            return None
        
        # Convert confusion data to matrix
        matrix = np.zeros((len(self.robot_types), len(self.robot_types)))
        for i, actual in enumerate(self.robot_types):
            for j, predicted in enumerate(self.robot_types):
                matrix[i][j] = self.confusion_data[actual][predicted]
        
        # Normalize for percentages
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums!=0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=normalized_matrix,
            x=self.robot_types,
            y=self.robot_types,
            colorscale='Blues',
            text=[[f'{matrix[i][j]:.0f}<br>({normalized_matrix[i][j]:.1%})' 
                   for j in range(len(self.robot_types))] 
                  for i in range(len(self.robot_types))],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Accuracy")
        ))
        
        fig.update_layout(
            title="🎯 Robot Selection Confusion Matrix",
            xaxis_title="Predicted Robot",
            yaxis_title="Actual Robot",
            font=dict(size=12),
            height=500,
            width=600
        )
        
        return fig
    
    def create_robot_capability_utilization_chart(self) -> go.Figure:
        """Create radar chart showing robot capability utilization"""
        
        if not self.capability_usage:
            return None
        
        # Calculate utilization percentages
        total_selections = sum(self.capability_usage.values())
        utilization_pct = {robot: (count / total_selections) * 100 
                          for robot, count in self.capability_usage.items()}
        
        # Ensure all robot types are represented
        for robot in self.robot_types:
            if robot not in utilization_pct:
                utilization_pct[robot] = 0.0
        
        # Create radar chart
        categories = list(utilization_pct.keys())
        values = list(utilization_pct.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Utilization %',
            line_color='rgba(78, 205, 196, 0.8)',
            fillcolor='rgba(78, 205, 196, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1] if values else [0, 100]
                )),
            title="🤖 Robot Capability Utilization",
            font=dict(size=12),
            height=500,
            width=500
        )
        
        return fig
    
    def create_xml_reasoning_trace_visualization(self, recent_traces: int = 5) -> go.Figure:
        """Create visualization of recent XML reasoning traces"""
        
        if len(self.reasoning_history) < recent_traces:
            return None
        
        recent_reasoning = self.reasoning_history[-recent_traces:]
        
        # Create flow diagram representation
        fig = make_subplots(
            rows=recent_traces, 
            cols=1,
            subplot_titles=[f"Step {trace['step']}: {trace['task'][:50]}..." 
                           for trace in recent_reasoning],
            vertical_spacing=0.05
        )
        
        for i, trace in enumerate(recent_reasoning, 1):
            # Extract reasoning components
            reasoning_text = trace['reasoning_trace']
            predicted = trace['predicted_robots'][:3]  # Top-3
            confidence = trace['confidence_scores'][:3] if trace['confidence_scores'] else [0.5, 0.3, 0.1]
            
            # Create bar chart for this trace
            fig.add_trace(
                go.Bar(
                    x=predicted,
                    y=confidence,
                    marker_color=[self.robot_colors.get(robot, '#CCCCCC') for robot in predicted],
                    showlegend=False,
                    text=[f'{conf:.2f}' for conf in confidence],
                    textposition='auto'
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title="🔍 Recent XML Reasoning Traces",
            height=200 * recent_traces,
            font=dict(size=10)
        )
        
        return fig
    
    def create_accuracy_trends_chart(self, window_size: int = 50) -> go.Figure:
        """Create accuracy trends over training"""
        
        if len(self.accuracy_history) < window_size:
            return None
        
        recent_data = self.accuracy_history[-window_size:]
        steps = [item['step'] for item in recent_data]
        top1_acc = [item['top1_accuracy'] for item in recent_data]
        top3_acc = [item['top3_accuracy'] for item in recent_data]
        confidence = [item['confidence'] for item in recent_data]
        
        # Calculate rolling averages
        window = 10
        top1_smooth = pd.Series(top1_acc).rolling(window=window, min_periods=1).mean()
        top3_smooth = pd.Series(top3_acc).rolling(window=window, min_periods=1).mean()
        conf_smooth = pd.Series(confidence).rolling(window=window, min_periods=1).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps, y=top1_smooth,
            mode='lines',
            name='Top-1 Accuracy',
            line=dict(color='#FF6B6B', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=steps, y=top3_smooth,
            mode='lines',
            name='Top-3 Accuracy',
            line=dict(color='#4ECDC4', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=steps, y=conf_smooth,
            mode='lines',
            name='Confidence',
            line=dict(color='#45B7D1', width=2, dash='dash'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="📈 Robot Selection Accuracy Trends",
            xaxis_title="Training Step",
            yaxis_title="Accuracy",
            yaxis2=dict(title="Confidence", overlaying='y', side='right'),
            font=dict(size=12),
            height=400,
            width=800
        )
        
        return fig
    
    def log_to_wandb(self, step: int):
        """Log all visualizations to Weights & Biases"""
        
        if not self.use_wandb or not wandb.run:
            return
        
        try:
            # Log scalar metrics
            if self.accuracy_history:
                latest_acc = self.accuracy_history[-1]
                wandb.log({
                    "reasoning/top1_accuracy": latest_acc['top1_accuracy'],
                    "reasoning/top3_accuracy": latest_acc['top3_accuracy'], 
                    "reasoning/confidence": latest_acc['confidence']
                }, step=step)
            
            if self.reasoning_quality_scores:
                latest_quality = self.reasoning_quality_scores[-1]
                wandb.log({
                    "reasoning/format_score": latest_quality['format_score'],
                    "reasoning/logic_score": latest_quality['logic_score'],
                    "reasoning/evidence_score": latest_quality['evidence_score'],
                    "reasoning/overall_quality": latest_quality['overall_score']
                }, step=step)
            
            # Log visualizations (every 100 steps to avoid spam)
            if step % 100 == 0:
                
                # Reasoning Evolution Heatmap
                heatmap_fig = self.create_reasoning_evolution_heatmap()
                if heatmap_fig:
                    wandb.log({"reasoning/evolution_heatmap": wandb.Plotly(heatmap_fig)}, step=step)
                
                # Enhanced Confusion Matrix
                confusion_fig = self.create_enhanced_confusion_matrix()
                if confusion_fig:
                    wandb.log({"reasoning/confusion_matrix": wandb.Plotly(confusion_fig)}, step=step)
                
                # Robot Capability Utilization
                utilization_fig = self.create_robot_capability_utilization_chart()
                if utilization_fig:
                    wandb.log({"reasoning/capability_utilization": wandb.Plotly(utilization_fig)}, step=step)
                
                # XML Reasoning Traces
                traces_fig = self.create_xml_reasoning_trace_visualization()
                if traces_fig:
                    wandb.log({"reasoning/xml_traces": wandb.Plotly(traces_fig)}, step=step)
                
                # Accuracy Trends
                trends_fig = self.create_accuracy_trends_chart()
                if trends_fig:
                    wandb.log({"reasoning/accuracy_trends": wandb.Plotly(trends_fig)}, step=step)
            
            logger.info(f"✅ Reasoning visualizations logged to W&B at step {step}")
            
        except Exception as e:
            logger.warning(f"Failed to log reasoning visualizations to W&B: {e}")
    
    def save_visualizations(self, step: int):
        """Save visualizations to disk"""
        
        step_dir = self.save_dir / f"step_{step}"
        step_dir.mkdir(exist_ok=True)
        
        # Save all plots
        plots = {
            'reasoning_evolution': self.create_reasoning_evolution_heatmap(),
            'confusion_matrix': self.create_enhanced_confusion_matrix(),
            'capability_utilization': self.create_robot_capability_utilization_chart(),
            'xml_traces': self.create_xml_reasoning_trace_visualization(),
            'accuracy_trends': self.create_accuracy_trends_chart()
        }
        
        for name, fig in plots.items():
            if fig:
                fig.write_html(step_dir / f"{name}.html")
                fig.write_image(step_dir / f"{name}.png", width=800, height=600)
        
        # Save raw data
        with open(step_dir / "reasoning_data.json", 'w') as f:
            json.dump({
                'accuracy_history': self.accuracy_history,
                'reasoning_quality_scores': self.reasoning_quality_scores,
                'capability_usage': dict(self.capability_usage),
                'confusion_data': {k: dict(v) for k, v in self.confusion_data.items()}
            }, f, indent=2)
        
        logger.info(f"💾 Reasoning visualizations saved to {step_dir}")
    
    def generate_reasoning_report(self) -> str:
        """Generate comprehensive reasoning performance report"""
        
        if not self.accuracy_history:
            return "No reasoning data available yet."
        
        # Calculate summary statistics
        recent_acc = self.accuracy_history[-10:] if len(self.accuracy_history) >= 10 else self.accuracy_history
        avg_top1 = np.mean([item['top1_accuracy'] for item in recent_acc])
        avg_top3 = np.mean([item['top3_accuracy'] for item in recent_acc])
        avg_conf = np.mean([item['confidence'] for item in recent_acc])
        
        recent_quality = self.reasoning_quality_scores[-10:] if len(self.reasoning_quality_scores) >= 10 else self.reasoning_quality_scores
        avg_quality = np.mean([item['overall_score'] for item in recent_quality]) if recent_quality else 0.0
        
        # Most/least used robots
        sorted_usage = sorted(self.capability_usage.items(), key=lambda x: x[1], reverse=True)
        most_used = sorted_usage[0] if sorted_usage else ("None", 0)
        least_used = sorted_usage[-1] if sorted_usage else ("None", 0)
        
        report = f"""
🤖 **Robot Reasoning Performance Report**
{'='*50}

📊 **Accuracy Metrics** (Recent 10 batches)
• Top-1 Accuracy: {avg_top1:.2%}
• Top-3 Accuracy: {avg_top3:.2%}
• Average Confidence: {avg_conf:.2f}

🧠 **Reasoning Quality** (Recent 10 batches)
• Overall Quality Score: {avg_quality:.2f}/1.0

🎯 **Robot Utilization**
• Most Selected: {most_used[0]} ({most_used[1]} times)
• Least Selected: {least_used[0]} ({least_used[1]} times)

📈 **Training Progress**
• Total Reasoning Samples: {len(self.accuracy_history)}
• Quality Assessments: {len(self.reasoning_quality_scores)}
"""
        
        return report


def create_reasoning_visualizer(config: Dict, use_wandb: bool = True) -> RobotReasoningVisualizer:
    """Factory function to create reasoning visualizer"""
    
    save_dir = config.get('output', {}).get('reasoning_dir', './reasoning_visualizations')
    return RobotReasoningVisualizer(save_dir=save_dir, use_wandb=use_wandb)
