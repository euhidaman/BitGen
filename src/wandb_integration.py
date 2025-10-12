"""
Enhanced WandB Integration for BitGen 2-Stage Training
Comprehensive metrics tracking and visualizations for 'babylm-ntust' team
Stage 1: Vision-Language Pre-training (FIBER + Larima)
Stage 2: Reasoning Module Training (Tiny-R1 + Robot Selection)
"""

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime
import psutil
import time

class WandBIntegration:
    """Enhanced WandB integration for BitGen training and inference monitoring"""

    def __init__(self,
                 project_name: str = "bitgen-training",
                 entity: str = "babylm-ntust",
                 run_name: Optional[str] = None,
                 config: Dict = None,
                 tags: List[str] = None,
                 stage: str = "stage1"):
        """
        Initialize WandB integration

        Args:
            project_name: WandB project name
            entity: WandB team/entity name ('babylm-ntust')
            run_name: Optional run name
            config: Model configuration dict
            tags: List of tags for the run
            stage: Training stage ('stage1' or 'stage2')
        """

        self.entity = entity
        self.project_name = project_name
        self.stage = stage
        self.logger = logging.getLogger(__name__)

        # Stage-specific tags
        stage_tags = {
            "stage1": ["vision-language", "fiber", "larima-gpm", "contrastive"],
            "stage2": ["reasoning", "tiny-r1", "robot-selection", "grpo"]
        }

        # Initialize WandB run
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            name=run_name or f"bitgen-{stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config or {},
            tags=(tags or []) + ["bitgen", "2-stage", stage] + stage_tags.get(stage, []),
            reinit=True
        )

        # Tracking variables
        self.step = 0
        self.epoch = 0
        self.best_metrics = {}
        self.metric_history = {}

        self.logger.info(f"Initialized WandB run: {self.run.name} in {entity}/{project_name} ({stage})")

    def log_training_metrics(self,
                           metrics: Dict,
                           step: Optional[int] = None,
                           epoch: Optional[int] = None):
        """
        Log training metrics to WandB

        Args:
            metrics: Dictionary of metrics to log
            step: Training step (optional)
            epoch: Training epoch (optional)
        """

        if step is not None:
            self.step = step
        if epoch is not None:
            self.epoch = epoch

        # Organize metrics by category
        organized_metrics = self._organize_metrics(metrics)

        # Log to WandB
        wandb.log(organized_metrics, step=self.step)

        # Update metric history
        for key, value in organized_metrics.items():
            if key not in self.metric_history:
                self.metric_history[key] = []
            self.metric_history[key].append(value)

        # Update best metrics
        self._update_best_metrics(organized_metrics)

    def log_model_architecture(self, model, config):
        """Log model architecture and complexity"""

        # Calculate model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Model size estimation
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming fp32

        # Log architecture metrics
        arch_metrics = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/model_size_mb": model_size_mb,
            "model/embed_dim": getattr(config, 'embed_dim', 0),
            "model/num_layers": getattr(config, 'num_layers', 0),
            "model/num_heads": getattr(config, 'num_heads', 0),
            "model/vocab_size": getattr(config, 'vocab_size', 0),
            "model/max_seq_len": getattr(config, 'max_seq_len', 0),
            "model/memory_size": getattr(config, 'memory_size', 0),
            "model/quantization_bits": getattr(config, 'quantization_bits', 32),
        }

        wandb.log(arch_metrics)

        # Create architecture visualization
        self._create_architecture_visualization(model, config)

    def log_system_metrics(self):
        """Log system performance metrics"""

        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        system_metrics = {
            "system/cpu_usage_percent": cpu_percent,
            "system/memory_usage_percent": memory.percent,
            "system/memory_available_gb": memory.available / (1024**3),
            "system/disk_usage_percent": disk.percent,
            "system/disk_free_gb": disk.free / (1024**3),
        }

        # GPU metrics if available
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                system_metrics.update({
                    "system/gpu_memory_allocated_gb": gpu_memory.get('allocated_bytes.all.current', 0) / (1024**3),
                    "system/gpu_memory_cached_gb": gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3),
                    "system/gpu_utilization_percent": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0,
                })
        except Exception as e:
            # Log warning but don't crash on GPU metrics collection failure
            pass

        wandb.log(system_metrics, step=self.step)

    def log_flops_metrics(self, flops_info: Dict):
        """Log FLOPS-related metrics"""

        flops_metrics = {
            "performance/model_flops": flops_info.get('flops', 0),
            "performance/model_params": flops_info.get('params', 0),
            "performance/training_flops_total": flops_info.get('training_flops_total', 0),
            "performance/flops_per_step": flops_info.get('flops_per_step', 0),
        }

        # Calculate efficiency metrics
        if flops_info.get('training_time_seconds', 0) > 0:
            flops_metrics["performance/flops_per_second"] = (
                flops_info.get('training_flops_total', 0) / flops_info.get('training_time_seconds', 1)
            )

        wandb.log(flops_metrics, step=self.step)

    def log_energy_metrics(self, energy_info: Dict):
        """Log energy consumption and carbon footprint metrics"""

        energy_metrics = {
            "sustainability/energy_consumed_kwh": energy_info.get('energy_consumed_kwh', 0),
            "sustainability/carbon_emissions_kg": energy_info.get('carbon_emissions_kg', 0),
            "sustainability/power_consumption_watts": energy_info.get('power_consumption_watts', 0),
        }

        # Calculate efficiency metrics
        if energy_info.get('tokens_generated', 0) > 0:
            energy_metrics.update({
                "sustainability/energy_per_token_mj": energy_info.get('energy_consumed_kwh', 0) * 3.6e6 / energy_info.get('tokens_generated', 1),
                "sustainability/carbon_per_token_mg": energy_info.get('carbon_emissions_kg', 0) * 1e6 / energy_info.get('tokens_generated', 1),
            })

        wandb.log(energy_metrics, step=self.step)

    def log_inference_metrics(self, inference_results: List[Dict]):
        """Log inference performance metrics"""

        if not inference_results:
            return

        # Calculate statistics
        throughput_values = [r.get('tokens_per_second', 0) for r in inference_results]
        latency_values = [r.get('latency_ms_per_token', 0) for r in inference_results]
        memory_values = [r.get('memory_peak_mb', 0) for r in inference_results]
        power_values = [r.get('estimated_power_mw', 0) for r in inference_results]
        temp_values = [r.get('cpu_temp_post_c', 0) for r in inference_results if r.get('cpu_temp_post_c', 0) > 0]

        inference_metrics = {
            "inference/avg_throughput_tokens_per_sec": np.mean(throughput_values) if throughput_values else 0,
            "inference/median_throughput_tokens_per_sec": np.median(throughput_values) if throughput_values else 0,
            "inference/p95_throughput_tokens_per_sec": np.percentile(throughput_values, 95) if throughput_values else 0,

            "inference/avg_latency_ms_per_token": np.mean(latency_values) if latency_values else 0,
            "inference/median_latency_ms_per_token": np.median(latency_values) if latency_values else 0,
            "inference/p95_latency_ms_per_token": np.percentile(latency_values, 95) if latency_values else 0,

            "inference/avg_memory_usage_mb": np.mean(memory_values) if memory_values else 0,
            "inference/peak_memory_usage_mb": np.max(memory_values) if memory_values else 0,

            "inference/avg_power_consumption_mw": np.mean(power_values) if power_values else 0,
            "inference/peak_power_consumption_mw": np.max(power_values) if power_values else 0,
        }

        if temp_values:
            inference_metrics.update({
                "inference/avg_temperature_c": np.mean(temp_values),
                "inference/peak_temperature_c": np.max(temp_values),
            })

        wandb.log(inference_metrics, step=self.step)

        # Create inference performance visualizations
        self._create_inference_visualizations(inference_results)

    def create_training_visualizations(self):
        """Create comprehensive training visualizations"""

        # Loss curves
        self._create_loss_curves()

        # Performance metrics over time
        self._create_performance_dashboard()

        # Model comparison charts
        self._create_model_comparison_charts()

        # System resource utilization
        self._create_system_utilization_dashboard()

    def log_epoch_summary(self, epoch: int, epoch_metrics: Dict, model_url: str = None):
        """Log comprehensive epoch summary"""

        self.epoch = epoch

        # Create epoch summary metrics
        summary_metrics = {
            "epoch": epoch,
            "epoch/training_loss": epoch_metrics.get('avg_loss', 0),
            "epoch/learning_rate": epoch_metrics.get('learning_rate', 0),
            "epoch/tokens_per_second": epoch_metrics.get('avg_tokens_per_second', 0),
        }

        # Add model URL if provided (HuggingFace)
        if model_url:
            summary_metrics["epoch/model_url"] = model_url

        wandb.log(summary_metrics, step=self.step)

        # Create epoch-specific visualizations
        self._create_epoch_visualizations(epoch, epoch_metrics)

        # Log model artifacts
        self._log_model_artifacts(epoch)

    def _organize_metrics(self, metrics: Dict) -> Dict:
        """Organize metrics into categories for better WandB visualization"""

        organized = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                # Categorize metrics
                if any(keyword in key.lower() for keyword in ['loss', 'error']):
                    organized[f"loss/{key}"] = float(value)
                elif any(keyword in key.lower() for keyword in ['accuracy', 'score', 'f1', 'bleu']):
                    organized[f"metrics/{key}"] = float(value)
                elif any(keyword in key.lower() for keyword in ['lr', 'learning_rate']):
                    organized[f"optimization/{key}"] = float(value)
                elif any(keyword in key.lower() for keyword in ['token', 'throughput', 'speed']):
                    organized[f"performance/{key}"] = float(value)
                elif any(keyword in key.lower() for keyword in ['memory', 'ram', 'cpu', 'gpu']):
                    organized[f"system/{key}"] = float(value)
                elif any(keyword in key.lower() for keyword in ['power', 'energy', 'carbon', 'temp']):
                    organized[f"sustainability/{key}"] = float(value)
                else:
                    organized[f"training/{key}"] = float(value)
            else:
                organized[key] = value

        return organized

    def _update_best_metrics(self, metrics: Dict):
        """Update best metrics tracking"""

        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                if 'loss' in key.lower() or 'error' in key.lower():
                    # Lower is better
                    if key not in self.best_metrics or value < self.best_metrics[key]:
                        self.best_metrics[key] = float(value)
                        wandb.log({f"best/{key}": float(value)}, step=self.step)
                elif 'accuracy' in key.lower() or 'score' in key.lower() or 'throughput' in key.lower():
                    # Higher is better
                    if key not in self.best_metrics or value > self.best_metrics[key]:
                        self.best_metrics[key] = float(value)
                        wandb.log({f"best/{key}": float(value)}, step=self.step)

    def _create_architecture_visualization(self, model, config):
        """Create model architecture visualization"""

        # Create architecture diagram data
        layers_info = []
        total_params = 0

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layers_info.append({
                        'name': name,
                        'type': type(module).__name__,
                        'parameters': params
                    })
                    total_params += params

        # Create visualization
        fig = go.Figure()

        names = [layer['name'][:30] + '...' if len(layer['name']) > 30 else layer['name']
                for layer in layers_info[:20]]  # Top 20 layers
        params = [layer['parameters'] for layer in layers_info[:20]]
        types = [layer['type'] for layer in layers_info[:20]]

        fig.add_trace(go.Bar(
            x=names,
            y=params,
            text=types,
            textposition='auto',
            name='Parameters per Layer'
        ))

        fig.update_layout(
            title='Model Architecture - Parameters per Layer',
            xaxis_title='Layer Name',
            yaxis_title='Parameters',
            xaxis_tickangle=-45,
            height=600
        )

        wandb.log({"architecture/parameters_per_layer": wandb.Plotly(fig)})

    def _create_loss_curves(self):
        """Create comprehensive loss curve visualizations"""

        if not self.metric_history:
            return

        # Find all loss metrics
        loss_metrics = {k: v for k, v in self.metric_history.items() if 'loss' in k.lower()}

        if not loss_metrics:
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Loss Components', 'Loss Trends', 'Loss Distribution'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Main loss curve
        if 'loss/total_loss' in loss_metrics:
            steps = list(range(len(loss_metrics['loss/total_loss'])))
            fig.add_trace(
                go.Scatter(x=steps, y=loss_metrics['loss/total_loss'],
                          name='Total Loss', line=dict(width=3)),
                row=1, col=1
            )

        # Loss components
        for i, (name, values) in enumerate(loss_metrics.items()):
            if name != 'loss/total_loss' and len(values) > 0:
                fig.add_trace(
                    go.Scatter(x=list(range(len(values))), y=values,
                              name=name.replace('loss/', ''), opacity=0.7),
                    row=1, col=2
                )

        fig.update_layout(height=800, title_text="Training Loss Analysis")
        wandb.log({"training/loss_analysis": wandb.Plotly(fig)})

    def _create_performance_dashboard(self):
        """Create performance metrics dashboard"""

        perf_metrics = {k: v for k, v in self.metric_history.items()
                       if 'performance' in k.lower() or 'throughput' in k.lower()}

        if not perf_metrics:
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Throughput Over Time', 'Memory Usage', 'System Metrics', 'Efficiency Trends')
        )

        # Plot performance metrics
        for name, values in perf_metrics.items():
            if len(values) > 0:
                steps = list(range(len(values)))
                fig.add_trace(
                    go.Scatter(x=steps, y=values, name=name.replace('performance/', ''),
                              mode='lines+markers'),
                    row=1, col=1
                )

        fig.update_layout(height=800, title_text="Performance Dashboard")
        wandb.log({"training/performance_dashboard": wandb.Plotly(fig)})

    def _create_inference_visualizations(self, inference_results: List[Dict]):
        """Create inference performance visualizations"""

        if not inference_results:
            return

        # Extract metrics
        throughput = [r.get('tokens_per_second', 0) for r in inference_results]
        latency = [r.get('latency_ms_per_token', 0) for r in inference_results]
        memory = [r.get('memory_peak_mb', 0) for r in inference_results]
        power = [r.get('estimated_power_mw', 0) for r in inference_results]

        # Create comprehensive inference dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Throughput Distribution', 'Latency vs Memory', 'Power Consumption', 'Performance Correlation'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )

        # Throughput histogram
        fig.add_trace(
            go.Histogram(x=throughput, name='Throughput', nbinsx=20, opacity=0.7),
            row=1, col=1
        )

        # Latency vs Memory scatter
        fig.add_trace(
            go.Scatter(x=latency, y=memory, mode='markers',
                      name='Latency vs Memory', marker=dict(size=8)),
            row=1, col=2
        )

        # Power consumption over time
        fig.add_trace(
            go.Scatter(x=list(range(len(power))), y=power,
                      name='Power Consumption', line=dict(width=2)),
            row=2, col=1
        )

        fig.update_layout(height=800, title_text="Inference Performance Analysis")
        wandb.log({"inference/performance_analysis": wandb.Plotly(fig)})

    def _create_epoch_visualizations(self, epoch: int, epoch_metrics: Dict):
        """Create epoch-specific visualizations"""

        # Epoch summary card
        summary_html = f"""
        <div style="padding: 20px; border: 1px solid #ccc; border-radius: 10px;">
        <h2>Epoch {epoch} Summary</h2>
        <p><strong>Training Loss:</strong> {epoch_metrics.get('avg_loss', 'N/A'):.4f}</p>
        <p><strong>Learning Rate:</strong> {epoch_metrics.get('learning_rate', 'N/A')}</p>
        <p><strong>Throughput:</strong> {epoch_metrics.get('avg_tokens_per_second', 'N/A'):.2f} tokens/sec</p>
        <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """

        wandb.log({f"epoch_summaries/epoch_{epoch}": wandb.Html(summary_html)})

    def _create_system_utilization_dashboard(self):
        """Create system utilization dashboard"""

        system_metrics = {k: v for k, v in self.metric_history.items() if 'system' in k.lower()}

        if not system_metrics:
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'GPU Utilization', 'Disk Usage')
        )

        for name, values in system_metrics.items():
            if len(values) > 0 and isinstance(values[0], (int, float)):
                steps = list(range(len(values)))
                if 'cpu' in name.lower():
                    row, col = 1, 1
                elif 'memory' in name.lower():
                    row, col = 1, 2
                elif 'gpu' in name.lower():
                    row, col = 2, 1
                else:
                    row, col = 2, 2

                fig.add_trace(
                    go.Scatter(x=steps, y=values, name=name.replace('system/', ''),
                              mode='lines'),
                    row=row, col=col
                )

        fig.update_layout(height=800, title_text="System Utilization Dashboard")
        wandb.log({"system/utilization_dashboard": wandb.Plotly(fig)})

    def _create_model_comparison_charts(self):
        """Create model comparison charts"""

        # Best metrics summary
        if self.best_metrics:
            metrics_names = list(self.best_metrics.keys())[:10]  # Top 10 metrics
            metrics_values = [self.best_metrics[name] for name in metrics_names]

            fig = go.Figure(data=[
                go.Bar(x=metrics_names, y=metrics_values, text=metrics_values, textposition='auto')
            ])

            fig.update_layout(
                title='Best Metrics Achieved',
                xaxis_title='Metric',
                yaxis_title='Value',
                xaxis_tickangle=-45
            )

            wandb.log({"summary/best_metrics": wandb.Plotly(fig)})

    def _log_model_artifacts(self, epoch: int):
        """Log model artifacts and checkpoints"""

        # Create metadata
        metadata = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "best_metrics": self.best_metrics,
            "total_steps": self.step
        }

        # Save metadata as artifact
        with open(f"epoch_{epoch}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
        artifact.add_file(f"epoch_{epoch}_metadata.json")
        wandb.log_artifact(artifact)

    def finish_run(self, summary_metrics: Dict = None):
        """Finish WandB run with final summary"""

        if summary_metrics:
            wandb.log(summary_metrics)

        # Log final best metrics
        if self.best_metrics:
            for key, value in self.best_metrics.items():
                wandb.run.summary[f"best_{key}"] = value

        # Log run summary
        wandb.run.summary["total_steps"] = self.step
        wandb.run.summary["total_epochs"] = self.epoch
        wandb.run.summary["run_duration_minutes"] = (
            (datetime.now() - wandb.run.start_time).total_seconds() / 60
        )

        self.logger.info(f"Finishing WandB run: {self.run.name}")
        wandb.finish()

    def log_stage1_metrics(self, 
                          epoch: int,
                          loss: float,
                          contrastive_loss: float,
                          memory_kl_loss: float,
                          acc_t2i: float,
                          acc_i2t: float,
                          lr: float):
        """
        Log Stage 1 (Vision-Language) specific metrics
        
        Args:
            epoch: Current epoch
            loss: Total loss
            contrastive_loss: Contrastive learning loss
            memory_kl_loss: Larima GPM KL divergence loss
            acc_t2i: Text-to-image accuracy
            acc_i2t: Image-to-text accuracy
            lr: Learning rate
        """
        
        metrics = {
            "stage1/epoch": epoch,
            "stage1/loss/total": loss,
            "stage1/loss/contrastive": contrastive_loss,
            "stage1/loss/memory_kl": memory_kl_loss,
            "stage1/accuracy/text_to_image": acc_t2i,
            "stage1/accuracy/image_to_text": acc_i2t,
            "stage1/accuracy/average": (acc_t2i + acc_i2t) / 2.0,
            "stage1/learning_rate": lr
        }
        
        wandb.log(metrics, step=self.step)
        
        # Update best metrics
        if acc_t2i + acc_i2t > self.best_metrics.get('best_stage1_accuracy', 0):
            self.best_metrics['best_stage1_accuracy'] = acc_t2i + acc_i2t
            self.best_metrics['best_stage1_epoch'] = epoch
    
    def log_stage2_metrics(self,
                          epoch: int,
                          loss: float,
                          robot_loss: float,
                          correctness_reward: float,
                          reasoning_reward: float,
                          accuracy: float,
                          lr: float):
        """
        Log Stage 2 (Reasoning) specific metrics
        
        Args:
            epoch: Current epoch
            loss: Total loss
            robot_loss: Robot selection loss
            correctness_reward: Correctness reward (GRPO)
            reasoning_reward: Reasoning trace reward
            accuracy: Robot selection accuracy
            lr: Learning rate
        """
        
        metrics = {
            "stage2/epoch": epoch,
            "stage2/loss/total": loss,
            "stage2/loss/robot_selection": robot_loss,
            "stage2/reward/correctness": correctness_reward,
            "stage2/reward/reasoning_trace": reasoning_reward,
            "stage2/reward/total": correctness_reward + reasoning_reward,
            "stage2/accuracy/robot_selection": accuracy,
            "stage2/learning_rate": lr
        }
        
        wandb.log(metrics, step=self.step)
        
        # Update best metrics
        if accuracy > self.best_metrics.get('best_stage2_accuracy', 0):
            self.best_metrics['best_stage2_accuracy'] = accuracy
            self.best_metrics['best_stage2_epoch'] = epoch
    
    def log_contrastive_visualization(self,
                                     text_features: torch.Tensor,
                                     image_features: torch.Tensor,
                                     epoch: int):
        """
        Create and log contrastive learning visualization
        
        Args:
            text_features: Text embeddings [B, D]
            image_features: Image embeddings [B, D]
            epoch: Current epoch
        """
        
        # Create similarity matrix
        similarity = torch.matmul(text_features, image_features.T)
        similarity_np = similarity.detach().cpu().numpy()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(similarity_np, cmap='RdYlGn', center=0, ax=ax)
        ax.set_title(f'Text-Image Similarity Matrix (Epoch {epoch})')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Text Index')
        
        wandb.log({f"stage1/contrastive_matrix_epoch{epoch}": wandb.Image(fig)}, step=self.step)
        plt.close(fig)
    
    def log_robot_confusion_matrix(self,
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   robot_names: List[str],
                                   epoch: int):
        """
        Log robot selection confusion matrix
        
        Args:
            predictions: Predicted labels [B, num_robots]
            targets: Target labels [B, num_robots]
            robot_names: List of robot names
            epoch: Current epoch
        """
        
        # Convert to binary predictions
        pred_binary = (predictions > 0.5).astype(int)
        
        # Create confusion matrix for each robot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, robot_name in enumerate(robot_names):
            if i < len(axes):
                # Binary confusion matrix for this robot
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(targets[:, i], pred_binary[:, i])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{robot_name}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Remove extra subplot
        if len(robot_names) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        wandb.log({f"stage2/confusion_matrix_epoch{epoch}": wandb.Image(fig)}, step=self.step)
        plt.close(fig)
    
    def log_memory_visualization(self,
                                memory_mean: torch.Tensor,
                                memory_read_count: torch.Tensor,
                                epoch: int):
        """
        Visualize Larima GPM memory state
        
        Args:
            memory_mean: Memory mean tensor [memory_size, code_size]
            memory_read_count: Read count per memory slot [memory_size]
            epoch: Current epoch
        """
        
        memory_np = memory_mean.detach().cpu().numpy()
        read_count_np = memory_read_count.detach().cpu().numpy()
        
        # Create memory heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Memory content heatmap
        sns.heatmap(memory_np.T, cmap='viridis', ax=ax1)
        ax1.set_title(f'Larima GPM Memory Content (Epoch {epoch})')
        ax1.set_xlabel('Memory Slot')
        ax1.set_ylabel('Embedding Dimension')
        
        # Memory usage bar chart
        ax2.bar(range(len(read_count_np)), read_count_np)
        ax2.set_title(f'Memory Slot Read Frequency (Epoch {epoch})')
        ax2.set_xlabel('Memory Slot')
        ax2.set_ylabel('Read Count')
        
        plt.tight_layout()
        wandb.log({f"stage1/memory_state_epoch{epoch}": wandb.Image(fig)}, step=self.step)
        plt.close(fig)


def setup_wandb_integration(project_name: str = "bitgen-training",
                           entity: str = "babylm-ntust",
                           run_name: Optional[str] = None,
                           config: Dict = None,
                           tags: List[str] = None,
                           stage: str = "stage1") -> WandBIntegration:
    """
    Setup WandB integration for BitGen 2-stage training

    Args:
        project_name: WandB project name
        entity: WandB team/entity ('babylm-ntust')
        run_name: Optional run name
        config: Model configuration dict
        tags: List of tags for the run
        stage: Training stage ('stage1' or 'stage2')

    Returns:
        WandBIntegration instance
    """

    return WandBIntegration(
        project_name=project_name,
        entity=entity,
        run_name=run_name,
        config=config,
        tags=tags,
        stage=stage
    )
