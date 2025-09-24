"""
Enhanced BitGen Training Script with Comprehensive Monitoring
Tracks FLOPS, energy consumption, carbon emissions for Raspberry Pi Zero deployment
WITH HuggingFace Hub integration and enhanced WandB tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict

# Import BitGen components
from bitgen_model import BitGenModel, BitGenConfig, create_bitgen_model
from adaptive_loss import BitGenLoss, AdaptiveLossManager, PerformanceTracker
from data_loader import COCODataset, BitGenDataLoader, RobotSelectionDataset

# Import monitoring
from raspberry_pi.rpi_monitor import RaspberryPiMonitor, get_monitor, start_monitoring, stop_monitoring

# Import new integrations
from huggingface_integration import HuggingFaceIntegration, setup_huggingface_integration
from wandb_integration import WandBIntegration, setup_wandb_integration
from advanced_metrics import create_advanced_metrics_logger

# Additional imports for comprehensive monitoring
try:
    from codecarbon import OfflineEmissionsTracker, track_emissions
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

try:
    from ptflops import get_model_complexity_info
    from thop import profile, clever_format
    FLOPS_AVAILABLE = True
except ImportError:
    FLOPS_AVAILABLE = False

class EnhancedBitGenTrainer:
    """BitGen trainer with comprehensive monitoring, HuggingFace, and WandB integration"""

    def __init__(self,
                 config: BitGenConfig,
                 model_size: str = 'tiny',
                 output_dir: str = 'checkpoints',
                 monitoring_dir: str = 'monitoring_results',
                 use_carbon_tracking: bool = True,
                 # HuggingFace integration
                 hf_repo_name: Optional[str] = None,
                 hf_organization: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 hf_private: bool = False,
                 push_to_hub: bool = True,
                 # WandB integration
                 wandb_project: str = "bitgen-training",
                 wandb_entity: str = "babylm-ntust",
                 wandb_run_name: Optional[str] = None,
                 wandb_tags: List[str] = None):

        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize monitoring
        self.monitor = RaspberryPiMonitor(monitoring_dir)
        self.use_carbon_tracking = use_carbon_tracking and CODECARBON_AVAILABLE

        # Initialize model
        self.model = create_bitgen_model(model_size)

        # Multi-GPU setup - detect and use all available GPUs for training
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_multi_gpu = self.num_gpus > 1

        if torch.cuda.is_available():
            print(f"🚀 CUDA available with {self.num_gpus} GPU(s) for training")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

        # Setup device - use GPU for training, CPU only for Pi Zero inference
        if self.use_multi_gpu:
            print(f"✅ Enabling DataParallel training across {self.num_gpus} GPUs")
            self.device = torch.device('cuda:0')
            self.model = self.model.to(self.device)
            # Wrap model with DataParallel for multi-GPU training
            self.model = nn.DataParallel(self.model)
            self.effective_batch_size_multiplier = self.num_gpus
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            self.effective_batch_size_multiplier = 1
            print("✅ Using single GPU training")
        else:
            self.device = torch.device('cpu')  # Fallback to CPU if no GPU
            self.model = self.model.to(self.device)
            self.effective_batch_size_multiplier = 1
            print("⚠️ Using CPU training (no CUDA available)")

        # Calculate model FLOPS
        self.model_flops = self._calculate_model_flops()

        # Initialize loss function
        self.loss_fn = BitGenLoss(config, config.vocab_size)

        # Performance tracking
        self.performance_tracker = PerformanceTracker()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Initialize HuggingFace integration
        self.hf_integration = None
        self.push_to_hub = push_to_hub
        if push_to_hub and hf_repo_name:
            try:
                self.hf_integration = setup_huggingface_integration(
                    model_name=hf_repo_name,
                    organization=hf_organization,
                    token=hf_token,
                    private=hf_private
                )
                self.logger.info(f"✅ HuggingFace integration initialized: {self.hf_integration.repo_id}")
            except Exception as e:
                self.logger.warning(f"Failed to setup HuggingFace integration: {e}")
                self.push_to_hub = False

        # Initialize WandB integration
        wandb_config = {
            **config.__dict__,
            'model_size': model_size,
            'flops_info': self.model_flops,
            'hf_repo_id': self.hf_integration.repo_id if self.hf_integration else None,
            'push_to_hub': push_to_hub
        }

        self.wandb_integration = setup_wandb_integration(
            project_name=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name or f"bitgen-{model_size}-{time.strftime('%Y%m%d-%H%M%S')}",
            config=wandb_config,
            tags=wandb_tags or ["bitgen", "multimodal", "embedded", "quantized", model_size]
        )

        # Initialize advanced metrics logger
        self.advanced_metrics = create_advanced_metrics_logger(
            wandb_integration=self.wandb_integration,
            config=config,
            output_dir=str(Path(monitoring_dir) / "advanced_metrics")
        )

        # Carbon tracking
        self.carbon_tracker = None
        if self.use_carbon_tracking:
            self.carbon_tracker = OfflineEmissionsTracker(
                country_iso_code="US",
                output_dir=str(self.monitor.output_dir),
                project_name="BitGen-Training"
            )

        # FLOPS tracking
        self.training_flops = 0
        self.inference_flops = 0

        # Epoch-level data collection for matrices
        self.epoch_robot_predictions = []
        self.epoch_robot_targets = []
        self.epoch_task_descriptions = []

        # Setup logging
        self.setup_logging()

        # Log model architecture to WandB
        self.wandb_integration.log_model_architecture(self.model, self.config)

    def setup_logging(self):
        """Setup logging for training with monitoring"""
        log_file = self.output_dir / 'enhanced_training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Log system information
        system_info = self.monitor.get_system_info()
        self.logger.info("System Information:")
        for category, details in system_info.items():
            self.logger.info(f"  {category}: {details}")

    def _calculate_model_flops(self) -> Dict:
        """Calculate model FLOPS for different operations"""
        if not FLOPS_AVAILABLE:
            self.logger.warning("FLOPS calculation tools not available")
            return {"forward_flops": 0, "params": 0}

        try:
            # Calculate FLOPS for forward pass
            input_shape = (self.config.max_seq_len,)  # Sequence length
            flops_info = self.monitor.calculate_model_flops(self.model, input_shape)

            self.logger.info(f"Model FLOPS Analysis:")
            self.logger.info(f"  Forward pass FLOPS: {flops_info.get('flops_human', 'Unknown')}")
            self.logger.info(f"  Parameters: {flops_info.get('params_human', 'Unknown')}")

            return flops_info

        except Exception as e:
            self.logger.error(f"FLOPS calculation failed: {e}")
            return {"forward_flops": 0, "params": 0}

    def setup_optimizer(self, learning_rate: float = 1e-4):
        """Setup optimizer for Raspberry Pi Zero constraints"""
        # Use Adam with lower memory footprint for Pi Zero
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,  # Reduced for smaller model
            eps=1e-8
        )

        # Simple learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=0.95
        )

        return optimizer, scheduler

    def train_step_with_monitoring(self, batch: Dict, optimizer: optim.Optimizer) -> Dict:
        """Training step with comprehensive monitoring including advanced metrics"""
        step_start_time = time.time()

        self.model.train()

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        images = batch.get('images')
        if images is not None:
            images = images.to(self.device)

        target_robot = batch.get('target_robot')
        if target_robot is not None:
            target_robot = target_robot.to(self.device)

        # Forward pass with FLOPS tracking and analysis data
        forward_start = time.time()

        outputs = self.model(
            input_ids=input_ids,
            images=images,
            return_robot_selection=(target_robot is not None),
            return_analysis_data=True  # Enable analysis data collection
        )

        forward_time = time.time() - forward_start

        # Compute loss
        total_loss, loss_dict = self.loss_fn(
            outputs,
            labels,
            images=images,
            target_robot=target_robot
        )

        # Backward pass
        backward_start = time.time()
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        optimizer.step()
        backward_time = time.time() - backward_start

        step_time = time.time() - step_start_time

        # Calculate FLOPS for this step
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Estimate FLOPS (forward + backward ≈ 3x forward)
        step_flops = self.model_flops.get('flops', 0) * batch_size * 3
        self.training_flops += step_flops

        # Log advanced metrics (episodic memory, attention, reasoning)
        if self.advanced_metrics and 'memory_attention' in outputs:
            self.advanced_metrics.log_step_metrics(
                model_outputs=outputs,
                batch_data=batch,
                step=self.global_step
            )

        # Collect epoch-level data for reasoning matrices
        if target_robot is not None and 'robot_selection' in outputs and outputs['robot_selection'] is not None:
            predicted_robots = outputs['robot_selection'].argmax(dim=-1)

            self.epoch_robot_predictions.extend(predicted_robots.cpu().numpy().tolist())
            self.epoch_robot_targets.extend(target_robot.cpu().numpy().tolist())

            # Collect task descriptions if available
            task_descriptions = batch.get('task_description', [''] * batch_size)
            self.epoch_task_descriptions.extend(task_descriptions)

        # Update metrics
        metrics = {
            'total_loss': total_loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'forward_time_ms': forward_time * 1000,
            'backward_time_ms': backward_time * 1000,
            'step_time_ms': step_time * 1000,
            'step_flops': step_flops,
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'tokens_per_second': (batch_size * seq_len) / step_time
        }

        # Add individual loss components
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            elif isinstance(value, dict):
                for k, v in value.items():
                    metrics[f"{key}_{k}"] = v if not isinstance(v, torch.Tensor) else v.item()

        return metrics

    def train_with_comprehensive_monitoring(self,
                                           coco_data_path: str,
                                           robot_data_path: Optional[str] = None,
                                           num_epochs: int = 5,  # Reduced for Pi Zero
                                           batch_size: int = 1,  # Small batch for Pi Zero
                                           learning_rate: float = 5e-5,  # Lower LR for stability
                                           eval_steps: int = 50,
                                           save_steps: int = 100):
        """Main training loop with comprehensive monitoring"""

        self.logger.info("Starting BitGen training with comprehensive monitoring for Raspberry Pi Zero")

        # Start monitoring
        self.monitor.start_monitoring()

        # Start carbon tracking
        if self.carbon_tracker:
            self.carbon_tracker.start()

        try:
            # Setup training components
            optimizer, scheduler = self.setup_optimizer(learning_rate)

            # Setup data loaders (small batches for Pi Zero)
            train_loader = self._setup_data_loader(coco_data_path, robot_data_path, batch_size)

            # Training metrics
            training_start_time = time.time()
            total_steps = 0

            # Training loop
            for epoch in range(num_epochs):
                self.epoch = epoch
                epoch_start_time = time.time()
                epoch_metrics = defaultdict(list)

                self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

                for step, batch in enumerate(train_loader):
                    # Training step with monitoring
                    step_metrics = self.train_step_with_monitoring(batch, optimizer)

                    # Update metrics
                    for key, value in step_metrics.items():
                        epoch_metrics[key].append(value)

                    # Update performance tracker
                    self.performance_tracker.update(step_metrics)

                    self.global_step += 1
                    total_steps += 1

                    # Logging
                    if step % 10 == 0:
                        current_metrics = self._get_current_monitoring_state()
                        self.logger.info(
                            f"Epoch {epoch+1}, Step {step}: "
                            f"Loss={step_metrics['total_loss']:.4f}, "
                            f"Tokens/s={step_metrics['tokens_per_second']:.2f}, "
                            f"RAM={current_metrics.get('ram_usage_mb', 0):.1f}MB, "
                            f"CPU={current_metrics.get('cpu_usage_percent', 0):.1f}%, "
                            f"Temp={current_metrics.get('cpu_temperature_c', 0):.1f}°C, "
                            f"Power={current_metrics.get('power_consumption_mw', 0):.1f}mW"
                        )

                    # Evaluation
                    if self.global_step % eval_steps == 0:
                        eval_results = self._evaluate_with_monitoring(train_loader, num_samples=5)
                        self.logger.info(f"Evaluation at step {self.global_step}: {eval_results}")

                    # Checkpointing
                    if self.global_step % save_steps == 0:
                        self._save_checkpoint_with_metrics(optimizer, scheduler)

                    # Early stopping if overheating (Pi Zero protection)
                    current_temp = self._get_current_temperature()
                    if current_temp > 75.0:  # Pi Zero thermal limit
                        self.logger.warning(f"High temperature detected: {current_temp}°C. Pausing training.")
                        time.sleep(30)  # Cool down pause

                # End of epoch processing with HuggingFace and WandB integration
                scheduler.step()
                epoch_time = time.time() - epoch_start_time

                # Calculate epoch averages
                epoch_avg = {key: np.mean(values) for key, values in epoch_metrics.items()}

                # Log epoch metrics to WandB
                epoch_wandb_metrics = {
                    'epoch/training_loss': epoch_avg['total_loss'],
                    'epoch/learning_rate': epoch_avg['learning_rate'],
                    'epoch/tokens_per_second': epoch_avg['tokens_per_second'],
                    'epoch/duration_seconds': epoch_time,
                    'epoch/step_time_ms': epoch_avg['step_time_ms'],
                    'epoch/forward_time_ms': epoch_avg['forward_time_ms'],
                    'epoch/backward_time_ms': epoch_avg['backward_time_ms'],
                }

                # Add FLOPS metrics
                if self.model_flops:
                    epoch_wandb_metrics.update({
                        'flops/cumulative_training_flops': self.training_flops,
                        'flops/epoch_flops': sum(epoch_metrics['step_flops']),
                        'flops/avg_flops_per_step': np.mean(epoch_metrics['step_flops']),
                    })

                # Log system metrics to WandB
                self.wandb_integration.log_system_metrics()

                # Log training metrics
                self.wandb_integration.log_training_metrics(epoch_wandb_metrics, step=self.global_step, epoch=epoch)

                # Log FLOPS metrics
                if self.model_flops:
                    flops_metrics = {
                        'flops': self.model_flops.get('flops', 0),
                        'params': self.model_flops.get('params', 0),
                        'training_flops_total': self.training_flops,
                        'flops_per_step': np.mean(epoch_metrics['step_flops']) if epoch_metrics['step_flops'] else 0,
                        'training_time_seconds': sum(epoch_metrics['step_time_ms']) / 1000,
                    }
                    self.wandb_integration.log_flops_metrics(flops_metrics)

                # Log energy metrics if available
                if self.carbon_tracker:
                    try:
                        # Try to get current energy data
                        energy_metrics = {
                            'energy_consumed_kwh': 0,  # Would be updated by CodeCarbon
                            'carbon_emissions_kg': 0,  # Would be updated by CodeCarbon
                            'power_consumption_watts': epoch_avg.get('power_consumption_mw', 0) / 1000,
                            'tokens_generated': sum(epoch_metrics['batch_size']) * epoch_avg.get('sequence_length', 0),
                        }
                        self.wandb_integration.log_energy_metrics(energy_metrics)
                    except Exception as e:
                        self.logger.warning(f"Failed to log energy metrics: {e}")

                # Push model to HuggingFace Hub after each epoch
                model_url = None
                if self.push_to_hub and self.hf_integration:
                    try:
                        self.logger.info(f"🚀 Pushing model to HuggingFace Hub for epoch {epoch+1}...")

                        # Create tokenizer for HF (simplified)
                        tokenizer_info = {
                            'vocab_size': self.config.vocab_size,
                            'max_length': self.config.max_seq_len
                        }

                        # Prepare metrics for model card
                        hf_metrics = {
                            'epoch': epoch + 1,
                            'training_loss': epoch_avg['total_loss'],
                            'tokens_per_second': epoch_avg['tokens_per_second'],
                            'total_flops': self.training_flops,
                            'avg_power_mw': epoch_avg.get('power_consumption_mw', 0),
                            'training_time_hours': epoch_time / 3600,
                        }

                        model_url = self.hf_integration.push_model_checkpoint(
                            model=self.model,
                            config=self.config,
                            tokenizer=tokenizer_info,
                            epoch=epoch + 1,
                            metrics=hf_metrics,
                            commit_message=f"Training checkpoint - Epoch {epoch+1}/{num_epochs} - Loss: {epoch_avg['total_loss']:.4f}"
                        )

                        self.logger.info(f"✅ Model pushed to HuggingFace: {model_url}")

                    except Exception as e:
                        self.logger.error(f"❌ Failed to push model to HuggingFace: {e}")

                # Log epoch summary to WandB with HF URL
                self.wandb_integration.log_epoch_summary(
                    epoch=epoch + 1,
                    epoch_metrics=epoch_avg,
                    model_url=model_url
                )

                # Create and log training visualizations
                self.wandb_integration.create_training_visualizations()

                # Generate advanced metrics analysis for this epoch
                self._process_epoch_advanced_metrics(epoch + 1)

                self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
                self.logger.info(f"  Average loss: {epoch_avg['total_loss']:.4f}")
                self.logger.info(f"  Average tokens/s: {epoch_avg['tokens_per_second']:.2f}")
                self.logger.info(f"  Total FLOPS: {self.training_flops:,}")
                if model_url:
                    self.logger.info(f"  🤗 Model URL: {model_url}")

                # Clear epoch data for next epoch
                self.epoch_robot_predictions = []
                self.epoch_robot_targets = []
                self.epoch_task_descriptions = []

            # Training completed
            training_time = time.time() - training_start_time

            self.logger.info(f"Training completed in {training_time:.2f}s")
            self.logger.info(f"Total training FLOPS: {self.training_flops:,}")

            # Final evaluation
            final_eval = self._evaluate_with_monitoring(train_loader, num_samples=10)
            self.logger.info(f"Final evaluation: {final_eval}")

            # Log final evaluation to WandB
            self.wandb_integration.log_training_metrics({
                'final_eval/loss': final_eval['eval_loss'],
                'final_eval/inference_time_ms': final_eval['avg_inference_time_ms'],
                'final_eval/inference_flops': final_eval['inference_flops']
            })

            # Push final model to HuggingFace
            if self.push_to_hub and self.hf_integration:
                try:
                    self.logger.info("🚀 Pushing final model to HuggingFace Hub...")

                    final_metrics = {
                        'total_epochs': num_epochs,
                        'final_loss': final_eval['eval_loss'],
                        'total_training_time_hours': training_time / 3600,
                        'total_flops': self.training_flops,
                        'final_tokens_per_second': epoch_avg.get('tokens_per_second', 0),
                    }

                    final_model_url = self.hf_integration.push_final_model(
                        model=self.model,
                        config=self.config,
                        tokenizer=tokenizer_info,
                        metrics=final_metrics
                    )

                    self.logger.info(f"✅ Final model pushed: {final_model_url}")

                    # Log final model URL to WandB
                    self.wandb_integration.wandb_integration.log({
                        'final_model_url': final_model_url
                    })

                except Exception as e:
                    self.logger.error(f"❌ Failed to push final model: {e}")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        finally:
            # Stop monitoring and save results
            monitoring_summary = self.monitor.stop_monitoring()

            if self.carbon_tracker:
                self.carbon_tracker.stop()

            # Log final energy metrics if CodeCarbon was used
            if self.carbon_tracker:
                try:
                    carbon_file = Path(self.monitor.output_dir) / "emissions.csv"
                    if carbon_file.exists():
                        import pandas as pd
                        emissions_df = pd.read_csv(carbon_file)
                        if not emissions_df.empty:
                            final_energy_metrics = {
                                'energy_consumed_kwh': emissions_df['energy_consumed'].iloc[-1],
                                'carbon_emissions_kg': emissions_df['emissions'].iloc[-1],
                                'power_consumption_watts': emissions_df['cpu_power'].iloc[-1] if 'cpu_power' in emissions_df.columns else 0,
                                'tokens_generated': total_steps * batch_size * self.config.max_seq_len,
                            }
                            self.wandb_integration.log_energy_metrics(final_energy_metrics)
                            self.logger.info(f"🌍 Final energy consumption: {final_energy_metrics['energy_consumed_kwh']:.6f} kWh")
                            self.logger.info(f"🌍 Final carbon emissions: {final_energy_metrics['carbon_emissions_kg']:.6f} kg CO2")
                except Exception as e:
                    self.logger.warning(f"Failed to log final energy metrics: {e}")

            # Generate comprehensive training report
            final_report = self._generate_training_report(monitoring_summary, training_time, total_steps)

            # Log final summary to WandB
            summary_metrics = {
                'training/total_time_hours': training_time / 3600,
                'training/total_steps': total_steps,
                'training/total_epochs': num_epochs,
                'training/final_loss': final_eval.get('eval_loss', 0),
                'performance/total_flops': self.training_flops,
                'performance/avg_tokens_per_second': final_report['performance_analysis']['training_efficiency_flops_per_second'] / self.training_flops if self.training_flops > 0 else 0,
                'system/peak_temperature_c': monitoring_summary.get('peak_cpu_temperature_c', 0),
                'system/peak_memory_mb': monitoring_summary.get('peak_ram_usage_mb', 0),
                'efficiency/flops_per_second': final_report['performance_analysis']['training_efficiency_flops_per_second'],
            }

            # Finish WandB run with summary
            self.wandb_integration.finish_run(summary_metrics)

            self.logger.info("🎯 Training completed with comprehensive monitoring!")
            self.logger.info(f"📊 WandB Run: https://wandb.ai/{self.wandb_integration.entity}/{self.wandb_integration.project_name}/runs/{self.wandb_integration.run.id}")
            if self.hf_integration:
                self.logger.info(f"🤗 HuggingFace Model: https://huggingface.co/{self.hf_integration.repo_id}")

    def _setup_data_loader(self, coco_data_path: str, robot_data_path: Optional[str], batch_size: int):
        """Setup data loader optimized for Pi Zero"""
        # Load datasets
        coco_dataset = COCODataset(
            coco_data_path,
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.config.vocab_size
        )

        # Use simple DataLoader with no multiprocessing for Pi Zero
        train_loader = DataLoader(
            coco_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # No multiprocessing on Pi Zero
            pin_memory=False,  # No GPU
            drop_last=True
        )

        return train_loader

    def _get_current_monitoring_state(self) -> Dict:
        """Get current monitoring state"""
        if self.monitor.metrics_history:
            latest_metrics = self.monitor.metrics_history[-1]
            return {
                'ram_usage_mb': latest_metrics.ram_usage_mb,
                'cpu_usage_percent': latest_metrics.cpu_usage_percent,
                'cpu_temperature_c': latest_metrics.cpu_temperature_c,
                'power_consumption_mw': latest_metrics.power_consumption_mw
            }
        return {}

    def _get_current_temperature(self) -> float:
        """Get current CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return int(f.read().strip()) / 1000.0
        except:
            return 50.0  # Default safe temperature

    def _evaluate_with_monitoring(self, data_loader: DataLoader, num_samples: int = 10) -> Dict:
        """Evaluation with monitoring"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        inference_times = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Monitor inference
                inference_start = time.time()
                outputs = self.model(input_ids=input_ids)
                inference_time = time.time() - inference_start

                loss, _ = self.loss_fn(outputs, labels)

                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                inference_times.append(inference_time * 1000)  # Convert to ms

                # Track inference FLOPS
                self.inference_flops += self.model_flops.get('flops', 0) * batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_inference_time = np.mean(inference_times)

        return {
            'eval_loss': avg_loss,
            'avg_inference_time_ms': avg_inference_time,
            'inference_flops': self.inference_flops
        }

    def _save_checkpoint_with_metrics(self, optimizer: optim.Optimizer, scheduler):
        """Save checkpoint with monitoring metrics"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_loss': self.best_loss,
            'training_flops': self.training_flops,
            'inference_flops': self.inference_flops,
            'model_flops_info': self.model_flops,
            'monitoring_summary': self.monitor._generate_summary() if self.monitor.metrics_history else {}
        }

        filename = f"bitgen_checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, self.output_dir / filename)
        self.logger.info(f"Saved checkpoint with metrics: {filename}")

    def _generate_training_report(self, monitoring_summary: Dict, training_time: float, total_steps: int):
        """Generate comprehensive training report"""
        report = {
            'training_summary': {
                'total_training_time_seconds': training_time,
                'total_steps': total_steps,
                'final_epoch': self.epoch,
                'training_flops': self.training_flops,
                'inference_flops': self.inference_flops,
                'model_info': self.model_flops
            },
            'system_info': self.monitor.get_system_info(),
            'monitoring_results': monitoring_summary,
            'performance_analysis': {
                'avg_flops_per_step': self.training_flops / total_steps if total_steps > 0 else 0,
                'training_efficiency_flops_per_second': self.training_flops / training_time if training_time > 0 else 0,
                'energy_efficiency_flops_per_mj': (
                    self.training_flops / monitoring_summary.get('total_energy_consumed_mj', 1)
                    if monitoring_summary.get('total_energy_consumed_mj', 0) > 0 else 0
                ),
                'carbon_efficiency_flops_per_g_co2': (
                    self.training_flops / monitoring_summary.get('total_carbon_emissions_g', 1)
                    if monitoring_summary.get('total_carbon_emissions_g', 0) > 0 else 0
                )
            },
            'raspberry_pi_metrics': {
                'thermal_performance': {
                    'avg_cpu_temp_c': monitoring_summary.get('avg_cpu_temperature_c', 0),
                    'peak_cpu_temp_c': monitoring_summary.get('peak_cpu_temperature_c', 0),
                    'thermal_throttling_risk': 'High' if monitoring_summary.get('peak_cpu_temperature_c', 0) > 70 else 'Low'
                },
                'power_efficiency': {
                    'avg_power_mw': monitoring_summary.get('avg_power_consumption_mw', 0),
                    'peak_power_mw': monitoring_summary.get('peak_power_consumption_mw', 0),
                    'total_energy_mj': monitoring_summary.get('total_energy_consumed_mj', 0),
                    'estimated_battery_life_hours': self._estimate_battery_life(monitoring_summary)
                },
                'memory_efficiency': {
                    'avg_ram_usage_mb': monitoring_summary.get('avg_ram_usage_mb', 0),
                    'peak_ram_usage_mb': monitoring_summary.get('peak_ram_usage_mb', 0),
                    'memory_utilization_percent': (
                        monitoring_summary.get('peak_ram_usage_mb', 0) /
                        self.monitor.get_system_info()['memory']['total_mb'] * 100
                    )
                }
            }
        }

        # Save comprehensive report
        report_path = self.output_dir / "comprehensive_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Comprehensive training report saved to {report_path}")

        # Log key metrics
        self.logger.info("Training Summary:")
        self.logger.info(f"  Total FLOPS: {self.training_flops:,}")
        self.logger.info(f"  Energy consumed: {monitoring_summary.get('total_energy_consumed_mj', 0):.2f} mJ")
        self.logger.info(f"  Carbon emissions: {monitoring_summary.get('total_carbon_emissions_g', 0):.3f} g CO2")
        self.logger.info(f"  Average power: {monitoring_summary.get('avg_power_consumption_mw', 0):.1f} mW")
        self.logger.info(f"  Peak temperature: {monitoring_summary.get('peak_cpu_temperature_c', 0):.1f}°C")
        self.logger.info(f"  Peak RAM usage: {monitoring_summary.get('peak_ram_usage_mb', 0):.1f} MB")

        return report

    def _estimate_battery_life(self, monitoring_summary: Dict) -> float:
        """Estimate battery life for Raspberry Pi Zero"""
        avg_power_mw = monitoring_summary.get('avg_power_consumption_mw', 300)

        # Typical power bank capacity for Pi Zero: 10,000 mAh at 5V = 50Wh = 180,000J = 180,000,000 mJ
        battery_capacity_mj = 180_000_000  # 10Ah power bank

        # Battery life in hours
        if avg_power_mw > 0:
            battery_life_ms = battery_capacity_mj / avg_power_mw
            battery_life_hours = battery_life_ms / (1000 * 3600)  # Convert to hours
            return battery_life_hours

        return 0.0

    def _process_epoch_advanced_metrics(self, epoch: int):
        """Process and log advanced metrics at the end of each epoch"""

        self.logger.info(f"🔬 Generating advanced metrics analysis for epoch {epoch}...")

        try:
            # 1. Generate Episodic Memory Heatmaps
            if self.advanced_metrics and self.advanced_metrics.memory_analyzer:
                self.logger.info("📊 Creating episodic memory heatmaps...")

                memory_heatmap_path = self.advanced_metrics.output_dir / f"memory_analysis_epoch_{epoch}.png"
                memory_fig = self.advanced_metrics.memory_analyzer.create_memory_heatmaps(str(memory_heatmap_path))

                # Log to WandB
                self.wandb_integration.run.log({
                    f"episodic_memory/heatmaps_epoch_{epoch}": wandb.Image(str(memory_heatmap_path)),
                    f"episodic_memory/utilization_epoch_{epoch}": self.advanced_metrics.memory_analyzer.memory_usage_history[-1]['utilization'] if self.advanced_metrics.memory_analyzer.memory_usage_history else 0,
                    f"episodic_memory/diversity_epoch_{epoch}": self.advanced_metrics.memory_analyzer.memory_usage_history[-1]['memory_diversity'] if self.advanced_metrics.memory_analyzer.memory_usage_history else 0,
                })

                self.logger.info(f"   ✅ Memory heatmaps saved and logged to WandB")

            # 2. Generate Attention Heatmaps focusing on important tokens
            if self.advanced_metrics and self.advanced_metrics.attention_analyzer:
                if self.advanced_metrics.attention_analyzer.attention_history:
                    self.logger.info("🎯 Creating attention heatmaps for important tokens...")

                    # Get latest attention data
                    latest_attention = self.advanced_metrics.attention_analyzer.attention_history[-1]

                    # Create static attention heatmap
                    attention_heatmap_path = self.advanced_metrics.output_dir / f"attention_heatmaps_epoch_{epoch}.png"
                    attention_fig = self.advanced_metrics.attention_analyzer.create_attention_heatmaps(
                        attention_weights=torch.tensor(latest_attention['attention_weights']),
                        input_tokens=torch.tensor(latest_attention['input_tokens']),
                        save_path=str(attention_heatmap_path)
                    )

                    # Create interactive attention heatmap
                    interactive_attention = self.advanced_metrics.attention_analyzer.create_interactive_attention_heatmap(
                        attention_weights=torch.tensor(latest_attention['attention_weights']),
                        input_tokens=torch.tensor(latest_attention['input_tokens'])
                    )

                    # Log attention metrics
                    attention_stats = latest_attention['stats']
                    attention_metrics = {
                        f"attention/entropy_epoch_{epoch}": np.mean(attention_stats['attention_entropy']),
                        f"attention/important_tokens_epoch_{epoch}": len(attention_stats['important_tokens']),
                        f"attention/num_sinks_epoch_{epoch}": len(attention_stats['attention_sinks']),
                    }

                    # Log head specialization metrics
                    for head_idx, head_info in enumerate(attention_stats['head_specialization']):
                        attention_metrics.update({
                            f"attention/head_{head_idx}_local_focus_epoch_{epoch}": head_info['local_focus'],
                            f"attention/head_{head_idx}_concentration_epoch_{epoch}": head_info['concentration'],
                        })

                    # Log to WandB
                    self.wandb_integration.run.log({
                        f"attention/heatmaps_epoch_{epoch}": wandb.Image(str(attention_heatmap_path)),
                        f"attention/interactive_epoch_{epoch}": wandb.Plotly(interactive_attention),
                        **attention_metrics
                    })

                    self.logger.info(f"   ✅ Attention heatmaps created and logged")
                    self.logger.info(f"   🎯 Important tokens identified: {len(attention_stats['important_tokens'])}")
                    self.logger.info(f"   🔍 Attention sinks detected: {len(attention_stats['attention_sinks'])}")

            # 3. Generate Reasoning Matrices (Robot Selection Confusion Matrix)
            if (self.epoch_robot_predictions and self.epoch_robot_targets and
                self.advanced_metrics and self.advanced_metrics.reasoning_analyzer):

                self.logger.info("🤖 Creating reasoning matrix for robot selection...")

                # Update epoch confusion matrix
                epoch_predictions = np.array(self.epoch_robot_predictions)
                epoch_targets = np.array(self.epoch_robot_targets)

                confusion_data = self.advanced_metrics.reasoning_analyzer.update_epoch_confusion_matrix(
                    epoch_predictions=epoch_predictions,
                    epoch_targets=epoch_targets,
                    epoch=epoch
                )

                # Create confusion matrix evolution visualization
                confusion_evolution_path = self.advanced_metrics.output_dir / f"reasoning_matrix_epoch_{epoch}.png"
                confusion_fig = self.advanced_metrics.reasoning_analyzer.create_confusion_matrix_evolution(
                    str(confusion_evolution_path)
                )

                # Create reasoning improvement chart
                improvement_chart_path = self.advanced_metrics.output_dir / f"reasoning_improvement_epoch_{epoch}.png"
                improvement_fig = self.advanced_metrics.reasoning_analyzer.create_reasoning_improvement_chart(
                    str(improvement_chart_path)
                )

                # Create interactive reasoning dashboard
                reasoning_dashboard = self.advanced_metrics.reasoning_analyzer.create_interactive_reasoning_dashboard()

                # Calculate improvement metrics
                epoch_accuracy = confusion_data['accuracy']
                per_class_accuracy = confusion_data['per_class_accuracy']

                # Log reasoning metrics to WandB
                reasoning_metrics = {
                    f"reasoning/accuracy_epoch_{epoch}": epoch_accuracy,
                    f"reasoning/avg_per_class_accuracy_epoch_{epoch}": np.mean(per_class_accuracy),
                    f"reasoning/confusion_matrix_epoch_{epoch}": wandb.Image(str(confusion_evolution_path)),
                    f"reasoning/improvement_chart_epoch_{epoch}": wandb.Image(str(improvement_chart_path)),
                }

                # Log per-robot accuracy
                robot_types = self.advanced_metrics.reasoning_analyzer.robot_types
                for i, robot_type in enumerate(robot_types):
                    if i < len(per_class_accuracy):
                        reasoning_metrics[f"reasoning/{robot_type}_accuracy_epoch_{epoch}"] = per_class_accuracy[i]

                # Add interactive dashboard
                if reasoning_dashboard:
                    reasoning_metrics[f"reasoning/dashboard_epoch_{epoch}"] = wandb.Plotly(reasoning_dashboard)

                # Log to WandB
                self.wandb_integration.run.log(reasoning_metrics)

                self.logger.info(f"   ✅ Reasoning matrix created and logged")
                self.logger.info(f"   🎯 Epoch accuracy: {epoch_accuracy:.3f}")
                self.logger.info(f"   📈 Average per-class accuracy: {np.mean(per_class_accuracy):.3f}")

                # Show improvement over epochs
                if len(self.advanced_metrics.reasoning_analyzer.confusion_matrices) > 1:
                    prev_accuracy = self.advanced_metrics.reasoning_analyzer.confusion_matrices[-2]['accuracy']
                    improvement = epoch_accuracy - prev_accuracy
                    self.logger.info(f"   📊 Improvement from previous epoch: {improvement:+.3f}")

            # 4. Log comprehensive epoch analysis summary
            epoch_summary_metrics = {
                f"epoch_analysis/epoch": epoch,
                f"epoch_analysis/timestamp": datetime.now().timestamp(),
            }

            # Add memory analysis summary
            if self.advanced_metrics and self.advanced_metrics.memory_analyzer.memory_usage_history:
                latest_memory = self.advanced_metrics.memory_analyzer.memory_usage_history[-1]
                epoch_summary_metrics.update({
                    f"epoch_analysis/memory_utilization": latest_memory['utilization'],
                    f"epoch_analysis/memory_diversity": latest_memory['memory_diversity'],
                    f"epoch_analysis/active_memories": latest_memory['active_memories'],
                })

            # Add reasoning analysis summary
            if self.epoch_robot_predictions and self.epoch_robot_targets:
                epoch_accuracy = (np.array(self.epoch_robot_predictions) == np.array(self.epoch_robot_targets)).mean()
                epoch_summary_metrics[f"epoch_analysis/reasoning_accuracy"] = epoch_accuracy

            # Log summary
            self.wandb_integration.run.log(epoch_summary_metrics)

            self.logger.info("🎨 Advanced metrics analysis completed and logged to WandB")

        except Exception as e:
            self.logger.error(f"❌ Failed to process advanced metrics for epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
