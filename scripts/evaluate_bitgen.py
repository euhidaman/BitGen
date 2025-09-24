"""
BitGen Evaluation Suite
Comprehensive evaluation for embedded multimodal language models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

# Import BitGen components
from bitgen_model import BitGenModel, BitGenConfig
from data_loader import COCODataset, RobotSelectionDataset, BitGenTokenizer
from adaptive_loss import PerformanceTracker

class BitGenEvaluator:
    """Comprehensive evaluation suite for BitGen models"""

    def __init__(self, model: BitGenModel, config: BitGenConfig, device: str = 'cpu'):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.tokenizer = BitGenTokenizer(config.vocab_size)

        # Evaluation metrics storage
        self.results = defaultdict(list)
        self.detailed_results = {}

    def evaluate_language_modeling(self, test_loader: DataLoader) -> Dict:
        """Evaluate language modeling capabilities"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        perplexities = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Language Modeling Eval"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids)
                logits = outputs['logits']

                # Calculate cross-entropy loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
                losses = loss_fn(shift_logits.view(-1, self.config.vocab_size),
                               shift_labels.view(-1))

                # Mask out padding tokens
                mask = (shift_labels != 0).float().view(-1)
                masked_losses = losses * mask

                batch_loss = masked_losses.sum()
                batch_tokens = mask.sum()

                total_loss += batch_loss.item()
                total_tokens += batch_tokens.item()

                # Calculate perplexity per sequence
                seq_losses = masked_losses.view(shift_labels.shape[0], -1).sum(dim=1)
                seq_tokens = mask.view(shift_labels.shape[0], -1).sum(dim=1)
                seq_perplexities = torch.exp(seq_losses / (seq_tokens + 1e-8))
                perplexities.extend(seq_perplexities.cpu().tolist())

        avg_loss = total_loss / total_tokens
        avg_perplexity = np.exp(avg_loss)

        return {
            'average_loss': avg_loss,
            'perplexity': avg_perplexity,
            'perplexity_std': np.std(perplexities),
            'total_tokens_evaluated': total_tokens
        }

    def evaluate_vision_text_alignment(self, test_loader: DataLoader) -> Dict:
        """Evaluate vision-text alignment capabilities"""
        self.model.eval()

        correct_matches = 0
        total_samples = 0
        similarity_scores = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Vision-Text Alignment Eval"):
                if 'images' not in batch or batch['images'] is None:
                    continue

                input_ids = batch['input_ids'].to(self.device)
                images = batch['images'].to(self.device)
                batch_size = input_ids.size(0)

                # Get model outputs
                outputs = self.model(input_ids, images=images)

                # Extract text and vision features (simplified evaluation)
                # In a real implementation, we'd have separate encoders
                text_features = outputs['logits'].mean(dim=1)  # Average over sequence

                # Create positive and negative pairs
                for i in range(batch_size):
                    # Positive pair (same index)
                    pos_sim = torch.cosine_similarity(
                        text_features[i:i+1], text_features[i:i+1], dim=1
                    ).item()

                    # Negative pairs (different indices)
                    neg_similarities = []
                    for j in range(batch_size):
                        if i != j:
                            neg_sim = torch.cosine_similarity(
                                text_features[i:i+1], text_features[j:j+1], dim=1
                            ).item()
                            neg_similarities.append(neg_sim)

                    if neg_similarities:
                        max_neg_sim = max(neg_similarities)
                        if pos_sim > max_neg_sim:
                            correct_matches += 1

                        similarity_scores.append({
                            'positive': pos_sim,
                            'max_negative': max_neg_sim,
                            'correct': pos_sim > max_neg_sim
                        })

                    total_samples += 1

        accuracy = correct_matches / total_samples if total_samples > 0 else 0.0

        return {
            'alignment_accuracy': accuracy,
            'total_samples': total_samples,
            'avg_positive_similarity': np.mean([s['positive'] for s in similarity_scores]),
            'avg_negative_similarity': np.mean([s['max_negative'] for s in similarity_scores]),
            'similarity_scores': similarity_scores[:100]  # Keep sample for analysis
        }

    def evaluate_reasoning_capabilities(self, test_data: List[Dict]) -> Dict:
        """Evaluate reasoning capabilities using reasoning-specific prompts"""
        self.model.eval()

        correct_reasoning = 0
        total_questions = 0
        reasoning_scores = []

        for item in tqdm(test_data, desc="Reasoning Evaluation"):
            question = item.get('question', '')
            expected_answer = item.get('answer', '')

            # Format input with reasoning tags
            input_text = f"<reasoning>Question: {question}</reasoning><answer>"
            input_ids = torch.tensor([self.tokenizer.encode(input_text)]).to(self.device)

            with torch.no_grad():
                # Generate response
                generated, _ = self.model.generate_embedded(
                    input_ids, max_length=100, temperature=0.7
                )

                # Decode response
                response = self.tokenizer.decode(generated[0].cpu().tolist())

                # Extract answer portion
                if '</answer>' in response:
                    answer_part = response.split('<answer>')[1].split('</answer>')[0].strip()
                else:
                    answer_part = response.split('<answer>')[1].strip() if '<answer>' in response else response

                # Simple answer matching (can be made more sophisticated)
                is_correct = self._check_answer_correctness(answer_part, expected_answer)

                if is_correct:
                    correct_reasoning += 1

                reasoning_scores.append({
                    'question': question,
                    'expected': expected_answer,
                    'generated': answer_part,
                    'correct': is_correct
                })

            total_questions += 1

        accuracy = correct_reasoning / total_questions if total_questions > 0 else 0.0

        return {
            'reasoning_accuracy': accuracy,
            'total_questions': total_questions,
            'sample_results': reasoning_scores[:20]  # Keep samples for analysis
        }

    def evaluate_robot_selection(self, robot_test_loader: DataLoader) -> Dict:
        """Evaluate robot selection accuracy"""
        self.model.eval()

        correct_selections = 0
        total_selections = 0
        selection_confidence = []

        with torch.no_grad():
            for batch in tqdm(robot_test_loader, desc="Robot Selection Eval"):
                input_ids = batch['input_ids'].to(self.device)
                target_robots = batch['target_robot'].to(self.device)

                outputs = self.model(input_ids, return_robot_selection=True)
                robot_probs = outputs['robot_selection']

                # Get predicted robots
                predicted_robots = robot_probs.argmax(dim=-1)

                # Calculate accuracy
                correct = (predicted_robots == target_robots).float()
                correct_selections += correct.sum().item()
                total_selections += target_robots.size(0)

                # Store confidence scores
                max_probs = robot_probs.max(dim=-1)[0]
                selection_confidence.extend(max_probs.cpu().tolist())

        accuracy = correct_selections / total_selections if total_selections > 0 else 0.0
        avg_confidence = np.mean(selection_confidence)

        return {
            'robot_selection_accuracy': accuracy,
            'average_confidence': avg_confidence,
            'total_selections': total_selections
        }

    def evaluate_memory_performance(self, test_loader: DataLoader) -> Dict:
        """Evaluate episodic memory performance"""
        self.model.eval()

        memory_retrieval_scores = []
        memory_consistency_scores = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Memory Performance Eval"):
                input_ids = batch['input_ids'].to(self.device)

                # First pass - store in memory
                outputs1 = self.model(input_ids)

                # Second pass - should retrieve from memory
                outputs2 = self.model(input_ids)

                # Measure consistency between passes
                logits1 = outputs1['logits']
                logits2 = outputs2['logits']

                # Calculate consistency (cosine similarity)
                consistency = torch.cosine_similarity(
                    logits1.view(logits1.size(0), -1),
                    logits2.view(logits2.size(0), -1),
                    dim=1
                )

                memory_consistency_scores.extend(consistency.cpu().tolist())

        avg_consistency = np.mean(memory_consistency_scores)

        return {
            'memory_consistency': avg_consistency,
            'consistency_std': np.std(memory_consistency_scores)
        }

    def evaluate_embedded_performance(self, test_loader: DataLoader) -> Dict:
        """Evaluate performance metrics relevant to embedded deployment"""
        self.model.eval()

        inference_times = []
        memory_usage = []
        token_throughput = []

        for batch in tqdm(test_loader, desc="Embedded Performance Eval"):
            input_ids = batch['input_ids'].to(self.device)
            batch_size, seq_len = input_ids.shape

            # Measure inference time
            start_time = time.time()

            with torch.no_grad():
                outputs = self.model(input_ids)

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to ms

            inference_times.append(inference_time)

            # Calculate token throughput
            tokens_per_second = (batch_size * seq_len) / (inference_time / 1000)
            token_throughput.append(tokens_per_second)

            # Memory usage (simplified - in practice would measure actual usage)
            estimated_memory = self._estimate_memory_usage(batch_size, seq_len)
            memory_usage.append(estimated_memory)

        return {
            'avg_inference_time_ms': np.mean(inference_times),
            'inference_time_std_ms': np.std(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'avg_tokens_per_second': np.mean(token_throughput),
            'estimated_memory_kb': np.mean(memory_usage),
            'max_memory_kb': np.max(memory_usage)
        }

    def run_comprehensive_evaluation(self,
                                   coco_test_path: str,
                                   robot_test_path: str,
                                   reasoning_test_path: str,
                                   batch_size: int = 4) -> Dict:
        """Run complete evaluation suite"""

        print("Starting comprehensive BitGen evaluation...")

        # Load test datasets
        coco_dataset = COCODataset(coco_test_path, max_seq_len=self.config.max_seq_len)
        coco_loader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=False)

        robot_dataset = RobotSelectionDataset(robot_test_path)
        robot_loader = DataLoader(robot_dataset, batch_size=batch_size, shuffle=False)

        # Load reasoning test data
        with open(reasoning_test_path, 'r') as f:
            reasoning_data = json.load(f)

        # Run all evaluations
        results = {}

        # Language modeling
        print("Evaluating language modeling...")
        results['language_modeling'] = self.evaluate_language_modeling(coco_loader)

        # Vision-text alignment
        print("Evaluating vision-text alignment...")
        results['vision_text_alignment'] = self.evaluate_vision_text_alignment(coco_loader)

        # Reasoning capabilities
        print("Evaluating reasoning capabilities...")
        results['reasoning'] = self.evaluate_reasoning_capabilities(reasoning_data)

        # Robot selection
        print("Evaluating robot selection...")
        results['robot_selection'] = self.evaluate_robot_selection(robot_loader)

        # Memory performance
        print("Evaluating memory performance...")
        results['memory'] = self.evaluate_memory_performance(coco_loader)

        # Embedded performance
        print("Evaluating embedded performance...")
        results['embedded'] = self.evaluate_embedded_performance(coco_loader)

        # Calculate overall score
        results['overall_score'] = self._calculate_overall_score(results)

        return results

    def _check_answer_correctness(self, generated: str, expected: str) -> bool:
        """Simple answer correctness check (can be made more sophisticated)"""
        generated = generated.lower().strip()
        expected = expected.lower().strip()

        # Exact match
        if generated == expected:
            return True

        # Contains expected answer
        if expected in generated:
            return True

        # Number matching for math problems
        import re
        gen_numbers = re.findall(r'\d+\.?\d*', generated)
        exp_numbers = re.findall(r'\d+\.?\d*', expected)

        if gen_numbers and exp_numbers:
            try:
                return float(gen_numbers[-1]) == float(exp_numbers[-1])
            except:
                pass

        return False

    def _estimate_memory_usage(self, batch_size: int, seq_len: int) -> float:
        """Estimate memory usage in KB"""
        # Simplified estimation
        hidden_state_mem = batch_size * seq_len * self.config.embed_dim * 4  # bytes
        attention_mem = (batch_size * self.config.num_layers *
                        self.config.attention_sinks * self.config.embed_dim * 4)
        memory_buffer_mem = self.config.memory_size * self.config.embed_dim * 4

        total_bytes = hidden_state_mem + attention_mem + memory_buffer_mem
        return total_bytes / 1024  # Convert to KB

    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate weighted overall performance score"""
        scores = []
        weights = []

        if 'language_modeling' in results:
            # Convert perplexity to score (lower is better)
            ppl_score = max(0, 1 - (results['language_modeling']['perplexity'] - 1) / 10)
            scores.append(ppl_score)
            weights.append(0.2)

        if 'vision_text_alignment' in results:
            scores.append(results['vision_text_alignment']['alignment_accuracy'])
            weights.append(0.2)

        if 'reasoning' in results:
            scores.append(results['reasoning']['reasoning_accuracy'])
            weights.append(0.2)

        if 'robot_selection' in results:
            scores.append(results['robot_selection']['robot_selection_accuracy'])
            weights.append(0.15)

        if 'memory' in results:
            scores.append(results['memory']['memory_consistency'])
            weights.append(0.15)

        if 'embedded' in results:
            # Penalize slow inference (convert ms to score)
            speed_score = max(0, 1 - results['embedded']['avg_inference_time_ms'] / 1000)
            scores.append(speed_score)
            weights.append(0.1)

        if scores and weights:
            return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            return 0.0

    def generate_evaluation_report(self, results: Dict, output_path: str):
        """Generate comprehensive evaluation report"""

        report = {
            'model_config': self.config.__dict__,
            'evaluation_results': results,
            'summary': {
                'overall_score': results.get('overall_score', 0.0),
                'strengths': self._identify_strengths(results),
                'weaknesses': self._identify_weaknesses(results),
                'recommendations': self._generate_recommendations(results)
            },
            'detailed_analysis': self._detailed_analysis(results)
        }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Evaluation report saved to {output_path}")

        # Generate visualizations
        self._generate_visualizations(results, Path(output_path).parent)

        return report

    def _identify_strengths(self, results: Dict) -> List[str]:
        """Identify model strengths based on evaluation results"""
        strengths = []

        if results.get('language_modeling', {}).get('perplexity', float('inf')) < 10:
            strengths.append("Good language modeling performance")

        if results.get('vision_text_alignment', {}).get('alignment_accuracy', 0) > 0.7:
            strengths.append("Strong vision-text alignment")

        if results.get('reasoning', {}).get('reasoning_accuracy', 0) > 0.6:
            strengths.append("Effective reasoning capabilities")

        if results.get('robot_selection', {}).get('robot_selection_accuracy', 0) > 0.8:
            strengths.append("Accurate robot selection")

        if results.get('embedded', {}).get('avg_inference_time_ms', float('inf')) < 200:
            strengths.append("Fast embedded inference")

        if not strengths:
            strengths.append("Model shows potential for improvement")

        return strengths

    def _identify_weaknesses(self, results: Dict) -> List[str]:
        """Identify model weaknesses based on evaluation results"""
        weaknesses = []

        if results.get('language_modeling', {}).get('perplexity', 0) > 20:
            weaknesses.append("High language modeling perplexity")

        if results.get('vision_text_alignment', {}).get('alignment_accuracy', 1) < 0.5:
            weaknesses.append("Poor vision-text alignment")

        if results.get('reasoning', {}).get('reasoning_accuracy', 1) < 0.4:
            weaknesses.append("Limited reasoning capabilities")

        if results.get('embedded', {}).get('avg_inference_time_ms', 0) > 500:
            weaknesses.append("Slow inference for embedded deployment")

        return weaknesses

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if results.get('language_modeling', {}).get('perplexity', 0) > 15:
            recommendations.append("Consider increasing training data or model capacity for language modeling")

        if results.get('vision_text_alignment', {}).get('alignment_accuracy', 1) < 0.6:
            recommendations.append("Improve cross-modal fusion training or increase vision data")

        if results.get('embedded', {}).get('avg_inference_time_ms', 0) > 300:
            recommendations.append("Apply more aggressive quantization or model pruning for embedded deployment")

        if results.get('memory', {}).get('memory_consistency', 1) < 0.7:
            recommendations.append("Tune episodic memory hyperparameters or training procedures")

        return recommendations

    def _detailed_analysis(self, results: Dict) -> Dict:
        """Perform detailed analysis of results"""
        analysis = {}

        # Efficiency analysis
        if 'embedded' in results:
            analysis['efficiency'] = {
                'inference_speed_rating': self._rate_inference_speed(results['embedded']),
                'memory_efficiency_rating': self._rate_memory_efficiency(results['embedded']),
                'embedded_readiness': self._assess_embedded_readiness(results)
            }

        # Capability analysis
        analysis['capabilities'] = {
            'multimodal_strength': self._assess_multimodal_strength(results),
            'reasoning_depth': self._assess_reasoning_depth(results),
            'memory_utilization': self._assess_memory_utilization(results)
        }

        return analysis

    def _rate_inference_speed(self, embedded_results: Dict) -> str:
        """Rate inference speed for embedded deployment"""
        avg_time = embedded_results.get('avg_inference_time_ms', float('inf'))

        if avg_time < 100:
            return "Excellent"
        elif avg_time < 200:
            return "Good"
        elif avg_time < 500:
            return "Fair"
        else:
            return "Poor"

    def _rate_memory_efficiency(self, embedded_results: Dict) -> str:
        """Rate memory efficiency"""
        max_memory = embedded_results.get('max_memory_kb', float('inf'))

        if max_memory < 256:
            return "Excellent"
        elif max_memory < 512:
            return "Good"
        elif max_memory < 1024:
            return "Fair"
        else:
            return "Poor"

    def _assess_embedded_readiness(self, results: Dict) -> str:
        """Assess overall readiness for embedded deployment"""
        embedded = results.get('embedded', {})

        time_ok = embedded.get('avg_inference_time_ms', float('inf')) < 300
        memory_ok = embedded.get('max_memory_kb', float('inf')) < 512
        accuracy_ok = results.get('overall_score', 0) > 0.6

        if time_ok and memory_ok and accuracy_ok:
            return "Ready"
        elif (time_ok and memory_ok) or accuracy_ok:
            return "Needs optimization"
        else:
            return "Not ready"

    def _assess_multimodal_strength(self, results: Dict) -> str:
        """Assess multimodal capabilities"""
        vision_acc = results.get('vision_text_alignment', {}).get('alignment_accuracy', 0)

        if vision_acc > 0.8:
            return "Strong"
        elif vision_acc > 0.6:
            return "Good"
        elif vision_acc > 0.4:
            return "Fair"
        else:
            return "Weak"

    def _assess_reasoning_depth(self, results: Dict) -> str:
        """Assess reasoning capabilities depth"""
        reasoning_acc = results.get('reasoning', {}).get('reasoning_accuracy', 0)

        if reasoning_acc > 0.8:
            return "Deep"
        elif reasoning_acc > 0.6:
            return "Good"
        elif reasoning_acc > 0.4:
            return "Basic"
        else:
            return "Limited"

    def _assess_memory_utilization(self, results: Dict) -> str:
        """Assess episodic memory utilization"""
        memory_consistency = results.get('memory', {}).get('memory_consistency', 0)

        if memory_consistency > 0.8:
            return "Effective"
        elif memory_consistency > 0.6:
            return "Good"
        elif memory_consistency > 0.4:
            return "Basic"
        else:
            return "Poor"

    def _generate_visualizations(self, results: Dict, output_dir: Path):
        """Generate evaluation visualizations"""
        output_dir.mkdir(exist_ok=True)

        # Performance radar chart
        self._create_radar_chart(results, output_dir / "performance_radar.png")

        # Time series plots if available
        if 'embedded' in results:
            self._create_performance_plots(results, output_dir / "performance_plots.png")

    def _create_radar_chart(self, results: Dict, output_path: Path):
        """Create radar chart of model capabilities"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Extract scores for radar chart
            categories = []
            scores = []

            if 'language_modeling' in results:
                categories.append('Language\nModeling')
                ppl = results['language_modeling'].get('perplexity', 10)
                score = max(0, 1 - (ppl - 1) / 10)  # Convert perplexity to 0-1 score
                scores.append(score)

            if 'vision_text_alignment' in results:
                categories.append('Vision-Text\nAlignment')
                scores.append(results['vision_text_alignment'].get('alignment_accuracy', 0))

            if 'reasoning' in results:
                categories.append('Reasoning')
                scores.append(results['reasoning'].get('reasoning_accuracy', 0))

            if 'robot_selection' in results:
                categories.append('Robot\nSelection')
                scores.append(results['robot_selection'].get('robot_selection_accuracy', 0))

            if 'memory' in results:
                categories.append('Memory\nConsistency')
                scores.append(results['memory'].get('memory_consistency', 0))

            if 'embedded' in results:
                categories.append('Embedded\nPerformance')
                time_ms = results['embedded'].get('avg_inference_time_ms', 1000)
                time_score = max(0, 1 - time_ms / 1000)
                scores.append(time_score)

            if len(categories) >= 3:  # Need at least 3 categories for radar chart
                # Create radar chart
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
                scores_np = np.array(scores)

                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                ax.plot(angles, scores_np, 'o-', linewidth=2, label='BitGen Performance')
                ax.fill(angles, scores_np, alpha=0.25)
                ax.set_xticks(angles)
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
                ax.grid(True)

                plt.title('BitGen Model Performance Overview', size=16, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Radar chart saved to {output_path}")

        except ImportError:
            print("Matplotlib not available, skipping radar chart generation")
        except Exception as e:
            print(f"Error generating radar chart: {e}")

    def _create_performance_plots(self, results: Dict, output_path: Path):
        """Create performance plots"""
        try:
            import matplotlib.pyplot as plt

            embedded = results.get('embedded', {})

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Inference time histogram
            if 'inference_times' in embedded:
                ax1.hist(embedded['inference_times'], bins=20, alpha=0.7, edgecolor='black')
                ax1.set_xlabel('Inference Time (ms)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Inference Time Distribution')

            # Memory usage
            if 'memory_usage' in embedded:
                ax2.plot(embedded['memory_usage'], marker='o', linewidth=2)
                ax2.set_xlabel('Batch Number')
                ax2.set_ylabel('Memory Usage (KB)')
                ax2.set_title('Memory Usage Over Time')

            # Token throughput
            if 'token_throughput' in embedded:
                ax3.bar(range(len(embedded['token_throughput'])), embedded['token_throughput'])
                ax3.set_xlabel('Batch Number')
                ax3.set_ylabel('Tokens/Second')
                ax3.set_title('Token Throughput')

            # Overall scores comparison
            scores = []
            labels = []
            for key, value in results.items():
                if key != 'embedded' and isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if 'accuracy' in subkey and isinstance(subvalue, (int, float)):
                            scores.append(subvalue)
                            labels.append(f"{key}\n{subkey}")

            if scores:
                ax4.bar(range(len(scores)), scores, color='skyblue', edgecolor='navy')
                ax4.set_xticks(range(len(scores)))
                ax4.set_xticklabels(labels, rotation=45, ha='right')
                ax4.set_ylabel('Accuracy')
                ax4.set_title('Performance Across Tasks')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Performance plots saved to {output_path}")

        except ImportError:
            print("Matplotlib not available, skipping performance plots")
        except Exception as e:
            print(f"Error generating performance plots: {e}")

def create_test_reasoning_data(output_file: str, num_questions: int = 100):
    """Create test data for reasoning evaluation"""

    reasoning_questions = [
        {
            "question": "If a robot needs to pick up a box that weighs 5kg and the robot's maximum lifting capacity is 10kg, can the robot pick up the box?",
            "answer": "yes",
            "category": "basic_logic"
        },
        {
            "question": "A robot is at position (0,0) and needs to reach (3,4). What is the minimum distance it needs to travel?",
            "answer": "5",
            "category": "mathematics"
        },
        {
            "question": "If it takes 2 minutes to charge a robot battery to 50% and 6 minutes total to charge to 100%, how long does it take to charge from 50% to 100%?",
            "answer": "4",
            "category": "arithmetic"
        }
    ]

    # Expand with variations
    test_data = []
    for i in range(num_questions):
        base_q = reasoning_questions[i % len(reasoning_questions)]
        test_data.append({
            "id": i,
            "question": base_q["question"],
            "answer": base_q["answer"],
            "category": base_q["category"]
        })

    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"Created {num_questions} reasoning test questions in {output_file}")

def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate BitGen model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--coco_test", type=str, required=True, help="Path to COCO test data")
    parser.add_argument("--robot_test", type=str, required=True, help="Path to robot test data")
    parser.add_argument("--reasoning_test", type=str, help="Path to reasoning test data")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)

    # Load model
    from bitgen_model import create_bitgen_model, BitGenConfig

    checkpoint = torch.load(args.model_path, map_location=args.device)
    config = BitGenConfig(**checkpoint['config'])

    model = create_bitgen_model('tiny')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create evaluator
    evaluator = BitGenEvaluator(model, config, args.device)

    # Create reasoning test data if not provided
    if not args.reasoning_test:
        args.reasoning_test = Path(args.output_dir) / "reasoning_test.json"
        create_test_reasoning_data(str(args.reasoning_test))

    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        args.coco_test,
        args.robot_test,
        args.reasoning_test,
        args.batch_size
    )

    # Generate report
    report_path = Path(args.output_dir) / "evaluation_report.json"
    evaluator.generate_evaluation_report(results, str(report_path))

    print(f"\nEvaluation completed!")
    print(f"Overall score: {results.get('overall_score', 0):.3f}")
    print(f"Full report available at: {report_path}")

if __name__ == "__main__":
    main()
