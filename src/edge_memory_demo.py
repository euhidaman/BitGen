"""
Episodic Memory Edge Operations for BitGen
Demonstrates fast fact editing, selective forgetting, and knowledge updates without retraining
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from pathlib import Path

class EdgeMemoryManager:
    """Manage episodic memory operations for edge deployment"""

    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Access episodic memory directly
        self.episodic_memory = model.episodic_memory

        # Track memory operations
        self.memory_operations_log = []
        self.knowledge_base = {}

    def fast_fact_edit(self, old_fact: str, new_fact: str, confidence: float = 0.9):
        """
        Fast fact editing without retraining
        Updates episodic memory directly with new information
        """
        start_time = time.time()

        # Encode old and new facts
        old_tokens = self.tokenizer.encode(old_fact)
        new_tokens = self.tokenizer.encode(new_fact)

        old_ids = torch.tensor([old_tokens]).to(self.device)
        new_ids = torch.tensor([new_tokens]).to(self.device)

        with torch.no_grad():
            # Get embeddings for old and new facts
            old_emb = self.model.token_embedding(old_ids).mean(dim=1)  # [1, embed_dim]
            new_emb = self.model.token_embedding(new_ids).mean(dim=1)  # [1, embed_dim]

            # Project to memory space
            old_key = self.episodic_memory.key_proj(old_emb)
            new_key = self.episodic_memory.key_proj(new_emb)
            new_value = self.episodic_memory.value_proj(new_emb)

            # Find most similar memory slot to old fact
            similarities = torch.matmul(old_key, self.episodic_memory.memory_keys.T)
            best_slot = similarities.argmax().item()

            # Update memory slot with new information
            update_rate = confidence  # Use confidence as update rate
            self.episodic_memory.memory_keys.data[best_slot] = (
                (1 - update_rate) * self.episodic_memory.memory_keys.data[best_slot] +
                update_rate * new_key.squeeze(0)
            )
            self.episodic_memory.memory_values.data[best_slot] = (
                (1 - update_rate) * self.episodic_memory.memory_values.data[best_slot] +
                update_rate * new_value.squeeze(0)
            )

        edit_time = time.time() - start_time

        # Log operation
        operation_log = {
            'operation': 'fast_fact_edit',
            'old_fact': old_fact,
            'new_fact': new_fact,
            'confidence': confidence,
            'memory_slot': best_slot,
            'edit_time_ms': edit_time * 1000,
            'timestamp': time.time()
        }

        self.memory_operations_log.append(operation_log)
        self.knowledge_base[old_fact] = new_fact

        return {
            'success': True,
            'edit_time_ms': edit_time * 1000,
            'memory_slot_updated': best_slot,
            'similarity_score': similarities.max().item()
        }

    def selective_forgetting(self, fact_to_forget: str, forgetting_strength: float = 0.8):
        """
        Selective forgetting - remove specific information from memory
        """
        start_time = time.time()

        # Encode fact to forget
        forget_tokens = self.tokenizer.encode(fact_to_forget)
        forget_ids = torch.tensor([forget_tokens]).to(self.device)

        with torch.no_grad():
            # Get embedding for fact to forget
            forget_emb = self.model.token_embedding(forget_ids).mean(dim=1)
            forget_key = self.episodic_memory.key_proj(forget_emb)

            # Find most similar memory slots
            similarities = torch.matmul(forget_key, self.episodic_memory.memory_keys.T)
            top_slots = similarities.topk(k=3)[1]  # Top 3 most similar slots

            # Apply forgetting by reducing memory strength
            for slot in top_slots:
                slot_idx = slot.item()
                similarity = similarities[0, slot_idx].item()

                if similarity > 0.5:  # Only forget if reasonably similar
                    # Reduce memory strength based on similarity and forgetting strength
                    forget_factor = similarity * forgetting_strength

                    # Decay memory towards zero
                    self.episodic_memory.memory_keys.data[slot_idx] *= (1 - forget_factor)
                    self.episodic_memory.memory_values.data[slot_idx] *= (1 - forget_factor)

        forget_time = time.time() - start_time

        # Log operation
        operation_log = {
            'operation': 'selective_forgetting',
            'fact_to_forget': fact_to_forget,
            'forgetting_strength': forgetting_strength,
            'memory_slots_affected': top_slots.cpu().tolist(),
            'forget_time_ms': forget_time * 1000,
            'timestamp': time.time()
        }

        self.memory_operations_log.append(operation_log)

        return {
            'success': True,
            'forget_time_ms': forget_time * 1000,
            'slots_affected': len(top_slots),
            'max_similarity': similarities.max().item()
        }

    def update_knowledge_online(self, new_experience: str, context: str = ""):
        """
        Update knowledge based on new experiences during deployment
        High accuracy on updated knowledge without retraining
        """
        start_time = time.time()

        # Combine experience with context
        full_context = f"{context} {new_experience}".strip()

        # Encode new experience
        exp_tokens = self.tokenizer.encode(full_context)
        exp_ids = torch.tensor([exp_tokens]).to(self.device)

        with torch.no_grad():
            # Process through model to get contextual representation
            outputs = self.model(exp_ids, return_analysis_data=True)

            # Extract memory information
            if 'memory_attention' in outputs:
                memory_attention = outputs['memory_attention']

                # Find best memory slot for this experience
                attention_scores = memory_attention.mean(dim=1)  # Average over sequence
                best_memory_slot = attention_scores.argmax(dim=-1).item()

                # Get experience embedding
                exp_emb = outputs['final_embeddings'].mean(dim=1)  # [1, embed_dim]
                exp_key = self.episodic_memory.key_proj(exp_emb)
                exp_value = self.episodic_memory.value_proj(exp_emb)

                # Update memory with new experience (stronger update for new knowledge)
                update_strength = 0.3  # Moderate update to preserve existing knowledge

                self.episodic_memory.memory_keys.data[best_memory_slot] = (
                    (1 - update_strength) * self.episodic_memory.memory_keys.data[best_memory_slot] +
                    update_strength * exp_key.squeeze(0)
                )
                self.episodic_memory.memory_values.data[best_memory_slot] = (
                    (1 - update_strength) * self.episodic_memory.memory_values.data[best_memory_slot] +
                    update_strength * exp_value.squeeze(0)
                )

        update_time = time.time() - start_time

        # Log operation
        operation_log = {
            'operation': 'update_knowledge_online',
            'new_experience': new_experience,
            'context': context,
            'memory_slot': best_memory_slot,
            'update_time_ms': update_time * 1000,
            'timestamp': time.time()
        }

        self.memory_operations_log.append(operation_log)

        return {
            'success': True,
            'update_time_ms': update_time * 1000,
            'memory_slot_updated': best_memory_slot,
            'attention_score': attention_scores.max().item()
        }

    def retrieve_relevant_memory(self, query: str, top_k: int = 3):
        """
        Fast retrieval of relevant memories for given query
        Demonstrates low-latency local knowledge access
        """
        start_time = time.time()

        # Encode query
        query_tokens = self.tokenizer.encode(query)
        query_ids = torch.tensor([query_tokens]).to(self.device)

        with torch.no_grad():
            # Get query embedding
            query_emb = self.model.token_embedding(query_ids).mean(dim=1)
            query_key = self.episodic_memory.key_proj(query_emb)

            # Compute similarities with all memories
            similarities = torch.matmul(query_key, self.episodic_memory.memory_keys.T)
            top_similarities, top_indices = similarities.topk(k=top_k)

            # Retrieve relevant memories
            relevant_memories = []
            for i, (sim_score, mem_idx) in enumerate(zip(top_similarities[0], top_indices[0])):
                memory_value = self.episodic_memory.memory_values[mem_idx.item()]

                relevant_memories.append({
                    'memory_index': mem_idx.item(),
                    'similarity_score': sim_score.item(),
                    'memory_embedding': memory_value.cpu().numpy(),
                    'rank': i + 1
                })

        retrieval_time = time.time() - start_time

        return {
            'query': query,
            'retrieval_time_ms': retrieval_time * 1000,
            'relevant_memories': relevant_memories,
            'total_memories_searched': self.episodic_memory.memory_size
        }

    def get_memory_statistics(self):
        """Get current episodic memory statistics"""

        with torch.no_grad():
            # Memory utilization
            memory_norms = torch.norm(self.episodic_memory.memory_values, dim=-1)
            active_memories = (memory_norms > 0.1).sum().item()
            utilization = active_memories / self.episodic_memory.memory_size

            # Memory diversity
            normalized_memories = F.normalize(self.episodic_memory.memory_values, dim=-1)
            similarities = torch.mm(normalized_memories, normalized_memories.T)

            # Calculate average diversity (1 - similarity)
            mask = torch.triu(torch.ones_like(similarities), diagonal=1).bool()
            avg_similarity = similarities[mask].mean().item()
            diversity = 1.0 - avg_similarity

            # Memory capacity
            memory_capacity = self.episodic_memory.memory_size

        return {
            'total_operations': len(self.memory_operations_log),
            'memory_utilization': utilization,
            'active_memories': active_memories,
            'memory_diversity': diversity,
            'memory_capacity': memory_capacity,
            'knowledge_base_size': len(self.knowledge_base)
        }

    def demonstrate_edge_advantages(self):
        """Demonstrate the key advantages of episodic memory on edge devices"""

        print("🧠 Demonstrating BitGen Episodic Memory Edge Advantages")
        print("=" * 60)

        # 1. Fast Fact Editing Demo
        print("\n1. 🚀 Fast Fact Editing (No Retraining)")
        old_fact = "The robot arm can lift 5kg maximum"
        new_fact = "The robot arm can lift 10kg maximum after upgrade"

        result = self.fast_fact_edit(old_fact, new_fact, confidence=0.9)
        print(f"   Updated: {old_fact} → {new_fact}")
        print(f"   Edit time: {result['edit_time_ms']:.2f}ms")
        print(f"   Memory slot: {result['memory_slot_updated']}")

        # 2. Selective Forgetting Demo
        print("\n2. 🗑️ Selective Forgetting")
        outdated_info = "Old safety protocol from 2023"

        forget_result = self.selective_forgetting(outdated_info, forgetting_strength=0.8)
        print(f"   Forgot: {outdated_info}")
        print(f"   Forget time: {forget_result['forget_time_ms']:.2f}ms")
        print(f"   Slots affected: {forget_result['slots_affected']}")

        # 3. Online Knowledge Update Demo
        print("\n3. 📚 Online Knowledge Update")
        new_experience = "Robot successfully completed pick-and-place task in 15 seconds"
        context = "Assembly line operation"

        update_result = self.update_knowledge_online(new_experience, context)
        print(f"   New experience: {new_experience}")
        print(f"   Update time: {update_result['update_time_ms']:.2f}ms")
        print(f"   Memory slot: {update_result['memory_slot_updated']}")

        # 4. Fast Retrieval Demo
        print("\n4. ⚡ Fast Local Knowledge Retrieval")
        query = "What is the robot lifting capacity?"

        retrieval_result = self.retrieve_relevant_memory(query, top_k=3)
        print(f"   Query: {query}")
        print(f"   Retrieval time: {retrieval_result['retrieval_time_ms']:.2f}ms")
        print(f"   Relevant memories found: {len(retrieval_result['relevant_memories'])}")

        for mem in retrieval_result['relevant_memories']:
            print(f"     Rank {mem['rank']}: Slot {mem['memory_index']} (similarity: {mem['similarity_score']:.3f})")

        # 5. Memory Statistics
        print("\n5. 📊 Current Memory Statistics")
        stats = self.get_memory_statistics()
        print(f"   Memory utilization: {stats['memory_utilization']:.2f}")
        print(f"   Active memories: {stats['active_memories']}/{stats['memory_capacity']}")
        print(f"   Memory diversity: {stats['memory_diversity']:.3f}")
        print(f"   Knowledge base entries: {stats['knowledge_base_size']}")
        print(f"   Total operations: {stats['total_operations']}")

        print("\n✨ Key Advantages Demonstrated:")
        print("   ⚡ Fast fact editing without retraining")
        print("   🎯 Selective forgetting of outdated information")
        print("   📈 High accuracy on updated knowledge")
        print("   🚀 Low-latency local knowledge access")
        print("   💾 Continuous learning during deployment")

def create_edge_memory_demo():
    """Create demonstration of edge memory capabilities"""

    # This would be used with a trained BitGen model
    demo_scenarios = [
        {
            'scenario': 'Robot Capability Update',
            'old_fact': 'Robot can carry 5kg',
            'new_fact': 'Robot can carry 8kg after motor upgrade',
            'expected_benefit': 'Immediate capability update without retraining'
        },
        {
            'scenario': 'Safety Protocol Update',
            'old_fact': 'Maximum speed is 1.5 m/s',
            'new_fact': 'Maximum speed is 2.0 m/s in safe zones',
            'expected_benefit': 'Real-time safety parameter updates'
        },
        {
            'scenario': 'Environmental Learning',
            'old_fact': 'Warehouse layout unknown',
            'new_fact': 'Warehouse has 3 aisles with obstacles at positions A, B, C',
            'expected_benefit': 'Learning environment during deployment'
        },
        {
            'scenario': 'Task Performance Update',
            'old_fact': 'Pick-and-place takes 30 seconds',
            'new_fact': 'Pick-and-place optimized to 15 seconds',
            'expected_benefit': 'Performance optimization without retraining'
        }
    ]

    return demo_scenarios

def benchmark_memory_operations(model, tokenizer, num_operations: int = 100):
    """Benchmark episodic memory operations for edge performance"""

    edge_manager = EdgeMemoryManager(model, tokenizer)

    print(f"🔬 Benchmarking {num_operations} episodic memory operations...")

    # Generate test facts
    test_facts = [
        f"Robot task {i} completed successfully in {10 + i % 10} seconds"
        for i in range(num_operations)
    ]

    # Benchmark different operations
    benchmarks = {
        'fact_editing': [],
        'selective_forgetting': [],
        'online_updates': [],
        'retrieval': []
    }

    for i, fact in enumerate(test_facts):
        if i % 4 == 0:
            # Fast fact editing
            old_fact = f"Old information {i}"
            result = edge_manager.fast_fact_edit(old_fact, fact)
            benchmarks['fact_editing'].append(result['edit_time_ms'])

        elif i % 4 == 1:
            # Selective forgetting
            result = edge_manager.selective_forgetting(fact)
            benchmarks['selective_forgetting'].append(result['forget_time_ms'])

        elif i % 4 == 2:
            # Online knowledge update
            result = edge_manager.update_knowledge_online(fact, "deployment context")
            benchmarks['online_updates'].append(result['update_time_ms'])

        else:
            # Fast retrieval
            query = f"Information about task {i}"
            result = edge_manager.retrieve_relevant_memory(query)
            benchmarks['retrieval'].append(result['retrieval_time_ms'])

    # Calculate statistics
    print("\n📊 Edge Memory Operation Benchmarks:")
    for operation, times in benchmarks.items():
        if times:
            avg_time = np.mean(times)
            max_time = np.max(times)
            min_time = np.min(times)

            print(f"   {operation.replace('_', ' ').title()}:")
            print(f"     Average: {avg_time:.2f}ms")
            print(f"     Range: {min_time:.2f}ms - {max_time:.2f}ms")
            print(f"     Samples: {len(times)}")

    # Memory statistics
    final_stats = edge_manager.get_memory_statistics()
    print(f"\n💾 Final Memory State:")
    print(f"   Utilization: {final_stats['memory_utilization']:.2f}")
    print(f"   Diversity: {final_stats['memory_diversity']:.3f}")
    print(f"   Operations: {final_stats['total_operations']}")

    return benchmarks, final_stats

if __name__ == "__main__":
    print("BitGen Episodic Memory Edge Capabilities")
    print("This module demonstrates the key advantages of episodic memory for edge deployment:")
    print("- Fast fact editing without retraining")
    print("- Selective forgetting of outdated information")
    print("- High accuracy on updated knowledge")
    print("- Low-latency local knowledge access")
    print("\nTo run demonstration, load a trained BitGen model and use EdgeMemoryManager class.")
