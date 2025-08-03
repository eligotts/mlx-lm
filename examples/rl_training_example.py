#!/usr/bin/env python3
"""
Example showing how an RL training environment would interact with the inference server.

This demonstrates the key patterns:
1. One-step off-policy generation
2. Tracking policy versions
3. Computing advantages from logprobs
4. Updating LoRA adapters after optimization
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import aiohttp


class RLTrainer:
    """Simulated RL trainer that interacts with the inference server."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.current_policy_version = 0
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_rollouts(
        self, 
        prompts: List[str], 
        step: int
    ) -> List[Dict[str, Any]]:
        """Generate rollouts for a batch of prompts."""
        tasks = []
        
        for i, prompt in enumerate(prompts):
            payload = {
                "prompt": prompt,
                "temperature": 0.8,
                "max_tokens": 100,
                "logprobs": 1,  # Need logprobs for GRPO
                "policy_version": self.current_policy_version,
                "step_id": step,
                "request_id": f"step{step}-prompt{i}"
            }
            
            task = self.session.post(f"{self.server_url}/generate", json=payload)
            tasks.append(task)
        
        # Gather all responses
        responses = await asyncio.gather(*[task.__aenter__() for task in tasks])
        results = []
        
        for resp in responses:
            result = await resp.json()
            results.append(result)
            await resp.__aexit__(None, None, None)
            
        return results
    
    def compute_rewards(self, rollouts: List[Dict[str, Any]]) -> List[float]:
        """Compute rewards for generated rollouts."""
        # In real RL, this would be your reward function
        # For demo, we'll use simple heuristics
        rewards = []
        
        for rollout in rollouts:
            text = rollout['text']
            
            # Example reward: longer responses with proper punctuation
            length_reward = min(len(text.split()) / 20.0, 1.0)
            punctuation_reward = 0.5 if any(p in text for p in ['.', '!', '?']) else 0.0
            
            # Penalize very short or cut-off responses
            if rollout['finish_reason'] == 'length':
                completion_penalty = -0.2
            else:
                completion_penalty = 0.1
                
            reward = length_reward + punctuation_reward + completion_penalty
            rewards.append(reward)
            
        return rewards
    
    def compute_advantages(
        self, 
        rollouts: List[Dict[str, Any]], 
        rewards: List[float],
        baseline: float = 0.5
    ) -> List[float]:
        """Compute advantages for GRPO using rewards and baseline."""
        # Simple advantage: reward - baseline
        # In real GRPO, you might use GAE or other methods
        advantages = [r - baseline for r in rewards]
        return advantages
    
    async def update_policy(
        self, 
        rollouts: List[Dict[str, Any]], 
        advantages: List[float]
    ) -> str:
        """Simulate policy update and return new adapter path."""
        # In real training:
        # 1. Use rollouts['logprobs'] and advantages to compute gradients
        # 2. Update LoRA adapter weights
        # 3. Save new adapter to disk
        
        print("  Computing policy gradients...")
        
        # Simulate gradient computation
        total_logprob = sum(
            sum(rollout.get('logprobs', []))
            for rollout in rollouts
        )
        mean_advantage = np.mean(advantages)
        
        print(f"  Mean advantage: {mean_advantage:.3f}")
        print(f"  Total logprob: {total_logprob:.3f}")
        
        # Simulate saving new adapter
        # In reality, you'd save actual LoRA weights here
        adapter_path = f"/tmp/adapter_v{self.current_policy_version + 1}"
        
        # Create dummy adapter structure
        Path(adapter_path).mkdir(exist_ok=True)
        (Path(adapter_path) / "adapter_config.json").write_text(
            json.dumps({
                "adapter_type": "lora",
                "version": self.current_policy_version + 1,
                "mean_advantage": float(mean_advantage)
            })
        )
        
        return adapter_path
    
    async def load_new_adapter(self, adapter_path: str) -> int:
        """Load new adapter to inference server."""
        try:
            payload = {"adapter_path": adapter_path}
            async with self.session.post(
                f"{self.server_url}/load_adapter", 
                json=payload
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    new_version = result['policy_version']
                    self.current_policy_version = new_version
                    return new_version
                else:
                    print(f"  Note: Adapter loading returned {resp.status}")
                    # In testing, this is expected without real weights
                    self.current_policy_version += 1
                    return self.current_policy_version
        except Exception as e:
            print(f"  Note: Adapter loading failed (expected in test): {e}")
            self.current_policy_version += 1
            return self.current_policy_version
    
    async def training_loop(self, num_steps: int = 3):
        """Main RL training loop."""
        print("Starting RL Training Loop")
        print("=" * 50)
        
        # Training prompts (in real scenario, these would come from your dataset)
        training_prompts = [
            "Explain the policy gradient theorem in simple terms:",
            "What are the advantages of using LoRA for fine-tuning?",
            "Describe the exploration-exploitation tradeoff in RL:",
            "How does value function approximation work in deep RL?",
        ]
        
        # Keep track of metrics
        all_rewards = []
        all_advantages = []
        
        for step in range(num_steps):
            print(f"\n=== Training Step {step} ===")
            
            # 1. Generate rollouts with current policy
            print("Generating rollouts...")
            batch_prompts = training_prompts[step % 2::2]  # Alternate batches
            rollouts = await self.generate_rollouts(batch_prompts, step)
            
            print(f"Generated {len(rollouts)} rollouts")
            for i, rollout in enumerate(rollouts):
                print(f"  Rollout {i}: {len(rollout['completion_ids'])} tokens, "
                      f"policy_v{rollout['policy_version']}")
            
            # 2. Compute rewards
            rewards = self.compute_rewards(rollouts)
            all_rewards.extend(rewards)
            print(f"Rewards: {[f'{r:.2f}' for r in rewards]}")
            
            # 3. Compute advantages
            baseline = np.mean(all_rewards) if all_rewards else 0.5
            advantages = self.compute_advantages(rollouts, rewards, baseline)
            all_advantages.extend(advantages)
            print(f"Advantages: {[f'{a:.2f}' for a in advantages]}")
            
            # 4. Update policy (simulate gradient update)
            print("\nUpdating policy...")
            new_adapter_path = await self.update_policy(rollouts, advantages)
            
            # 5. Load new adapter to inference server
            print(f"Loading new adapter: {new_adapter_path}")
            new_version = await self.load_new_adapter(new_adapter_path)
            print(f"Policy updated to version {new_version}")
            
            # Clean up temp adapter
            import shutil
            shutil.rmtree(new_adapter_path, ignore_errors=True)
        
        # Summary statistics
        print("\n" + "=" * 50)
        print("Training Summary:")
        print(f"  Total steps: {num_steps}")
        print(f"  Final policy version: {self.current_policy_version}")
        print(f"  Mean reward: {np.mean(all_rewards):.3f}")
        print(f"  Mean advantage: {np.mean(all_advantages):.3f}")


async def main():
    """Run the RL training example."""
    server_url = "http://localhost:8000"
    
    print("RL Training Example")
    print("==================")
    print(f"Server URL: {server_url}")
    print("\nMake sure the inference server is running!")
    print("Start with: python -m mlx_lm.inference_server --model <model_path>\n")
    
    # Check server health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/health") as resp:
                health = await resp.json()
                if not health.get('model_loaded'):
                    print("Error: Server is running but model is not loaded!")
                    return
                print("✓ Server is healthy and ready\n")
    except Exception as e:
        print(f"Error: Cannot connect to server: {e}")
        print("Please start the inference server first.")
        return
    
    # Run training loop
    async with RLTrainer(server_url) as trainer:
        await trainer.training_loop(num_steps=3)


if __name__ == "__main__":
    asyncio.run(main())