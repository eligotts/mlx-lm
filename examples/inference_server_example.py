#!/usr/bin/env python3
"""
Example client for the MLX RL Inference Server.

This demonstrates how to interact with the inference server for RL training workflows.
"""

import asyncio
import json
import aiohttp
from typing import List, Dict, Any


class InferenceClient:
    """Client for interacting with the MLX RL Inference Server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 128,
        policy_version: int = None,
        step_id: int = None,
        request_id: str = None,
    ) -> Dict[str, Any]:
        """Generate text from a single prompt."""
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "policy_version": policy_version,
            "step_id": step_id,
            "request_id": request_id,
        }
        
        async with self.session.post(f"{self.base_url}/generate", json=payload) as resp:
            return await resp.json()
    
    async def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 128,
    ) -> List[Dict[str, Any]]:
        """Generate text from multiple prompts."""
        payload = {
            "prompt": prompts,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        async with self.session.post(f"{self.base_url}/generate", json=payload) as resp:
            result = await resp.json()
            return result.get("results", [result])
    
    async def load_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """Load a new LoRA adapter."""
        payload = {"adapter_path": adapter_path}
        
        async with self.session.post(f"{self.base_url}/load_adapter", json=payload) as resp:
            return await resp.json()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        async with self.session.get(f"{self.base_url}/metrics") as resp:
            return await resp.json()


async def main():
    """Example usage of the inference client."""
    
    async with InferenceClient() as client:
        # Example 1: Single prompt generation
        print("=== Single Prompt Generation ===")
        result = await client.generate(
            prompt="What is machine learning?",
            temperature=0.7,
            max_tokens=50,
            policy_version=1,
            step_id=100,
        )
        print(f"Generated text: {result['text']}")
        print(f"Policy version: {result['policy_version']}")
        print(f"Finish reason: {result['finish_reason']}")
        print()
        
        # Example 2: Batch generation (will be automatically batched by server)
        print("=== Batch Generation ===")
        prompts = [
            "Tell me about reinforcement learning",
            "What is a neural network?",
            "Explain gradient descent",
        ]
        
        # Submit multiple requests concurrently
        tasks = [
            client.generate(prompt, temperature=0.5, max_tokens=30)
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            print(f"Prompt {i+1}: {prompts[i][:30]}...")
            print(f"Response: {result['text'][:50]}...")
            print(f"Batch size: {result['metadata']['batch_size']}")
            print()
        
        # Example 3: Load a new adapter
        print("=== Loading Adapter ===")
        # This would load a new LoRA adapter if the path exists
        # result = await client.load_adapter("/path/to/adapter")
        # print(f"Adapter loaded: {result}")
        
        # Example 4: Get server metrics
        print("=== Server Metrics ===")
        metrics = await client.get_metrics()
        print(f"Current policy version: {metrics['current_policy_version']}")
        print(f"Queue depth: {metrics['queue_depth']}")
        print(f"Max batch size: {metrics['max_batch_size']}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())