#!/usr/bin/env python3
"""
Test script for the MLX RL Inference Server.

This script mimics the workflow of an RL training environment:
1. Sends prompt requests with metadata
2. Receives completions with logprobs
3. Simulates LoRA weight updates
4. Demonstrates one-step off-policy pattern
"""

import asyncio
import json
import time
import aiohttp
from pathlib import Path
from typing import List, Dict, Any


async def test_single_generation(session: aiohttp.ClientSession, base_url: str):
    """Test single prompt generation."""
    print("\n=== Testing Single Generation ===")
    
    prompt = "The capital of France is"
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 20,
        "logprobs": 1,  # Request logprobs for GRPO
        "policy_version": 1,
        "step_id": 100,
        "request_id": "test-001"
    }
    
    async with session.post(f"{base_url}/generate", json=payload) as resp:
        result = await resp.json()
        
    print(f"Prompt: {prompt}")
    print(f"Generated: {result['text']}")
    print(f"Tokens: {len(result['completion_ids'])}")
    if result.get('logprobs'):
        print(f"First 5 logprobs: {result['logprobs'][:5]}")
    print(f"Finish reason: {result['finish_reason']}")
    print(f"Batch size: {result['metadata']['batch_size']}")
    

async def test_batch_behavior(session: aiohttp.ClientSession, base_url: str):
    """Test automatic batching behavior."""
    print("\n=== Testing Batch Behavior ===")
    
    # Create multiple prompts that will be batched
    prompts = [
        "What is reinforcement learning?",
        "Explain neural networks in simple terms:",
        "The key to successful ML training is",
        "Gradient descent works by",
        "The transformer architecture includes"
    ]
    
    # Send all requests concurrently - server will batch them
    tasks = []
    for i, prompt in enumerate(prompts):
        payload = {
            "prompt": prompt,
            "temperature": 0.5,
            "max_tokens": 30,
            "logprobs": 1,
            "policy_version": 1,
            "step_id": 200 + i,
            "request_id": f"batch-{i:03d}"
        }
        task = session.post(f"{base_url}/generate", json=payload)
        tasks.append(task)
    
    # Wait for all responses
    responses = await asyncio.gather(*[task.__aenter__() for task in tasks])
    results = []
    for resp in responses:
        result = await resp.json()
        results.append(result)
        await resp.__aexit__(None, None, None)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i][:40]}...")
        print(f"Response: {result['text'][:50]}...")
        print(f"Batch size: {result['metadata']['batch_size']}")
        print(f"Request ID: {result['request_id']}")


async def test_one_step_offpolicy(session: aiohttp.ClientSession, base_url: str):
    """Test one-step off-policy pattern used in RL training."""
    print("\n=== Testing One-Step Off-Policy Pattern ===")
    
    # Simulate training loop with one batch ahead
    num_steps = 3
    batch_size = 2
    
    for step in range(num_steps):
        print(f"\n--- Training Step {step} ---")
        
        # Generate prompts for this step
        prompts = [
            f"Step {step}: Tell me about policy gradient methods",
            f"Step {step}: What is the value function in RL?"
        ]
        
        # Submit generation requests (these run while we process previous batch)
        print(f"Submitting {len(prompts)} prompts for generation...")
        generation_tasks = []
        
        for i, prompt in enumerate(prompts):
            payload = {
                "prompt": prompt,
                "temperature": 0.8,
                "max_tokens": 50,
                "logprobs": 1,
                "policy_version": step,  # Current policy version
                "step_id": step * 100,
                "request_id": f"step{step}-prompt{i}"
            }
            task = session.post(f"{base_url}/generate", json=payload)
            generation_tasks.append(task)
        
        # Simulate processing previous batch (computing advantages, updating policy)
        if step > 0:
            print("Processing previous batch (computing GRPO advantages)...")
            await asyncio.sleep(0.1)  # Simulate computation time
        
        # Collect generation results
        responses = await asyncio.gather(*[task.__aenter__() for task in generation_tasks])
        results = []
        for resp in responses:
            result = await resp.json()
            results.append(result)
            await resp.__aexit__(None, None, None)
        
        # Display results
        for i, result in enumerate(results):
            print(f"\nGenerated {i+1}:")
            print(f"  Text: {result['text'][:60]}...")
            print(f"  Tokens: {len(result['completion_ids'])}")
            print(f"  Policy version: {result['policy_version']}")
            
        # Simulate policy update would happen here
        await asyncio.sleep(0.05)


async def test_lora_update(session: aiohttp.ClientSession, base_url: str):
    """Test LoRA adapter update workflow."""
    print("\n=== Testing LoRA Adapter Update ===")
    
    # Note: This requires an actual adapter file to exist
    # In real training, this would be the newly trained adapter
    adapter_path = "/tmp/test_adapter"
    
    # Create a dummy adapter directory for testing
    adapter_dir = Path(adapter_path)
    adapter_dir.mkdir(exist_ok=True)
    
    # In real scenario, you'd have actual adapter weights here
    # For testing, we'll just create the directory structure
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "adapter_type": "lora",
        "rank": 16
    }))
    
    print(f"Simulating LoRA update from: {adapter_path}")
    print("(In real training, this would contain actual trained weights)")
    
    # Get current metrics
    async with session.get(f"{base_url}/metrics") as resp:
        metrics = await resp.json()
        old_version = metrics['current_policy_version']
        print(f"Current policy version: {old_version}")
    
    # Attempt to load adapter (will fail without real weights, but shows the flow)
    try:
        payload = {"adapter_path": str(adapter_path)}
        async with session.post(f"{base_url}/load_adapter", json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"Adapter loaded successfully!")
                print(f"New policy version: {result['policy_version']}")
            else:
                error = await resp.text()
                print(f"Note: Adapter loading failed (expected without real weights)")
                print(f"Error: {error}")
    except Exception as e:
        print(f"Note: Adapter loading failed (expected without real weights)")
        print(f"Error: {e}")
    
    # Clean up
    import shutil
    shutil.rmtree(adapter_dir, ignore_errors=True)


async def test_stop_sequences(session: aiohttp.ClientSession, base_url: str):
    """Test stop sequence handling."""
    print("\n=== Testing Stop Sequences ===")
    
    payload = {
        "prompt": "Count from 1 to 10: 1, 2, 3,",
        "temperature": 0.1,
        "max_tokens": 50,
        "stop": [", 7", "10"],  # Stop at either sequence
        "policy_version": 1,
        "step_id": 500
    }
    
    async with session.post(f"{base_url}/generate", json=payload) as resp:
        result = await resp.json()
    
    print(f"Prompt: {payload['prompt']}")
    print(f"Generated: {result['text']}")
    print(f"Finish reason: {result['finish_reason']}")


async def test_server_metrics(session: aiohttp.ClientSession, base_url: str):
    """Test server metrics endpoint."""
    print("\n=== Testing Server Metrics ===")
    
    async with session.get(f"{base_url}/metrics") as resp:
        metrics = await resp.json()
    
    print("Server Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


async def main():
    """Run all tests."""
    base_url = "http://localhost:8000"
    
    print(f"Testing MLX RL Inference Server at {base_url}")
    print("Make sure the server is running with:")
    print("  python -m mlx_lm.inference_server --model <model_path>")
    
    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                health = await resp.json()
                if not health.get('model_loaded'):
                    print("\nError: Server is running but model is not loaded!")
                    return
                print(f"\nServer is healthy: {health}")
    except aiohttp.ClientConnectorError:
        print(f"\nError: Cannot connect to server at {base_url}")
        print("Please start the inference server first.")
        return
    
    # Run tests
    async with aiohttp.ClientSession() as session:
        try:
            # Test individual features
            await test_single_generation(session, base_url)
            await test_batch_behavior(session, base_url)
            await test_stop_sequences(session, base_url)
            await test_server_metrics(session, base_url)
            
            # Test RL training patterns
            await test_one_step_offpolicy(session, base_url)
            # await test_lora_update(session, base_url)
            
            print("\n=== All Tests Completed ===")
            
        except Exception as e:
            print(f"\nError during testing: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())