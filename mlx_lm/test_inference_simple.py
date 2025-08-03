#!/usr/bin/env python3
"""
Simple test script for the MLX RL Inference Server using standard library only.

This version uses urllib instead of aiohttp for easier testing without dependencies.
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any


def post_request(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a POST request and return JSON response."""
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"HTTP Error {e.code}: {error_body}")
        raise


def get_request(url: str) -> Dict[str, Any]:
    """Make a GET request and return JSON response."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode('utf-8'))


def test_single_prompt():
    """Test a single prompt generation."""
    print("\n=== Single Prompt Test ===")
    
    data = {
        "prompt": "The three laws of robotics are:",
        "temperature": 0.7,
        "max_tokens": 50,
        "logprobs": 1,
        "policy_version": 1,
        "step_id": 100,
        "request_id": "test-001"
    }
    
    result = post_request("http://localhost:8000/generate", data)
    
    print(f"Prompt: {data['prompt']}")
    print(f"Generated: {result['text']}")
    print(f"Tokens generated: {len(result.get('completion_ids', []))}")
    if result.get('logprobs'):
        print(f"Sample logprobs: {result['logprobs'][:5]}...")
    print(f"Policy version: {result.get('policy_version')}")
    print(f"Batch size: {result['metadata']['batch_size']}")


def test_rl_training_loop():
    """Simulate a simple RL training loop."""
    print("\n=== Simulated RL Training Loop ===")
    
    # Simulate 3 training steps
    for step in range(3):
        print(f"\n--- Step {step} ---")
        
        # Generate responses for this step's prompts
        prompts = [
            f"Step {step}: Define reinforcement learning as",
            f"Step {step}: The reward function in RL is"
        ]
        
        for i, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "temperature": 0.8,
                "max_tokens": 30,
                "logprobs": 1,
                "policy_version": step,
                "step_id": step * 100 + i,
                "request_id": f"step{step}-req{i}"
            }
            
            print(f"\nSending prompt {i+1}...")
            result = post_request("http://localhost:8000/generate", data)
            
            print(f"Response: {result['text'][:50]}...")
            print(f"Logprobs available: {'Yes' if result.get('logprobs') else 'No'}")
            
            # In real RL, you would:
            # 1. Compute rewards for this completion
            # 2. Calculate advantages using logprobs
            # 3. Update policy parameters
            
        # Simulate time for policy update
        time.sleep(0.1)
        print("(Simulated policy update completed)")


def test_metrics():
    """Check server metrics."""
    print("\n=== Server Metrics ===")
    
    metrics = get_request("http://localhost:8000/metrics")
    
    for key, value in metrics.items():
        print(f"{key}: {value}")


def main():
    """Run simple tests."""
    base_url = "http://localhost:8000"
    
    print("Simple MLX RL Inference Server Test")
    print("===================================")
    print(f"Server URL: {base_url}")
    print("\nMake sure the server is running with:")
    print("  python -m mlx_lm.inference_server --model <model_path>\n")
    
    # Check server health
    try:
        health = get_request(f"{base_url}/health")
        if health.get('model_loaded'):
            print(f"✓ Server is healthy and model is loaded")
        else:
            print("✗ Server is running but model is not loaded!")
            return
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("Please start the inference server first.")
        return
    
    # Run tests
    try:
        test_single_prompt()
        test_rl_training_loop()
        test_metrics()
        
        print("\n=== Tests Completed Successfully ===")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")


if __name__ == "__main__":
    main()