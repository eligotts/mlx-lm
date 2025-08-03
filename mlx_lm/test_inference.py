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
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear, LoRAEmbedding, LoRASwitchLinear
from mlx_lm.tuner.dora import DoRALinear, DoRAEmbedding
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear
from mlx_lm.utils import load
from mlx.utils import tree_flatten, tree_unflatten


def linear_to_lora_layers(
    model: nn.Module,
    num_layers: int,
    config: Dict,
    use_dora: bool = False,
):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        num_layers (int): The number of blocks to convert to lora layers
        starting from the last layer.
        config (dict): More configuration parameters for LoRA, including the
          rank, scale, and optional layer keys.
        use_dora (bool): If True, uses DoRA instead of LoRA.
          Default: ``False``
    """

    def to_lora(layer):
        if not use_dora and hasattr(layer, "to_lora"):
            return layer.to_lora(
                r=config["rank"],
                scale=config["scale"],
                dropout=config["dropout"],
            )

        if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            LoRALayer = DoRALinear if use_dora else LoRALinear
        elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
            if use_dora:
                raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
            LoRALayer = LoRASwitchLinear
        elif isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
            LoRALayer = DoRAEmbedding if use_dora else LoRAEmbedding
        else:
            raise ValueError(
                f"Can't convert layer of type {type(layer).__name__} to LoRA"
            )

        return LoRALayer.from_base(
            layer,
            r=config["rank"],
            scale=config["scale"],
            dropout=config["dropout"],
        )

    keys = config.get("keys", None)
    if keys is not None:
        keys = set(keys)
    elif model.model_type in [
        "mistral",
        "mistral3",
        "llama",
        "phi",
        "mixtral",
        "nemotron",
        "stablelm",
        "hunyuan",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "phimoe",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "granite",
        "helium",
        "starcoder2",
        "cohere",
        "cohere2",
        "minicpm",
        "minicpm3",
        "minicpm4",
        "deepseek",
        "olmo2",
        "olmoe",
        "internlm3",
        "glm4",
        "mimo",
        "ernie4_5",
        "dots1",
        "smollm3",
    ]:
        keys = set(["self_attn.q_proj", "self_attn.v_proj"])
        if model.model_type in ["mixtral", "phimoe"]:
            keys.add("block_sparse_moe.gate")
        if model.model_type == "qwen2_moe":
            keys.add("mlp.gate")
            keys.add("mlp.shared_expert_gate")
        if model.model_type in ["olmoe", "qwen3_moe", "dots1"]:
            keys.add("mlp.gate")

    elif model.model_type == "gpt_bigcode":
        keys = set(["attn.c_attn"])
    elif model.model_type == "gpt2":
        keys = set(["attn.c_attn"])
    elif model.model_type == "gpt_neox":
        keys = set(["attention.query_key_value"])
    elif model.model_type == "olmo":
        keys = set(["att_proj"])
    elif model.model_type == "openelm":
        keys = set(["attn.qkv_proj"])
    elif model.model_type == "phi3":
        keys = set(["self_attn.qkv_proj"])
    elif model.model_type == "phi-msft":
        keys = set(["mixer.Wqkv", "moe.gate"])
    elif model.model_type == "dbrx":
        keys = set(["norm_attn_norm.attn.Wqkv", "ffn.router.layer"])
    elif model.model_type == "internlm2":
        keys = set(["attention.wqkv", "attention.wo"])
    elif model.model_type == "deepseek_v2" or model.model_type == "minicpm3":
        keys = set(
            [
                "self_attn.q_proj",
                "self_attn.q_a_proj",
                "self_attn.q_b_proj",
                "self_attn.kv_a_proj_with_mqa",
                "self_attn.kv_b_proj",
            ]
        )
    elif model.model_type == "mamba":
        keys = set(
            [
                "mixer.in_proj",
                "mixer.x_proj",
                "mixer.dt_proj",
                "mixer.out_proj",
            ]
        )
    elif model.model_type == "exaone":
        keys = set(["attn.attention.q_proj", "attn.attention.v_proj"])
    else:
        raise ValueError(f"Lora does not support {model.model_type}")

    for l in model.layers[-max(num_layers, 0) :]:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
        if lora_layers:
            l.update_modules(tree_unflatten(lora_layers))

    lora_modules = [(k, to_lora(m)) for k, m in model.named_modules() if k in keys]
    if lora_modules:
        model.update_modules(tree_unflatten(lora_modules))


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
    
    # Initialize a small model for testing
    model_name = "/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-0.6B-MLX-bf16"
    adapter_path = "/tmp/test_lora_adapter"
    adapter_path_updated = "/tmp/test_lora_adapter_updated"
    
    print(f"Loading base model: {model_name}")
    
    # First, let's do an initial generation before LoRA to get baseline
    print("\n1. Testing generation with base model...")
    test_prompt = "The meaning of life is"
    
    payload = {
        "prompt": test_prompt,
        "temperature": 0.7,
        "max_tokens": 30,
        "logprobs": 1,
        "policy_version": 0,
        "request_id": "test-lora-base"
    }
    
    async with session.post(f"{base_url}/generate", json=payload) as resp:
        if resp.status != 200:
            error_text = await resp.text()
            print(f"Error generating with base model: Status {resp.status}")
            print(f"Error message: {error_text}")
            return  # Exit early if base generation fails
        base_result = await resp.json()
        print(f"Base model response: {base_result['text'][:100]}...")
    
    # Now let's create and load a LoRA adapter
    print("\n2. Creating LoRA adapter...")
    
    # Load the model locally to create LoRA adapter
    try:
        model, tokenizer = load(model_name)
        model.freeze()
        
        # Convert to LoRA
        num_layers = len(model.layers)
        lora_parameters = {"rank": 16, "dropout": 0.0, "scale": 10.0}
        
        linear_to_lora_layers(
            model=model,
            num_layers=num_layers,
            config=lora_parameters,
            use_dora=False,
        )
        
        # Save initial adapter weights
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        
        # Create adapter directory
        adapter_dir = Path(adapter_path)
        adapter_dir.mkdir(exist_ok=True, parents=True)
        
        # Save adapter config in the format expected by load_adapters
        adapter_config = {
            "model_type": model.model_type,
            "num_layers": num_layers,
            "lora_parameters": lora_parameters,
            "fine_tune_type": "lora",
            "target_modules": ["self_attn.q_proj", "self_attn.v_proj"],
            "trainable": True
        }
        
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
        
        # Save initial adapter weights
        mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), adapter_weights)
        
        print(f"Initial adapter saved to: {adapter_path}")
        print(f"Number of trainable parameters: {len(adapter_weights)}")
        
        # Upload the initial adapter using multipart form data
        print("\n3. Uploading initial LoRA adapter...")
        
        # Prepare files for upload
        with open(adapter_dir / "adapters.safetensors", "rb") as weights_file:
            weights_data = weights_file.read()
        with open(adapter_dir / "adapter_config.json", "rb") as config_file:
            config_data = config_file.read()
        
        # Create multipart form data
        form_data = aiohttp.FormData()
        form_data.add_field('adapter_weights', weights_data, 
                          filename='adapters.safetensors',
                          content_type='application/octet-stream')
        form_data.add_field('adapter_config', config_data,
                          filename='adapter_config.json', 
                          content_type='application/json')
        
        async with session.post(f"{base_url}/upload_adapter", data=form_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"Initial adapter uploaded successfully!")
                print(f"Policy version: {result['policy_version']}")
            else:
                error = await resp.text()
                print(f"Failed to upload initial adapter: {error}")
                
        # Test generation with initial adapter
        print("\n4. Testing generation with initial LoRA adapter...")
        payload = {
            "prompt": test_prompt,
            "temperature": 0.7,
            "max_tokens": 30,
            "logprobs": 1,
            "policy_version": 1,
            "request_id": "test-lora-initial"
        }
        
        async with session.post(f"{base_url}/generate", json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"Error generating with initial LoRA adapter: Status {resp.status}")
                print(f"Error message: {error_text}")
                lora_result = {"text": "ERROR: Failed to generate"}
            else:
                lora_result = await resp.json()
                print(f"Initial LoRA response: {lora_result['text'][:100]}...")
        
        # Now modify the adapter weights to simulate training
        print("\n5. Modifying adapter weights (simulating training)...")
        
        # Randomly modify some weights
        modified_weights = {}
        for key, value in adapter_weights.items():
            # Add random noise to simulate training changes
            noise = mx.random.normal(shape=value.shape, scale=0.1)
            modified_weights[key] = value + noise
        
        # Save modified adapter
        adapter_dir_updated = Path(adapter_path_updated)
        adapter_dir_updated.mkdir(exist_ok=True, parents=True)
        
        # Copy config
        with open(adapter_dir / "adapter_config.json", "r") as f:
            config = json.load(f)
        with open(adapter_dir_updated / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save modified weights
        mx.save_safetensors(str(adapter_dir_updated / "adapters.safetensors"), modified_weights)
        
        print(f"Modified adapter saved to: {adapter_path_updated}")
        
        # Upload the modified adapter
        print("\n6. Uploading modified LoRA adapter...")
        
        # Prepare files for upload
        with open(adapter_dir_updated / "adapters.safetensors", "rb") as weights_file:
            weights_data = weights_file.read()
        with open(adapter_dir_updated / "adapter_config.json", "rb") as config_file:
            config_data = config_file.read()
        
        # Create multipart form data
        form_data = aiohttp.FormData()
        form_data.add_field('adapter_weights', weights_data,
                          filename='adapters.safetensors',
                          content_type='application/octet-stream')
        form_data.add_field('adapter_config', config_data,
                          filename='adapter_config.json',
                          content_type='application/json')
        
        async with session.post(f"{base_url}/upload_adapter", data=form_data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"Modified adapter uploaded successfully!")
                print(f"New policy version: {result['policy_version']}")
            else:
                error = await resp.text()
                print(f"Failed to upload modified adapter: {error}")
        
        # Test generation with modified adapter
        print("\n7. Testing generation with modified LoRA adapter...")
        payload = {
            "prompt": test_prompt,
            "temperature": 0.7,
            "max_tokens": 30,
            "logprobs": 1,
            "policy_version": 2,
            "request_id": "test-lora-modified"
        }
        
        async with session.post(f"{base_url}/generate", json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"Error generating with modified LoRA adapter: Status {resp.status}")
                print(f"Error message: {error_text}")
                modified_result = {"text": "ERROR: Failed to generate"}
            else:
                modified_result = await resp.json()
                print(f"Modified LoRA response: {modified_result['text'][:100]}...")
            
        # Compare results
        print("\n8. Comparison of outputs:")
        print(f"Base model: {base_result['text'][:50]}...")
        print(f"Initial LoRA: {lora_result['text'][:50]}...")
        print(f"Modified LoRA: {modified_result['text'][:50]}...")
        print("\nNote: Modified LoRA should produce different (possibly garbled) output due to random weight changes.")
        
    except Exception as e:
        print(f"Error during LoRA test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        import shutil
        for path in [adapter_path, adapter_path_updated]:
            if Path(path).exists():
                shutil.rmtree(path, ignore_errors=True)
                print(f"Cleaned up: {path}")


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
            # await test_single_generation(session, base_url)
            # await test_batch_behavior(session, base_url)
            # await test_stop_sequences(session, base_url)
            # await test_server_metrics(session, base_url)
            
            # Test RL training patterns
            # await test_one_step_offpolicy(session, base_url)
            await test_lora_update(session, base_url)
            
            print("\n=== All Tests Completed ===")
            
        except Exception as e:
            print(f"\nError during testing: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())