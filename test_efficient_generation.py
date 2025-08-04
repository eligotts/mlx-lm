#!/usr/bin/env python3
"""
Simple script to test and compare performance between original and efficient generation.
"""

import time
import argparse
from mlx_lm.utils import load
from mlx_lm.inference_server import batch_generate_detailed, batch_generate_efficient


def test_generation_performance(model_path, batch_size=4, max_tokens=50, num_runs=3):
    """Test both generation methods and compare performance."""
    
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    
    # Create test prompts with varying complexity to test early stopping
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about the ocean.",
        "List three benefits of exercise.",
        "Describe the process of photosynthesis.",
        "What are the main causes of climate change?",
        "How does a computer processor work?",
        "What is the meaning of life?",
    ]
    
    # Select prompts based on batch size
    prompts = test_prompts[:batch_size]
    
    print(f"\nTesting with batch_size={batch_size}, max_tokens={max_tokens}")
    print(f"Prompts: {prompts}")
    print("=" * 80)
    
    # Test original method
    print("\n🔴 ORIGINAL METHOD (batch_generate_detailed)")
    print("-" * 50)
    
    original_times = []
    original_results = None
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        start_time = time.perf_counter()
        
        result = batch_generate_detailed(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_tokens=max_tokens,
            verbose=True,
            format_prompts=False,
            return_logprobs=True,
            temp=0.7,
        )
        
        end_time = time.perf_counter()
        runtime = end_time - start_time
        original_times.append(runtime)
        
        if run == 0:  # Save first run results for comparison
            original_results = result
        
        print(f"Total runtime: {runtime:.3f}s")
    
    avg_original_time = sum(original_times) / len(original_times)
    print(f"\nOriginal method average time: {avg_original_time:.3f}s")
    
    # Test efficient method
    print("\n🟢 EFFICIENT METHOD (batch_generate_efficient)")
    print("-" * 50)
    
    efficient_times = []
    efficient_results = None
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        start_time = time.perf_counter()
        
        result = batch_generate_efficient(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_tokens=max_tokens,
            verbose=True,
            format_prompts=False,
            return_logprobs=True,
            temp=0.7,
        )
        
        end_time = time.perf_counter()
        runtime = end_time - start_time
        efficient_times.append(runtime)
        
        if run == 0:  # Save first run results for comparison
            efficient_results = result
        
        print(f"Total runtime: {runtime:.3f}s")
    
    avg_efficient_time = sum(efficient_times) / len(efficient_times)
    print(f"\nEfficient method average time: {avg_efficient_time:.3f}s")
    
    # Compare results
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"Original method avg time:  {avg_original_time:.3f}s")
    print(f"Efficient method avg time: {avg_efficient_time:.3f}s")
    
    if avg_efficient_time < avg_original_time:
        speedup = avg_original_time / avg_efficient_time
        time_saved = avg_original_time - avg_efficient_time
        print(f"🚀 Speedup: {speedup:.2f}x faster ({time_saved:.3f}s saved)")
    else:
        slowdown = avg_efficient_time / avg_original_time
        time_lost = avg_efficient_time - avg_original_time
        print(f"⚠️  Slowdown: {slowdown:.2f}x slower ({time_lost:.3f}s lost)")
    
    # Validate results are similar
    print("\n🔍 RESULT VALIDATION")
    print("-" * 30)
    
    if original_results and efficient_results:
        # Check that we get similar number of tokens
        orig_total_tokens = sum(len(seq) for seq in original_results['token_ids'])
        eff_total_tokens = sum(len(seq) for seq in efficient_results['token_ids'])
        
        print(f"Original total tokens: {orig_total_tokens}")
        print(f"Efficient total tokens: {eff_total_tokens}")
        
        if 'steps_taken' in efficient_results:
            print(f"Efficient steps taken: {efficient_results['steps_taken']}/{max_tokens}")
        
        if 'finish_reasons' in efficient_results:
            early_stops = sum(1 for reason in efficient_results['finish_reasons'] if reason == "stop")
            print(f"Early stops in efficient method: {early_stops}/{batch_size}")
        
        # Show first response comparison
        print(f"\nFirst response comparison:")
        print(f"Original:  '{original_results['responses'][0][:100]}...'")
        print(f"Efficient: '{efficient_results['responses'][0][:100]}...'")
    
    return {
        'original_avg_time': avg_original_time,
        'efficient_avg_time': avg_efficient_time,
        'speedup': avg_original_time / avg_efficient_time if avg_efficient_time > 0 else 0,
        'original_results': original_results,
        'efficient_results': efficient_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test efficient generation performance")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the MLX model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for testing (default: 4)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs to average (default: 3)"
    )
    
    args = parser.parse_args()
    
    print("🧪 EFFICIENT GENERATION PERFORMANCE TEST")
    print("=" * 80)
    
    results = test_generation_performance(
        model_path=args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs
    )
    
    print(f"\n✅ TEST COMPLETE")
    print(f"Speedup achieved: {results['speedup']:.2f}x")


if __name__ == "__main__":
    main()