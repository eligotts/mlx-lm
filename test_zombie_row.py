#!/usr/bin/env python3
"""
Test script for the zombie-row efficient generation implementation.
"""

import time
from mlx_lm.utils import load
from mlx_lm.inference_server import batch_generate_detailed, batch_generate_efficient


def test_zombie_row_generation():
    """Test the zombie-row implementation."""
    
    model_path = "/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-0.6B-MLX-bf16"
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    
    # Test prompts designed to finish at different times
    test_prompts = [
        "What is 2+2? Answer: /no_think",  # Should be short
        "Explain the theory of relativity in three concise sentences /no_think",  # Should be longer
        "List three colors: red, /no_think",  # Should be medium
        # "Once upon a time, /no_think",  # Story beginning, could be long
    ]
    
    print(f"\nTesting zombie-row efficient generation")
    print(f"Prompts: {test_prompts}")
    print("=" * 80)
    
    # Test efficient generation
    print("\n🟢 ZOMBIE-ROW EFFICIENT METHOD")
    print("-" * 50)
    
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    start_time = time.perf_counter()
    
    result = batch_generate_efficient(
        model=model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        max_tokens=300,
        verbose=True,
        format_prompts=True,
        return_logprobs=True,
        temp=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    end_time = time.perf_counter()
    efficient_time = end_time - start_time
    
    print(f"\nTotal time: {efficient_time:.2f}s")
    
    # Verify results
    print("\n🔍 VERIFICATION")
    print("-" * 50)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response: {result['responses'][i][:100]}...")
        print(f"Tokens generated: {len(result['token_ids'][i])}")
        print(f"Logprobs generated: {len(result['logprobs'][i])}")
        print(f"Finish reason: {result['finish_reasons'][i]}")
        
        # Check for zombie tokens (repeated EOS)
        tokens = result['token_ids'][i]
        if len(tokens) > 0:
            # Check if there are any EOS tokens in the middle (shouldn't be)
            eos_positions = [j for j, tok in enumerate(tokens) if tok == tokenizer.eos_token_id]
            if eos_positions:
                print(f"WARNING: Found EOS token at positions: {eos_positions}")
    
    # Compare with original method
    print("\n\n🔴 ORIGINAL METHOD (for comparison)")
    print("-" * 50)
    
    start_time = time.perf_counter()
    
    result_orig = batch_generate_detailed(
        model=model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        max_tokens=300,
        verbose=False,
        format_prompts=True,
        return_logprobs=True,
        temp=0.7,
    )
    
    end_time = time.perf_counter()
    original_time = end_time - start_time
    
    print(f"Original time: {original_time:.2f}s")
    print(f"Speedup: {original_time/efficient_time:.2f}x")
    
    # Compare outputs
    print("\n📊 OUTPUT COMPARISON")
    print("-" * 50)
    
    for i in range(len(test_prompts)):
        eff_len = len(result['token_ids'][i])
        orig_len = len(result_orig['token_ids'][i])
        
        print(f"\nSequence {i+1}:")
        print(f"  Efficient length: {eff_len} tokens")
        print(f"  Original length: {orig_len} tokens")
        
        # Check if outputs are similar (first few tokens)
        min_len = min(eff_len, orig_len, 10)
        if min_len > 0:
            eff_start = result['responses'][i][:50]
            orig_start = result_orig['responses'][i][:50]
            print(f"  Efficient start: {eff_start}...")
            print(f"  Original start: {orig_start}...")
            
            if eff_start != orig_start:
                print("  ⚠️  Outputs differ!")


if __name__ == "__main__":
    test_zombie_row_generation()