#!/usr/bin/env python3

"""
Simple test script for batch_generate_detailed EOS token handling.
Tests how the function deals with EOS tokens and what tokens look like after EOS.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mlx_lm'))

from mlx_lm.utils import load
from mlx_lm.inference_server import batch_generate_efficient

def test_eos_handling():
    print("=== Testing batch_generate_detailed EOS handling ===\n")
    
    # Load a small model for testing
    print("Loading model...")
    try:
        # Try to load a small model - you can change this path to any model you have
        model, tokenizer = load("/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen3-0.6B-MLX-bf16")  # or any model you have
        print(f"✓ Loaded model. EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please adjust the model path in the script to a model you have locally.")
        return
    
    # Test prompts that are likely to generate EOS tokens
    test_prompts = [
        "Hello, my name is /no_think",
        "The capital of France is /no_think",
        "2 + 2 equals /no_think"
    ]
    
    print(f"\nTesting with prompts: {test_prompts}")
    print(f"Max tokens: 20")
    print("-" * 50)
    
    # Call batch_generate_detailed
    result = batch_generate_efficient(
        model=model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        max_tokens=200,  # Small number to increase chance of EOS
        verbose=False,
        format_prompts=True,  # Don't apply chat formatting
        return_logprobs=False,
        return_full_logits=False,
        temp=0.1  # Low temp for more predictable outputs
    )
    
    # Analyze results
    responses = result['responses']
    token_ids = result['token_ids']
    
    for i, (prompt, response, tokens) in enumerate(zip(test_prompts, responses, token_ids)):
        print(f"\n--- Prompt {i+1}: '{prompt}' ---")
        print(f"Raw generated tokens: {tokens}")
        print(f"Number of tokens generated: {len(tokens)}")
        print(f"Decoded response: '{response}'")
        
        # Check if EOS token was generated
        eos_in_tokens = tokenizer.eos_token_id in tokens
        print(f"EOS token ({tokenizer.eos_token_id}) present in tokens: {eos_in_tokens}")
        
        if eos_in_tokens:
            eos_position = tokens.index(tokenizer.eos_token_id)
            print(f"EOS token position: {eos_position}")
            print(f"Tokens before EOS: {tokens[:eos_position]}")
            print(f"Tokens after EOS: {tokens[eos_position+1:]}")
            
            # Decode different parts
            print(f"Decoded before EOS: '{tokenizer.decode(tokens[:eos_position])}'")
            if tokens[eos_position+1:]:
                print(f"Decoded after EOS: '{tokenizer.decode(tokens[eos_position+1:])}'")
            print(f"Full raw decode (with EOS): '{tokenizer.decode(tokens)}'")
        else:
            print("No EOS token found - generation likely hit max_tokens limit")
        
        print("-" * 30)
    
    # Test with higher max_tokens to force EOS generation
    print(f"\n=== Testing with higher max_tokens (50) to encourage EOS ===")
    
    result2 = batch_generate_efficient(
        model=model,
        tokenizer=tokenizer,
        prompts=["Complete this sentence: The quick brown fox"],
        max_tokens=500,
        verbose=False,
        format_prompts=True,
        return_logprobs=False,
        return_full_logits=False,
        temp=0.1
    )
    
    tokens = result2['token_ids'][0]
    response = result2['responses'][0]
    
    print(f"Prompt: 'Complete this sentence: The quick brown fox'")
    print(f"Generated tokens: {tokens}")
    print(f"Length: {len(tokens)}")
    print(f"Response: '{response}'")
    print(f"EOS present: {tokenizer.eos_token_id in tokens}")
    
    if tokenizer.eos_token_id in tokens:
        eos_pos = tokens.index(tokenizer.eos_token_id)
        print(f"EOS at position: {eos_pos}")
        print(f"Tokens after EOS: {tokens[eos_pos+1:]} (should be empty due to stripping)")

if __name__ == "__main__":
    test_eos_handling()