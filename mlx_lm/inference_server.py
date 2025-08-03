# Copyright © 2023-2024 Apple Inc.

import argparse
import asyncio
import copy
import inspect
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from .generate import generate_step
from .models.cache import make_prompt_cache, trim_prompt_cache
from .tokenizer_utils import TokenizerWrapper
from .tuner.utils import load_adapters
from .utils import load


def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


class BatchedKVCache:

    def __init__(self, head_dim, n_kv_heads, batch_size=1):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            shape = (self.batch_size, self.n_kv_heads, n_steps * self.step, self.head_dim)
            new_k = mx.zeros(shape, keys.dtype)
            new_v = mx.zeros(shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def top_p_sampling(logits: mx.array, top_p: float, temperature: float, axis: int = -1) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion.
    """
    # Apply temperature and compute softmax
    probs = mx.softmax(logits / temperature, axis=axis)
    
    # Sort probs in descending order
    sorted_indices = mx.argsort(-probs, axis=axis)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=axis)
    
    # Compute cumulative probabilities
    cumulative_probs = mx.cumsum(sorted_probs, axis=axis)
    
    # Create a mask for probs above the threshold
    mask = cumulative_probs <= top_p
    
    # Apply the mask to the sorted probabilities
    masked_probs = sorted_probs * mask
    
    # Normalize the masked probabilities
    normalized_probs = masked_probs / mx.sum(masked_probs, axis=axis, keepdims=True)
    
    # Sample from the normalized probabilities
    sampled_indices = mx.random.categorical(mx.log(normalized_probs), axis=axis)
    
    # Gather the original token indices
    tokens = mx.take_along_axis(sorted_indices, mx.expand_dims(sampled_indices, axis=axis), axis=axis)
    
    return tokens


def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        generated_tokens (any): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.
    """

    if len(generated_tokens) > 0:
        indices = mx.array([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits


def _infer_head_dim(model):
    """
    Return the attention head dimension for *this* model implementation.
    Works on both old and new mlx-lm versions.
    """
    # 1. Newer mlx-lm keeps it on the first attention layer
    attn = model.layers[0].self_attn
    if hasattr(attn, "head_dim"):
        return attn.head_dim

    # 2. Derive from q_proj weight shape: out_features == n_heads * head_dim
    n_heads = attn.n_heads
    out_feats = attn.q_proj.weight.shape[0]   # (out_features, in_features)
    return out_feats // n_heads


def _infer_kv_heads(model):
    # new versions: grab the value from every layer
    if hasattr(model, "n_kv_heads"):
        heads = model.n_kv_heads
        return [heads] * len(model.layers) if isinstance(heads, int) else heads
    # fall-back: per-layer
    return [layer.self_attn.n_kv_heads for layer in model.layers]


def generate_step_detailed(
    prompts: mx.array,
    model: nn.Module,
    return_logprobs: bool = True,
    return_full_logits: bool = False,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    eos_token_id: Optional[int] = None,
) -> Generator[Tuple[mx.array, Optional[mx.array], Optional[mx.array]], None, None]:
    """
    Generate tokens step by step with efficient per-token log probability computation.
    
    This uses the minimal overhead approach: log p(x) = logit(x) - logsumexp(all_logits)
    
    Args:
        prompts: Input token array
        model: The language model
        return_logprobs: Whether to return log probabilities
        return_full_logits: Whether to return full logits
        temp: Temperature for sampling
        repetition_penalty: Repetition penalty factor
        repetition_context_size: Context size for repetition penalty
        top_p: Top-p sampling parameter
        logit_bias: Logit bias dictionary
    
    Yields:
        Tuple of (tokens, logprobs, full_logits)
    """
    
    def sample_efficient(logits: mx.array) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Efficient sampling with log probabilities.
        Returns: (tokens, log_probs) where both are (bs, 1) shape
        """
        # --- up-cast to higher precision for numerics ---
        logits_f32 = logits.astype(mx.float32)
        
        # Optional logit_bias (unchanged)
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values

        # Use bf16 for sampling (fast) but keep logits_f32 for maths below
        if temp == 0:
            tokens = mx.argmax(logits, axis=-1, keepdims=True)
        else:
            if 0.0 < top_p < 1.0:
                tokens = top_p_sampling(logits, top_p, temp)
            else:
                scaled = logits * (1 / temp)
                tokens = mx.random.categorical(scaled, axis=-1)
                tokens = mx.expand_dims(tokens, axis=-1)

        # ---------- NEW: Efficient log p for the chosen token ----------
        # log p(x) = logit(x) - log ∑_v exp(logit(v))
        log_probs = None
        if return_logprobs:
            # Use temperature-scaled logits if temp > 0, otherwise raw logits
            computation_logits = logits_f32 * (1 / temp) if temp > 0 else logits_f32
            
            lse = mx.logsumexp(computation_logits, axis=-1, keepdims=True)   # (bs, 1)
            token_logit = mx.take_along_axis(computation_logits, tokens, axis=-1)  # (bs, 1)
            log_probs = token_logit - lse                                    # (bs, 1)  fp32

        return tokens, log_probs

    if repetition_penalty:
        raise NotImplementedError("repetition_penalty not supported.")

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    # (bs, ntoks)
    y = prompts
    
    kv_heads = _infer_kv_heads(model)
    head_dim = _infer_head_dim(model)
    cache = [BatchedKVCache(head_dim, n, y.shape[0]) for n in kv_heads]

    repetition_context = prompts

    if repetition_context_size and repetition_penalty:
        repetition_context = repetition_context[:, -repetition_context_size:]

    def _step(y):
        nonlocal repetition_context
        logits = model(y, cache=cache)
        logits = logits[:, -1, :]                    # (bs, vocab)

        # Store original logits if needed for full_logits return
        step_logits = logits if return_full_logits else None

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )

        tokens, log_p = sample_efficient(logits)     # ← now returns both efficiently
        
        if repetition_penalty:
            repetition_context = mx.concatenate([repetition_context, tokens])

        if repetition_context_size and repetition_context.shape[1] > repetition_context_size:
            repetition_context = repetition_context[:, -repetition_context_size:]

        return tokens, log_p, step_logits

    y, logprobs, logits = _step(y)
    mx.async_eval(y)
    while True:
        next_y, next_logprobs, next_logits = _step(y)

        mx.eval(y)
        yield y, logprobs, logits  # logprobs are now computed efficiently during generation
        y, logprobs, logits = next_y, next_logprobs, next_logits


def batch_generate_detailed(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts: List[str],
    max_tokens: int = 100,
    verbose: bool = False,
    format_prompts: bool = True,
    return_logprobs: bool = True,
    return_full_logits: bool = False,
    **kwargs,
) -> Dict[str, Union[List[str], List[List[int]], List[List[float]], List[mx.array]]]:
    """
    Generate a complete response from the model with efficient streaming log probabilities.
    
    This uses the minimal overhead approach where log probabilities are computed during
    generation with just logsumexp + gather operations - no extra forward passes needed.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompts (List[str]): List of string prompts.
       max_tokens (int): The maximum number of tokens. Default: ``100``.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       format_prompts (bool): Whether to apply chat formatting to prompts.
       return_logprobs (bool): Whether to return per-token log probabilities.
       return_full_logits (bool): Whether to return full logits for each position.
       kwargs: The remaining options get passed to :func:`generate_step_detailed`.
          See :func:`generate_step_detailed` for more details.
    
    Returns:
        Dict containing:
            - 'responses': List of decoded text responses
            - 'token_ids': List of token ID sequences 
            - 'logprobs': List of log probability sequences (if return_logprobs=True)
            - 'full_logits': List of full logits arrays (if return_full_logits=True)
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if verbose:
        print("=" * 10)
    
    if format_prompts:
        prompts_fm = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompts_fm = [tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in prompts_fm]
    else:
        prompts_fm = prompts

    # left-padding for batched generation
    tokenizer._tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer._tokenizer.pad_token = tokenizer.eos_token
        tokenizer._tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts_toks = mx.array(tokenizer._tokenizer(prompts_fm, padding=True)['input_ids'])
    batch_size = prompts_toks.shape[0]
    
    tic = time.perf_counter()

    # Efficient generation with streaming log probabilities
    output_toks = []
    logp_stream = [] if return_logprobs else None    # NEW: collect streaming log probs
    full_logits_list = [] if return_full_logits else None
    
    for (tokens, lp, step_logits), n in zip(
        generate_step_detailed(prompts_toks, model, return_logprobs, return_full_logits, **kwargs),
        range(max_tokens),
    ): 
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        
        output_toks.append(tokens)
        if return_logprobs and lp is not None:
            logp_stream.append(lp)              # NEW: collect log probs
        if return_full_logits and step_logits is not None:
            full_logits_list.append(step_logits)
    
    output_toks = mx.concatenate(output_toks, axis=1)   # (bs, L)
    gen_time = time.perf_counter() - tic
    
    if return_logprobs and logp_stream:
        logp_stream = mx.concatenate(logp_stream, axis=1)   # (bs, L)

    # Prepare return data
    result = {}
    
    # Decode responses and strip pad/eos tokens
    responses = [response.split(tokenizer.eos_token)[0].split(tokenizer.pad_token)[0] 
                for response in tokenizer.batch_decode(output_toks.tolist())]
    result['responses'] = responses
    result['token_ids'] = output_toks.tolist()
    
    if return_logprobs and logp_stream is not None:
        result['logprobs'] = logp_stream.tolist()               # Python list conversion
    
    if return_full_logits and full_logits_list:
        # Stack logits across time steps
        result['full_logits'] = [mx.stack([step_logits[i] for step_logits in full_logits_list]) 
                                for i in range(batch_size)]
    
    if verbose:
        prompt_tps = prompts_toks.size / prompt_time
        gen_tps = output_toks.size / gen_time
        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")
        print("Log prob computation: included in generation (minimal overhead)")
        for prompt, response in zip(prompts, responses):
            print("=" * 10)
            print("Prompt:", prompt)
            print(response)
    
    # Quick sanity check if verbose
    if verbose and return_logprobs and logp_stream is not None:
        print("\nSanity check (first batch, first token):")
        print(f"Token ID: {int(output_toks[0,0])}")
        print(f"Log prob: {float(logp_stream[0,0]):.6f} (should be ≤ 0)")
    
    return result


# Request/Response Models
class GenerationRequest(BaseModel):
    prompt: Union[str, List[str]]
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: Optional[float] = 1.0
    repetition_context_size: int = 20
    max_tokens: int = 128
    logprobs: int = 0
    stop: Optional[Union[str, List[str]]] = None
    # RL-specific metadata
    policy_version: Optional[Union[int, str]] = None
    step_id: Optional[int] = None
    request_id: Optional[str] = None


class LoadAdapterRequest(BaseModel):
    adapter_path: str


@dataclass
class PromptRequest:
    """Internal representation of a prompt request."""
    prompt_text: str
    options: Dict[str, Any]
    future: asyncio.Future
    request_id: str
    policy_version: Optional[Union[int, str]] = None
    step_id: Optional[int] = None


@dataclass
class BatchGenerationResult:
    """Result from batch generation."""
    prompt_ids: List[List[int]]
    completion_ids: List[List[int]]
    completion_texts: List[str]
    logprobs: List[List[float]]
    token_ids: List[List[int]]
    finish_reasons: List[str]
    metrics: Dict[str, float]


class Batcher:
    """Handles automatic batching of requests."""
    
    def __init__(
        self,
        model,
        tokenizer,
        max_batch: int = 5,
        flush_interval_ms: int = 10,
        max_tokens: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.flush_interval = flush_interval_ms / 1000.0
        self.max_tokens = max_tokens
        
        self.queue: List[PromptRequest] = []
        self.lock = asyncio.Lock()
        self._running = True
        self._flush_task = None
        
        # Model versioning for stale rollout detection
        self.current_policy_version = 0
        
        # Cache management
        self.prompt_cache = None
        self._init_cache()
        
        # Generation lock for safe adapter swapping
        self.generation_lock = asyncio.Lock()
        self.active_generations = 0
    
    def _init_cache(self):
        """Initialize prompt cache."""
        self.prompt_cache = make_prompt_cache(self.model)
    
    async def start(self):
        """Start the batcher background task."""
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self):
        """Stop the batcher."""
        self._running = False
        if self._flush_task:
            await self._flush_task
    
    async def submit(
        self,
        prompt_text: str,
        options: Dict[str, Any],
        request_id: Optional[str] = None,
        policy_version: Optional[Union[int, str]] = None,
        step_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Submit a request for batch processing."""
        fut = asyncio.get_event_loop().create_future()
        request_id = request_id or str(uuid.uuid4())
        
        req = PromptRequest(
            prompt_text=prompt_text,
            options=options,
            future=fut,
            request_id=request_id,
            policy_version=policy_version,
            step_id=step_id,
        )
        
        async with self.lock:
            self.queue.append(req)
            # Trigger immediate flush if batch is full
            if len(self.queue) >= self.max_batch:
                batch = self.queue[:self.max_batch]
                self.queue = self.queue[self.max_batch:]
                asyncio.create_task(self._process_batch(batch))
        
        return await fut
    
    async def _flush_loop(self):
        """Background task that flushes the queue periodically."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            async with self.lock:
                if self.queue:
                    batch = self.queue[:self.max_batch]
                    self.queue = self.queue[len(batch):]
                    asyncio.create_task(self._process_batch(batch))
    
    async def _process_batch(self, batch: List[PromptRequest]):
        """Process a batch of requests using true batch generation."""
        start_time = time.time()
        
        # Track active generations for safe adapter swapping
        self.active_generations += 1
        
        try:
            # Collect all prompts and parameters for batch processing
            prompts = [req.prompt_text for req in batch]
            
            # Get sampling parameters from first request (assume all are the same for now)
            # In a more sophisticated version, we might group by parameters
            if batch:
                options = batch[0].options
                temperature = options.get("temperature", 0.0)
                top_p = options.get("top_p", 1.0)
                top_k = options.get("top_k", 0)
                min_p = options.get("min_p", 0.0)
                repetition_penalty = options.get("repetition_penalty", None)
                repetition_context_size = options.get("repetition_context_size", 20)
                max_tokens = options.get("max_tokens", self.max_tokens)
                logprobs_count = options.get("logprobs", 0)
                stop_sequences = options.get("stop", [])
            
            # Prepare kwargs for batch_generate_detailed
            generate_kwargs = {
                "temp": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty if repetition_penalty != 1.0 else None,
                "repetition_context_size": repetition_context_size,
                # Note: top_k and min_p are not directly supported in generate_step_detailed
                # but we could add them if needed
            }
            
            # Use batch_generate_detailed for true batched generation
            batch_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: batch_generate_detailed(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompts=prompts,
                    max_tokens=max_tokens,
                    verbose=False,
                    format_prompts=False,  # We'll handle formatting ourselves
                    return_logprobs=(logprobs_count > 0),
                    return_full_logits=False,
                    **generate_kwargs
                )
            )
            
            # Process results and fulfill futures
            responses = batch_result.get('responses', [])
            token_ids = batch_result.get('token_ids', [])
            logprobs = batch_result.get('logprobs', []) if logprobs_count > 0 else [None] * len(batch)
            
            for i, req in enumerate(batch):
                if i < len(responses):
                    # Get the completion text
                    completion_text = responses[i]
                    completion_ids = token_ids[i] if i < len(token_ids) else []
                    
                    # Handle stop sequences and EOS
                    finish_reason = "length"
                    
                    # Check if EOS token was generated
                    if self.tokenizer.eos_token_id in completion_ids:
                        finish_reason = "stop"
                        # Find the position of EOS token and trim
                        eos_pos = completion_ids.index(self.tokenizer.eos_token_id)
                        completion_ids = completion_ids[:eos_pos]
                        completion_text = self.tokenizer.decode(completion_ids)
                    
                    # Check for custom stop sequences
                    if stop_sequences:
                        for stop_seq in stop_sequences:
                            if stop_seq in completion_text:
                                finish_reason = "stop"
                                completion_text = completion_text[:completion_text.find(stop_seq)]
                                # Trim token IDs to match
                                text_so_far = ""
                                for j in range(len(completion_ids)):
                                    text_so_far = self.tokenizer.decode(completion_ids[:j+1])
                                    if stop_seq in text_so_far:
                                        completion_ids = completion_ids[:j]
                                        break
                                break
                    
                    # Get prompt tokens
                    prompt_tokens = self.tokenizer.encode(req.prompt_text, add_special_tokens=True)
                    
                    # Build result
                    result = {
                        "request_id": req.request_id,
                        "policy_version": req.policy_version or self.current_policy_version,
                        "step_id": req.step_id,
                        "prompt_ids": prompt_tokens,
                        "completion_ids": completion_ids,
                        "text": completion_text,
                        "logprobs": logprobs[i] if logprobs and i < len(logprobs) else None,
                        "finish_reason": finish_reason,
                        "metadata": {
                            "batch_size": len(batch),
                            "latency_ms": (time.time() - start_time) * 1000,
                            "current_policy_version": self.current_policy_version,
                        },
                    }
                    
                    req.future.set_result(result)
                else:
                    # Handle case where we didn't get enough results
                    req.future.set_exception(RuntimeError("Batch generation did not return enough results"))
                
        except Exception as e:
            # On error, reject all requests in batch
            logging.error(f"Batch processing error: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
        finally:
            # Always decrement active generations
            self.active_generations -= 1


class InferenceServer:
    """Main inference server class."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.batcher = None
        self.adapter_lock = asyncio.Lock()
        
        # Initialize FastAPI app
        self.app = FastAPI(title="MLX RL Inference Server")
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        
        @self.app.on_event("startup")
        async def startup():
            """Initialize model and batcher on startup."""
            await self._init_model()
        
        @self.app.on_event("shutdown")
        async def shutdown():
            """Clean up on shutdown."""
            if self.batcher:
                await self.batcher.stop()
        
        @self.app.post("/generate")
        async def generate(request: GenerationRequest):
            """Generate completions for prompts."""
            if not self.batcher:
                raise HTTPException(status_code=503, detail="Server not initialized")
            
            # Handle single or multiple prompts
            prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
            results = []
            
            # Convert stop sequences
            stop_sequences = request.stop or []
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            
            # Submit all prompts
            for prompt in prompts:
                options = {
                    "n": request.n,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "min_p": request.min_p,
                    "repetition_penalty": request.repetition_penalty,
                    "repetition_context_size": request.repetition_context_size,
                    "max_tokens": request.max_tokens,
                    "logprobs": request.logprobs,
                    "stop": stop_sequences,
                }
                
                result = await self.batcher.submit(
                    prompt_text=prompt,
                    options=options,
                    request_id=request.request_id,
                    policy_version=request.policy_version,
                    step_id=request.step_id,
                )
                results.append(result)
            
            # Return single result if single prompt
            if len(results) == 1:
                return JSONResponse(results[0])
            return JSONResponse({"results": results})
        
        @self.app.post("/load_adapter")
        async def load_adapter(request: LoadAdapterRequest):
            """Load a new LoRA adapter."""
            adapter_path = Path(request.adapter_path)
            
            if not adapter_path.exists():
                raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_path}")
            
            async with self.adapter_lock:
                # Wait for all active generations to complete
                wait_start = time.time()
                while self.batcher.active_generations > 0:
                    await asyncio.sleep(0.01)
                    if time.time() - wait_start > 30:  # 30 second timeout
                        raise HTTPException(
                            status_code=503, 
                            detail="Timeout waiting for active generations to complete"
                        )
                
                try:
                    # Increment policy version
                    self.batcher.current_policy_version += 1
                    
                    # Load adapter weights IN-PLACE into the existing model
                    # This avoids creating a duplicate model in memory
                    logging.info(f"Loading adapter from {adapter_path}")
                    
                    # load_adapters modifies the model in-place and returns it
                    # We don't need to create a new model instance
                    load_adapters(self.model, str(adapter_path))
                    self.model.eval()
                    
                    # The model reference in batcher already points to the same model
                    # so we don't need to update it
                    
                    # Reinitialize the prompt cache for the updated model
                    self.batcher._init_cache()
                    
                    # Force garbage collection to free any temporary memory
                    import gc
                    gc.collect()
                    
                    return JSONResponse({
                        "status": "success",
                        "policy_version": self.batcher.current_policy_version,
                        "message": f"Loaded adapter from {adapter_path}"
                    })
                    
                except Exception as e:
                    logging.error(f"Failed to load adapter: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/upload_adapter")
        async def upload_adapter(file: UploadFile = File(...)):
            """Upload and load a new LoRA adapter."""
            temp_path = Path(f"/tmp/adapter_{uuid.uuid4()}.safetensors")
            
            try:
                # Save uploaded file
                content = await file.read()
                with open(temp_path, "wb") as f:
                    f.write(content)
                
                # Load the adapter
                request = LoadAdapterRequest(adapter_path=str(temp_path))
                result = await load_adapter(request)
                
                # Clean up temp file
                temp_path.unlink()
                
                return result
                
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "model_loaded": self.model is not None}
        
        @self.app.get("/metrics")
        async def metrics():
            """Get server metrics."""
            if not self.batcher:
                return {"status": "not_initialized"}
            
            return {
                "current_policy_version": self.batcher.current_policy_version,
                "queue_depth": len(self.batcher.queue),
                "max_batch_size": self.batcher.max_batch,
                "flush_interval_ms": self.batcher.flush_interval * 1000,
            }
    
    async def _init_model(self):
        """Initialize the model and tokenizer."""
        logging.info(f"Loading model from {self.args.model}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = load(
            self.args.model,
            adapter_path=self.args.adapter_path,
            tokenizer_config={"trust_remote_code": self.args.trust_remote_code}
        )
        
        # Initialize batcher
        self.batcher = Batcher(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch=self.args.max_batch,
            flush_interval_ms=self.args.flush_interval_ms,
            max_tokens=self.args.max_tokens,
        )
        
        await self.batcher.start()
        logging.info("Model loaded and batcher started")
    
    def run(self):
        """Run the server."""
        import uvicorn
        uvicorn.run(
            self.app,
            host=self.args.host,
            port=self.args.port,
            log_level=self.args.log_level.lower(),
        )


def main():
    parser = argparse.ArgumentParser(description="MLX RL Inference Server")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the server (default: 8000)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=5,
        help="Maximum batch size (default: 5)",
    )
    parser.add_argument(
        "--flush-interval-ms",
        type=int,
        default=10,
        help="Flush interval in milliseconds (default: 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    server = InferenceServer(args)
    server.run()


if __name__ == "__main__":
    main()