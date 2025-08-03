# Copyright © 2023-2024 Apple Inc.

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .generate import generate_step, stream_generate
from .models.cache import make_prompt_cache, trim_prompt_cache
from .sample_utils import make_logits_processors, make_sampler
from .tokenizer_utils import TokenizerWrapper
from .tuner.utils import load_adapters
from .utils import load


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
        """Process a batch of requests using individual generation for each prompt."""
        start_time = time.time()
        
        try:
            # Process each request individually using the existing stream_generate
            # This approach maintains compatibility with MLX's generation infrastructure
            # while allowing us to batch at the request level
            results = []
            
            for req in batch:
                # Get sampling parameters
                options = req.options
                temperature = options.get("temperature", 0.0)
                top_p = options.get("top_p", 1.0)
                top_k = options.get("top_k", 0)
                min_p = options.get("min_p", 0.0)
                repetition_penalty = options.get("repetition_penalty", 1.0)
                repetition_context_size = options.get("repetition_context_size", 20)
                max_tokens = options.get("max_tokens", self.max_tokens)
                logprobs_count = options.get("logprobs", 0)
                stop_sequences = options.get("stop", [])
                
                # Set up sampling
                sampler = make_sampler(
                    temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
                
                logits_processors = make_logits_processors(
                    None,  # logit_bias
                    repetition_penalty,
                    repetition_context_size,
                )
                
                # Tokenize prompt
                prompt_tokens = self.tokenizer.encode(req.prompt_text, add_special_tokens=True)
                prompt_array = mx.array(prompt_tokens)
                
                # Track generation
                completion_tokens = []
                completion_logprobs = []
                completion_text = ""
                finish_reason = "length"
                
                # Generate using stream_generate for proper MLX integration
                for response in stream_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt_array,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    prompt_cache=self.prompt_cache,
                ):
                    # Track token and logprobs
                    if response.token is not None:
                        completion_tokens.append(response.token)
                        if logprobs_count > 0 and response.logprobs is not None:
                            # Get the logprob for the generated token
                            token_logprob = response.logprobs[response.token].item()
                            completion_logprobs.append(token_logprob)
                    
                    # Update text
                    completion_text = response.text
                    
                    # Check stop sequences
                    if stop_sequences:
                        for stop_seq in stop_sequences:
                            if stop_seq in completion_text:
                                finish_reason = "stop"
                                # Trim the stop sequence from the text
                                completion_text = completion_text[:completion_text.find(stop_seq)]
                                break
                    
                    # Check for finish reason
                    if response.finish_reason:
                        finish_reason = response.finish_reason
                        break
                
                # Store result
                result = {
                    "request_id": req.request_id,
                    "policy_version": req.policy_version or self.current_policy_version,
                    "step_id": req.step_id,
                    "prompt_ids": prompt_tokens,
                    "completion_ids": completion_tokens,
                    "text": completion_text,
                    "logprobs": completion_logprobs if logprobs_count > 0 else None,
                    "finish_reason": finish_reason,
                    "metadata": {
                        "batch_size": len(batch),
                        "latency_ms": (time.time() - start_time) * 1000,
                        "current_policy_version": self.current_policy_version,
                    },
                }
                
                results.append((req, result))
            
            # Fulfill all futures
            for req, result in results:
                req.future.set_result(result)
                
        except Exception as e:
            # On error, reject all requests in batch
            logging.error(f"Batch processing error: {e}")
            for req in batch:
                req.future.set_exception(e)


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
                try:
                    # Increment policy version
                    self.batcher.current_policy_version += 1
                    
                    # Load adapter weights into the model
                    logging.info(f"Loading adapter from {adapter_path}")
                    
                    # Create a fresh copy of the base model with the new adapter
                    # This ensures thread safety and allows hot-swapping
                    updated_model = load_adapters(self.model, str(adapter_path))
                    updated_model.eval()
                    
                    # Update the model in the batcher
                    self.batcher.model = updated_model
                    self.model = updated_model
                    
                    # Reinitialize the prompt cache for the updated model
                    self.batcher._init_cache()
                    
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