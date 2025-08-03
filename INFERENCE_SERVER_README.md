# MLX RL Inference Server

A high-performance asynchronous inference server designed for reinforcement learning training workflows using MLX models.

## Features

- **Automatic Request Batching**: Transparently batches individual requests up to a configurable batch size (default: 5) with a flush timeout (default: 10ms)
- **LoRA Adapter Hot-Reload**: Dynamically load new LoRA adapter weights without restarting the server
- **Metadata Tracking**: Track policy versions, step IDs, and request IDs for proper RL training coordination
- **Token-Level Logprobs**: Return per-token log probabilities for GRPO advantage computation
- **FastAPI-Based**: Built on FastAPI for high performance async handling
- **One-Step Off-Policy**: Designed to work with training environments that maintain one batch ahead

## Installation

```bash
# Install additional dependencies
pip install -r requirements-inference-server.txt
```

## Usage

### Starting the Server

```bash
python -m mlx_lm.inference_server \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --host 0.0.0.0 \
    --port 8000 \
    --max-batch 5 \
    --flush-interval-ms 10 \
    --max-tokens 128
```

### Command Line Arguments

- `--model`: Path to MLX model (required)
- `--adapter-path`: Optional initial LoRA adapter path
- `--host`: Server host (default: 0.0.0.0)
- `--port`: Server port (default: 8000)
- `--max-batch`: Maximum batch size (default: 5)
- `--flush-interval-ms`: Flush timeout in milliseconds (default: 10)
- `--max-tokens`: Maximum tokens to generate (default: 128)
- `--trust-remote-code`: Enable trusting remote code for tokenizer
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## API Endpoints

### POST /generate

Generate completions for one or more prompts.

**Request:**
```json
{
    "prompt": "What is machine learning?",
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 128,
    "logprobs": 1,
    "policy_version": 42,
    "step_id": 1000,
    "request_id": "unique-request-id"
}
```

**Response:**
```json
{
    "request_id": "unique-request-id",
    "policy_version": 42,
    "step_id": 1000,
    "prompt_ids": [1, 2, 3, ...],
    "completion_ids": [4, 5, 6, ...],
    "text": "Machine learning is...",
    "logprobs": [-0.23, -1.45, ...],
    "finish_reason": "stop",
    "metadata": {
        "batch_size": 3,
        "latency_ms": 142.5,
        "current_policy_version": 42
    }
}
```

### POST /load_adapter

Load a new LoRA adapter from disk.

**Request:**
```json
{
    "adapter_path": "/path/to/adapter"
}
```

**Response:**
```json
{
    "status": "success",
    "policy_version": 43,
    "message": "Loaded adapter from /path/to/adapter"
}
```

### POST /upload_adapter

Upload and load a new LoRA adapter.

**Request:** Multipart form with adapter file

**Response:** Same as `/load_adapter`

### GET /health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### GET /metrics

Get server metrics.

**Response:**
```json
{
    "current_policy_version": 42,
    "queue_depth": 2,
    "max_batch_size": 5,
    "flush_interval_ms": 10
}
```

## Example Client Usage

See `examples/inference_server_example.py` for a complete example client implementation.

```python
import asyncio
from inference_client import InferenceClient

async def main():
    async with InferenceClient("http://localhost:8000") as client:
        # Generate text
        result = await client.generate(
            prompt="What is reinforcement learning?",
            temperature=0.7,
            max_tokens=100,
            policy_version=1,
            step_id=100
        )
        print(f"Generated: {result['text']}")
        
        # Load new adapter
        adapter_result = await client.load_adapter("/path/to/adapter")
        print(f"New policy version: {adapter_result['policy_version']}")

asyncio.run(main())
```

## Integration with RL Training

The server is designed to work seamlessly with RL training loops:

1. **Trainer** sends prompts with metadata (policy_version, step_id)
2. **Server** batches requests automatically up to max_batch size
3. **Server** returns completions with logprobs and metadata
4. **Trainer** can hot-reload new LoRA adapters after optimization steps
5. **Trainer** uses policy_version to detect and handle stale rollouts

## Performance Considerations

- Batch size of 5 is optimal for M-series Macs based on the requirements
- 10ms flush interval provides good balance between latency and throughput
- Use one-step off-policy pattern: submit next batch while processing current
- Monitor queue depth via `/metrics` to detect backpressure

## Architecture

The server uses a request batcher that:
- Collects incoming requests in a queue
- Groups them into batches up to `max_batch` size
- Flushes partial batches after `flush_interval_ms`
- Processes each batch using MLX's stream_generate
- Returns individual responses with batch metadata

This design hides batching complexity from the trainer while maximizing throughput.