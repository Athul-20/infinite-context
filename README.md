# Infinite Context

A hybrid AI context gateway combining **Headroom cloud compression** and **In-Place Local Test-Time Training (TTT)**.

## Installation

To install the base gateway:
```bash
pip install infinite_context
```

To install with Cloud Phase 1 dependencies (Headroom API):
```bash
pip install infinite_context[cloud]
```

To install with Local Phase 2 dependencies (PyTorch, Transformers, PEFT):
```bash
pip install infinite_context[local]
```

## Usage

### Phase 1: Cloud Compression
Uses the Headroom API to semantically compress a massive context and send it to cloud providers (OpenAI, Anthropic) while keeping costs low.

```python
from infinite_context import ContextGateway

gateway = ContextGateway(
    engine="cloud",
    model_id="claude-3-5-sonnet-20240620",
    api_key="your_anthropic_api_key",
    compression_ratio=0.8
)

# You can pass in conversational history (rolling window)
history = [
    {"role": "user", "content": "What is the codebase about?"},
    {"role": "assistant", "content": "It is a Python application..."}
]

response = gateway.chat("How does the failover protocol work?", massive_context, history=history)
print(response)
```

### Phase 2: Local Test-Time Training (TTT)
Bypasses the KV-Cache entirely by injecting a PEFT LoRA adapter and baking the context directly into the model's neural weights on your local GPU. Includes Early Stopping latency optimizations and Generation repetition penalties.

```python
from infinite_context import ContextGateway

gateway = ContextGateway(
    engine="local",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    load_in_4bit=True
)

response = gateway.chat("What is the failover protocol?", massive_context)
print(response)
```

### Checkpoint Persistence
Save trained Fast Weights to disk and resume later without re-reading the document:

```python
from infinite_context import ContextGateway

# Train and keep state
gateway = ContextGateway(engine="local", model_id="Qwen/Qwen2.5-0.5B-Instruct")
gateway.chat("Summarise the document.", massive_context, keep_state=True)
gateway.save_state("./my_checkpoint")

del gateway  # Free all GPU memory

# Resume later — zero re-training
gateway = ContextGateway(engine="local", model_id="Qwen/Qwen2.5-0.5B-Instruct")
gateway.load_state("./my_checkpoint")
response = gateway.chat("What was the protocol?", context="")
print(response)
```
