# Infinite Context Gateway

**Infinite Context Gateway** is a production-ready Python library designed to solve the "context window limit" and high API token costs associated with analyzing massive documents. 

Whether you are building RAG pipelines, autonomous agents, or large-scale document analysis tools, this library allows you to bypass standard memory limitations by routing your data through two highly optimized, infinite-context engines:

1. **Cloud Engine (Groq Optimized)**: Uses a highly-optimized local compression algorithm to compress massive documents locally for *free* (shrinking token counts by up to 90%), before routing the smaller payload directly to **Groq's** lightning-fast Llama 3 models. **Use case**: Cost-effective, extremely high-speed document analysis using Groq's LPU inference.
2. **Local Engine**: Bypasses the KV-Cache entirely using **In-Place Test-Time Training (TTT)**. It injects a PEFT DoRA adapter to bake massive documents directly into the neural weights of local open-source models (like Qwen) running on your own GPU. **Use case**: 100% offline, privacy-preserving document reasoning with zero API costs.

---

## Installation

You can install the package directly from PyPI or GitHub. Choose the extras based on which engine you want to use.

**For Cloud Engine (Lightweight, CPU only):**
```bash
pip install "infinite-context-gateway[cloud]"
```

**For Local Engine (Requires NVIDIA GPU & CUDA):**
```bash
pip install "infinite-context-gateway[local]"
```

---

## Usage

### Option 1: Cloud Engine (Local Compression + Groq API)

When you use the Cloud Engine, the gateway instantly compresses your massive documents **locally on your computer for free**. 

It then sends the shrunken document securely to the Groq API to answer your question at lightning speed. Because the document was compressed locally first, you save massive amounts of money on Groq API token costs!

```python
from infinite_context import ContextGateway

# Example: Using Groq's super-fast Llama 3 API
gateway = ContextGateway(
    engine="cloud",
    model_id="llama3-70b-8192",          # Target model on Groq
    api_key="your_groq_api_key_here",    # Your Groq API Key
    base_url="https://api.groq.com/openai/v1/chat/completions",
    compression_ratio=0.8                # Compress the document by 80% locally
)

# Pass in a massive document and ask a question!
massive_document = "..."
response = gateway.chat("What is the failover protocol?", massive_document)
print(response)
```

---

### Option 2: Local Engine (Test-Time Training)

When you use the Local Engine, the gateway bypasses the standard KV-Cache memory limits entirely. 

Instead of putting the document into the prompt, it injects a fast PEFT DoRA adapter into a local model (like Qwen 0.5B) and **trains the document directly into the neural weights** over a few seconds. 

This requires absolutely zero internet connection and no API keys.

```python
from infinite_context import ContextGateway

# Loads the model in 4-bit precision into your GPU
gateway = ContextGateway(
    engine="local",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    load_in_4bit=True
)

response = gateway.chat("What is the failover protocol?", massive_document)
print(response)
```

#### Checkpoint Persistence
Because the local engine alters the neural weights, you can save the trained "Fast Weights" to your hard drive. You can then reload them days later and ask questions about the massive document *without ever reading the document again*.

```python
# 1. Train the model on the document
gateway.chat("Summarise.", massive_document, keep_state=True)

# 2. Save the brain state
gateway.save_state("./memory_checkpoint")

# 3. Reload it later (zero retraining time)
new_gateway = ContextGateway(engine="local", model_id="Qwen/Qwen2.5-0.5B-Instruct")
new_gateway.load_state("./memory_checkpoint")

# Ask a question without passing the context!
answer = new_gateway.chat("What was the protocol?", context="")
```
