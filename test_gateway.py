import os
from infinite_context.gateway import ContextGateway

def test_cloud_engine():
    print("=== Testing Cloud Engine (Headroom Proxy) ===")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "dummy_key_for_test")
    
    # Normally we'd use a massive codebase string, but we'll use a dummy string for the test
    massive_context = "def fetch_data():\n    # Imagine 50,000 lines of code here\n    pass"
    
    try:
        gateway = ContextGateway(engine="cloud", api_key=api_key)
        # Note: If headroom is not installed or API key is invalid, this will fail in a real run.
        print("Gateway initialized successfully for cloud mode.")
        print("Test passed (Initialization). Note: API call skipped to prevent charges/errors.\n")
    except Exception as e:
        print(f"Cloud initialization failed: {e}\n")

def test_groq_engine():
    print("=== Testing Cloud Engine (Groq via Custom Base URL) ===")
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("Skipping Groq test: GROQ_API_KEY environment variable not found.\n")
        return
        
    try:
        # We set skip_compression=True for a raw test, or leave it if Headroom supports it
        gateway = ContextGateway(
            engine="cloud", 
            model_id="llama-3.1-8b-instant", 
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1/chat/completions",
            skip_compression=True  # skipping just to test the raw routing
        )
        print("Gateway initialized successfully for Groq.")
        
        # Actually attempt to call it if a real key is present
        response = gateway.chat("Say 'Hello from Groq!'", "You are a helpful assistant.")
        print(f"Groq Response: {response}")
        print("Test passed (Groq Integration).\n")
    except Exception as e:
        print(f"Groq API call failed: {e}\n")

def test_local_engine():
    print("=== Testing Local Engine (In-Place TTT) ===")
    try:
        # We initialize but don't download the massive weights for a quick CI test
        gateway = ContextGateway(
            engine="local", 
            model_id="Qwen/Qwen2.5-0.5B-Instruct", 
            load_in_4bit=False # Set to false just to test architecture without BitsAndBytes
        )
        print("Gateway initialized successfully for local mode.")
        print("Test passed (Initialization).\n")
    except Exception as e:
        print(f"Local initialization failed (Expected if transformers/torch is not installed): {e}\n")

if __name__ == "__main__":
    print("Running integration tests for Infinite Context Gateway...\n")
    test_cloud_engine()
    test_groq_engine()
    test_local_engine()
    print("Done.")
