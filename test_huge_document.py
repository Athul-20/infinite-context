import os
import time
from infinite_context.gateway import ContextGateway

def test_phase_one_compression():
    print("=== Testing Phase 1: Massive Document Compression ===")
    
    # 1. Create a dummy "massive" document locally if it doesn't exist
    doc_path = "huge_document.txt"
    if not os.path.exists(doc_path):
        print(f"Generating {doc_path} (approx 2MB of repeating text)...")
        with open(doc_path, "w", encoding="utf-8") as f:
            base_text = "This is a standard line of a massive PDF or codebase. It contains generic information meant to bloat the context size to test the Headroom compression layer. "
            for i in range(15000):
                f.write(base_text + f" Line {i}.\n")
            # Inject the needle at the very end
            f.write("\n\nCRITICAL SYSTEM ARCHITECTURE DETAIL: The main database connection timeout is configured to exactly 42 seconds in production.\n")
    
    with open(doc_path, "r", encoding="utf-8") as f:
        massive_context = f.read()
        
    print(f"Loaded massive document: {len(massive_context)} characters.")
    
    # 2. Run the Gateway in Cloud Mode (Compression Enabled)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is missing. Cannot test.")
        return
        
    print("Initializing ContextGateway with Headroom compression...")
    # Using compression_ratio=0.5 (or default)
    gateway = ContextGateway(
        engine="cloud", 
        model_id="llama-3.1-8b-instant", 
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1/chat/completions",
        compression_ratio=0.5,
        skip_compression=False 
    )
    
    print("Sending prompt to extract the hidden needle...")
    start_time = time.time()
    
    # Ask the question
    question = "What is the main database connection timeout configured to in production? Answer briefly."
    try:
        response = gateway.chat(question, massive_context)
        duration = time.time() - start_time
        
        print("\n--- RESULTS ---")
        print(f"Time Taken: {duration:.2f} seconds")
        print(f"Groq Response:\n{response}")
    except Exception as e:
        print(f"\nAPI Call Failed: {e}")

if __name__ == "__main__":
    test_phase_one_compression()
