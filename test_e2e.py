import time
from infinite_context.gateway import ContextGateway
import shutil
import os

def run_checkpoint_test():
    print("=== Testing Phase 2: Local In-Place TTT with Context Checkpointing ===")
    
    doc = []
    doc.append("SYSTEM MANUAL: The protocol sequence is Alpha, Bravo, Charlie. Memorize this sequence.\n\n")
    for i in range(10):
        doc.append(f"Network telemetry {i}: Packets routed successfully.\n")
        
    massive_context = "".join(doc)
    checkpoint_dir = "./my_session_checkpoint"
    
    print("\n--- Part 1: Initializing Gateway and Memorizing ---")
    gateway = ContextGateway(
        engine="local", 
        model_id="Qwen/Qwen2.5-0.5B-Instruct", 
        load_in_4bit=False
    )
    
    start_time = time.time()
    question = "What is the protocol sequence?"
    print(f"Asking question: '{question}'")
    
    # We pass keep_state=True to prevent the engine from deleting the LoRA adapter!
    response1 = gateway.chat(question, massive_context, keep_state=True)
    duration1 = time.time() - start_time
    
    print("\n--- Part 1 RESULTS ---")
    print(f"Time Taken (including Reading & Training): {duration1:.2f} seconds")
    print(f"Response: {response1}")
    
    print("\n--- Part 2: Saving the State to Disk ---")
    gateway.save_state(checkpoint_dir)
    print("Deleting Gateway from RAM...")
    del gateway
    
    print("\n--- Part 3: Resuming Chat Later (Loading State) ---")
    print("Re-initializing a fresh Gateway...")
    gateway_resumed = ContextGateway(
        engine="local", 
        model_id="Qwen/Qwen2.5-0.5B-Instruct", 
        load_in_4bit=False
    )
    
    print("Loading previous state from disk...")
    gateway_resumed.load_state(checkpoint_dir)
    
    start_time2 = time.time()
    question2 = "Repeat the protocol sequence again."
    print(f"Asking question: '{question2}'")
    
    # Notice we pass an EMPTY context because the knowledge is already baked in!
    # And we don't need to keep state this time since it's the end of the test.
    response2 = gateway_resumed.chat(question2, context="")
    duration2 = time.time() - start_time2
    
    print("\n--- Part 3 RESULTS ---")
    print(f"Time Taken (0 Reading, Instant Generation): {duration2:.2f} seconds")
    print(f"Response: {response2}")
    
    # Cleanup
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    run_checkpoint_test()
