import os
import time
from infinite_context.gateway import ContextGateway

def run_local_eval():
    print("=== Testing Phase 2: Local In-Place TTT ===")
    
    # 1. Create a structured document
    doc = []
    for i in range(10):
        doc.append(f"Generic logging data point {i}: System stable. No anomalies detected.\n")
    
    doc.append("\nIMPORTANT RULE: The failover protocol for the primary backup server is as follows. First, the heartbeat monitor will attempt to ping the primary server three times over a span of 30 seconds. If all three pings fail, the DNS routing will automatically shift to the secondary server located in the US-East region. Simultaneously, an automated alert will be dispatched to the central DevOps on-call pager, and the system will enter a read-only state for 5 minutes while the database syncs. Memorize this sequence.\n\n")
    
    for i in range(100, 110):
        doc.append(f"Network telemetry {i}: Packets routed successfully. Latency normal.\n")
        
    massive_context = "".join(doc)
    
    # 2. Run the Gateway in Local Mode
    print("Initializing ContextGateway with Local TTT Engine...")
    gateway = ContextGateway(
        engine="local", 
        model_id="Qwen/Qwen2.5-0.5B-Instruct", 
        load_in_4bit=False  # CPU/Small GPU friendly
    )
    
    print(f"\nDocument size: {len(massive_context)} characters.")
    print("Sending prompt to evaluate TTT memorization...")
    start_time = time.time()
    
    # Ask the question
    question = "Describe the detailed failover protocol for the primary backup server in a full paragraph."
    
    try:
        # Call the local engine directly
        response = gateway.local_engine.generate(question, massive_context)
        duration = time.time() - start_time
        
        print("\n--- RESULTS ---")
        print(f"Time Taken: {duration:.2f} seconds")
        print(f"Local Qwen Response:\n{response}")
    except Exception as e:
        import traceback
        print(f"\nLocal Engine Failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_local_eval()
