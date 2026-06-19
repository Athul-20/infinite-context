import os
import time
from infinite_context.gateway import ContextGateway

def generate_test_document() -> str:
    print("Generating structured test document...")
    doc = ["# ENTERPRISE SYSTEM SPECIFICATIONS\n\n"]
    
    # Needle 1: Beginning
    doc.append("## 1. ALPHA PROTOCOL\n")
    doc.append("The Alpha Protocol dictates that if the primary server fails, the backup must wait exactly 14 seconds before taking over. The exact phrase to trigger manual override is: 'ECLIPSE-VANGUARD-77'.\n\n")
    
    # Filler
    for i in range(20):
        doc.append(f"Standard operational logging parameter block {i}: System nominal. CPU usage at {(i % 50) + 10}%. Memory stable.\n")
        
    # Needle 2: Middle
    doc.append("\n## 2. DATABASE CONFIGURATION\n")
    doc.append("The main database connection timeout is configured to exactly 42 seconds in production. There are three secret override codes used by admins: 1. OMEGA-9, 2. DELTA-4, 3. SIGMA-1.\n\n")

    # More Filler
    for i in range(20):
        doc.append(f"Network telemetry data frame {i}: Packet loss 0.0{i%10}%. Latency {(i % 20) + 5}ms. Routing optimal.\n")
        
    # Needle 3: End
    doc.append("\n## 3. PROJECT ORION\n")
    doc.append("Project Orion is scheduled for deployment on October 24th, 2027. The project lead is Dr. Aris Thorne. Quote this exact rule: 'Under no circumstances should Project Orion be exposed to the public internet without the Neural Firewall enabled.'\n")
    
    return "".join(doc)

def run_evaluations():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found.")
        return

    # Using skip_compression=True to pass the entire document cleanly to Groq 
    # to evaluate how the Cloud Path handles exact extraction natively
    gateway = ContextGateway(
        engine="cloud", 
        model_id="llama-3.1-8b-instant", 
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1/chat/completions",
        skip_compression=True 
    )

    document = generate_test_document()
    print(f"Document size: {len(document)} characters.\n")
    
    prompts = [
        ("Standard Prompt", "What is the deployment date for Project Orion and who is the lead?"),
        ("Tricky Prompt", "List the three secret admin override codes in reverse order (starting from 3)."),
        ("Exact Extraction", "Quote the exact phrase required to trigger the manual override for the Alpha Protocol.")
    ]

    for name, prompt in prompts:
        print(f"--- {name} ---")
        print(f"Q: {prompt}")
        start = time.time()
        try:
            response = gateway.chat(prompt, document)
            duration = time.time() - start
            print(f"A: {response}")
            print(f"[Time taken: {duration:.2f}s]\n")
        except Exception as e:
            print(f"API Error: {e}\n")

if __name__ == "__main__":
    run_evaluations()
