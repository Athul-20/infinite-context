import gradio as gr
from infinite_context import ContextGateway

# Initialize the Context Gateway with a lightweight model by default
# On a Space with a GPU, it will automatically use Unsloth optimization.
gateway = ContextGateway(model_id="Qwen/Qwen2.5-0.5B-Instruct")

def ask_question(context: str, question: str) -> str:
    if not context.strip():
        return "Please provide a context document."
    if not question.strip():
        return "Please ask a question."
        
    try:
        # Memorize the context using Test-Time Training (in-place)
        gateway.memorise(context, in_place=True)
        # Ask the question based on the memorized context
        response = gateway.ask(question)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

with gr.Blocks(title="Infinite Context - TTT Memory") as app:
    gr.Markdown("# Infinite Context 🧠")
    gr.Markdown("Inject unlimited context into LLMs using Test-Time Training (TTT) Fast Weights. Paste your large document below, and the model will learn it dynamically.")
    
    with gr.Row():
        with gr.Column(scale=2):
            context_box = gr.Textbox(
                label="Context Document",
                lines=15,
                placeholder="Paste your long document, codebase, or book here...",
            )
        with gr.Column(scale=1):
            question_box = gr.Textbox(
                label="Question",
                lines=3,
                placeholder="What is this document about?",
            )
            submit_btn = gr.Button("Ask", variant="primary")
            output_box = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False,
            )
            
    submit_btn.click(
        fn=ask_question,
        inputs=[context_box, question_box],
        outputs=[output_box],
    )

if __name__ == "__main__":
    app.launch()
