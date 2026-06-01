import gradio as gr
from dotenv import load_dotenv

from answer import answer_question
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn

load_dotenv(override=True)


def clean_chunk_text(text: str) -> str:
    if text.startswith("passage: "):
        return text[len("passage: "):]
    return text


def format_context(context_docs) -> str:
    if not context_docs:
        return "No retrieved context."

    sections = ["## Retrieved Context\n"]

    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("file_name", "unknown")
        doc_type = doc.metadata.get("type", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")

        text = clean_chunk_text(doc.page_content)

        sections.append(
            f"### Source {i}\n"
            f"**File:** {source}  \n"
            f"**Type:** {doc_type}  \n"
            f"**Chunk ID:** {chunk_id}  \n\n"
            f"{text}\n"
        )

    return "\n\n---\n\n".join(sections)


def add_user_message(message, history):
    if history is None:
        history = []

    history = history + [{"role": "user", "content": message}]
    return "", history


def respond(history):
    if not history:
        return history, "No retrieved context."

    last_user_message = history[-1]["content"]
    prior_history = history[:-1]

    answer, context_docs = answer_question(last_user_message, prior_history)

    history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return history, format_context(context_docs)


custom_css = """
.gradio-container {
    max-width: 1450px !important;
}

.main-title {
    text-align: center;
    margin-bottom: 0.2rem;
}

.sub-title {
    text-align: center;
    opacity: 0.8;
    margin-bottom: 1.2rem;
}

.example-row button {
    min-height: 42px !important;
    border-radius: 12px !important;
}

button {
    border-radius: 12px !important;
}

.context-box {
    max-height: 620px;
    overflow-y: auto;
    padding-right: 6px;
}

footer {
    visibility: hidden;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
)


with gr.Blocks(
    title="Insurance RAG Assistant",
    theme=theme,
    css=custom_css,
    fill_height=True
) as demo:
    gr.Markdown(
        """
        <div class="main-title">
            <h1>Insurance RAG Assistant</h1>
        </div>
        <div class="sub-title">
            Chat naturally or ask insurance questions about coverage, claims, exclusions, and policy terms.
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=7):
            with gr.Group():
                chatbot = gr.Chatbot(
                    label="Conversation",
                    type="messages",
                    height=620,
                    show_copy_button=True,
                    avatar_images=(None, None),
                )

                user_input = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    lines=1,
                    container=False,
                )

                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat", variant="secondary")

            gr.Markdown("### Quick Examples")

            with gr.Row(elem_classes=["example-row"]):
                ex1 = gr.Button("What is a premium?")
                ex2 = gr.Button("How do I file a claim?")
                ex3 = gr.Button("Does homeowners insurance cover flood damage?")

            with gr.Row(elem_classes=["example-row"]):
                ex4 = gr.Button("What is a deductible?")
                ex5 = gr.Button("Does auto insurance cover damage to my own car?")
                ex6 = gr.Button("What can you do?")
                ex7 = gr.Button("What does the health insurance policy exclude?")
                ex8 = gr.Button("What is liability coverage?")
                ex9 = gr.Button("What is collision coverage?")
                ex10 = gr.Button("What does the home insurance policy cover?")
                ex11 = gr.Button("What information do I need to provide when filing a claim?")
                ex12 = gr.Button("Does health insurance cover my hospital expenses?")

        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("### Retrieved Context")
                context_panel = gr.Markdown(
                    "Retrieved context will appear here for insurance questions.",
                    elem_classes=["context-box"]
                )

    user_input.submit(
        fn=add_user_message,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
    ).then(
        fn=respond,
        inputs=[chatbot],
        outputs=[chatbot, context_panel],
    )

    send_btn.click(
        fn=add_user_message,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
    ).then(
        fn=respond,
        inputs=[chatbot],
        outputs=[chatbot, context_panel],
    )

    clear_btn.click(
        fn=lambda: ([], "Retrieved context will appear here for insurance questions."),
        inputs=[],
        outputs=[chatbot, context_panel],
    )

    for btn, question in [
        (ex1, "What is a premium?"),
        (ex2, "How do I file a claim?"),
        (ex3, "Does homeowners insurance cover flood damage?"),
        (ex4, "What is a deductible?"),
        (ex5, "Does auto insurance cover damage to my own car?"),
        (ex6, "What can you do?"),
        (ex7, "What does the health insurance policy exclude?"),
        (ex8, "What is liability coverage?"),
        (ex9, "What is collision coverage?"),
        (ex10, "What does the home insurance policy cover?"),
        (ex11, "What information do I need to provide when filing a claim?"),
        (ex12, "Does health insurance cover my hospital expenses?"),
    ]:
        btn.click(
            fn=lambda history, q=question: ("", (history or []) + [{"role": "user", "content": q}]),
            inputs=[chatbot],
            outputs=[user_input, chatbot],
        ).then(
            fn=respond,
            inputs=[chatbot],
            outputs=[chatbot, context_panel],
        )


api = FastAPI()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

@api.get("/")
def root():
    return RedirectResponse(url="/ui")

@api.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer, context_docs = answer_question(req.question, [])
    sources = []
    for doc in context_docs:
        name = doc.metadata.get("file_name", "unknown")
        if name not in sources:
            sources.append(name)
    return AskResponse(answer=answer, sources=sources)

app = gr.mount_gradio_app(api, demo, path="/ui")


uvicorn.run(app, host="0.0.0.0", port=7860)