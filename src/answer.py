import re
import requests
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_K,
    LLAMA_SERVER_URL,
)

load_dotenv(override=True)


class E5Embeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(prefixed_texts)


CHAT_SYSTEM_PROMPT = """
You are a helpful, friendly insurance assistant.

You should always speak as an insurance-focused assistant, even in simple conversation.
Do not behave like a general-purpose assistant.
Do not offer help with unrelated topics.

Behavior rules:
- For greetings or simple chat, respond briefly and naturally as an insurance assistant.
- For simple non-technical questions, keep the response short.
- If the user asks what you can help with, say that you help with insurance-related questions such as coverage, claims, exclusions, policies, and insurance terms.
- Do not invent additional user messages.
- Do not continue the conversation by writing both sides.
"""

RAG_SYSTEM_PROMPT = """
You are a careful and helpful insurance assistant.

Answer the user's insurance question using ONLY the retrieved context.
Do not use outside knowledge.
Do not invent policy facts, exclusions, deadlines, or coverage decisions.

Rules:
- If the context clearly answers the question, answer directly.
- If the context partially answers the question, give the useful part clearly and briefly.
- If the context does not clearly answer the question, say exactly:
  "The answer is not clearly available in the knowledge base."
- Do not copy large sections of the context.
- Do not write step numbers.
- Do not explain your reasoning process.
- Do not write phrases like "the final answer is".
- Do not use LaTeX.
- Do not mention chunk numbers.
- Keep the answer short, direct, and grounded.
- If useful, add one short "Sources:" line with file names only.

Retrieved context:
{context}
"""


def load_vectorstore() -> Chroma:
    embeddings = E5Embeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    return vectordb


def load_retriever():
    vectordb = load_vectorstore()

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": 10,
            "lambda_mult": 0.7,
        },
    )

    return retriever


def format_docs(docs):
    if not docs:
        return "No relevant context found."

    formatted_parts = []

    for doc in docs:
        source = doc.metadata.get("file_name", "unknown")
        doc_type = doc.metadata.get("type", "unknown")

        text = doc.page_content
        if text.startswith("passage: "):
            text = text[len("passage: "):]

        formatted_parts.append(
            f"Source: {source} | Type: {doc_type}\n{text}"
        )

    return "\n\n---\n\n".join(formatted_parts)


def is_small_talk(question: str) -> bool:
    q = question.strip().lower()
    small_talk_inputs = {
        "hi", "hello", "hey", "good morning", "good evening",
        "how are you", "how are you?", "thanks", "thank you",
        "ok", "okay", "bye", "goodbye"
    }
    return q in small_talk_inputs


def is_memory_question(question: str) -> bool:
    q = question.strip().lower()
    patterns = [
        "what did i ask",
        "what was my question",
        "what did i say",
        "what did i just ask",
        "what was the last question",
    ]
    return any(p in q for p in patterns)


def is_insurance_question(question: str) -> bool:
    q = question.strip().lower()

    keywords = [
        "insurance", "car insurance", "auto insurance", "home insurance",
        "homeowners", "health insurance", "claim", "claims", "policy",
        "premium", "deductible", "coverage", "covered", "exclude",
        "exclusion", "liability", "collision", "comprehensive",
        "flood", "damage", "loss", "quote", "premium cost", "insurer"
    ]

    return any(keyword in q for keyword in keywords)


def is_definition_question(question: str) -> bool:
    q = question.strip().lower()
    return q.startswith("what is ") or q.startswith("what does ")


def looks_like_gibberish(text: str) -> bool:
    q = text.strip()
    return len(q) <= 4 and q.isalpha()


def simple_chat_override(question: str):
    q = question.strip().lower()

    overrides = {
        "hi": "Hello — I’m here to help with insurance-related questions.",
        "hello": "Hello — I’m here to help with insurance-related questions.",
        "hey": "Hi — I can help with insurance questions such as coverage, claims, exclusions, and policy terms.",
        "how are you": "I’m doing well — I’m here to help with insurance-related questions.",
        "how are you?": "I’m doing well — I’m here to help with insurance-related questions.",
        "thanks": "You’re welcome.",
        "thank you": "You’re welcome.",
        "ok": "Okay.",
        "okay": "Okay.",
        "no thanks": "Okay — if you have any insurance-related question later, I’ll be here.",
        "bye": "Goodbye — feel free to return if you have another insurance-related question.",
        "goodbye": "Goodbye — feel free to return if you have another insurance-related question.",
        "what can you do": "I can help with insurance-related questions, such as coverage, claims, exclusions, policy terms, and insurance definitions.",
        "what can you do?": "I can help with insurance-related questions, such as coverage, claims, exclusions, policy terms, and insurance definitions.",
        "can you help me": "Yes — I can help with insurance-related questions, such as coverage, claims, exclusions, policy terms, and definitions.",
        "can you help me?": "Yes — I can help with insurance-related questions, such as coverage, claims, exclusions, policy terms, and definitions.",
    }

    return overrides.get(q)


def clean_model_output(text: str) -> str:
    stop_markers = [
        "\nUser:",
        "\nAssistant:",
        "\nUser question:",
        "\nQuestion:",
        "\nAnswer:",
        "User:",
        "User",
        "Assistant:",
        "User question:",
        "Question:",
        "Answer:",
        "\ndef ",
        "def ",
        "\nclass ",
        "class ",
        "\nNote:",
        "Note:",
        "The final answer is:",
        "\nThe final answer is:",
        "End of answer",
        "End of context",
        "End of conversation",
        "End of file",
        "End of session",
        "End of help",
        "End of assistance",
        "End of support",
        "End of service",
        "```",
        "\n```",
        "\nSources:",
        "\nSource:",
        "Sources:",
        "Source:",
    ]

    cleaned = text
    for marker in stop_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]

    fallback = "The answer is not clearly available in the knowledge base."
    if fallback in cleaned and len(cleaned) > len(fallback) + 20:
        cleaned = cleaned.split(fallback)[0].strip()

    return cleaned.strip()


def collapse_repeated_phrases(text: str) -> str:
    text = text.strip()

    parts = [p.strip() for p in text.split("|") if p.strip()]
    if parts and all(p == parts[0] for p in parts):
        return parts[0]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences and all(s == sentences[0] for s in sentences):
        return sentences[0]

    return text


def call_llama_server(prompt: str, n_predict: int = 60, temperature: float = 0.0) -> str:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stop": [
            "\nUser:",
            "\nAssistant:",
            "\nUser question:",
            "\nQuestion:",
            "\nAnswer:",
            "\ndef ",
            "\nclass ",
            "\nSources:",
            "\nSource:",
            "User:",
            "User",
            "Assistant:",
            "User question:",
            "Question:",
            "Answer:",
            "def ",
            "class ",
            "Sources:",
            "Source:",
            "End of answer",
            "End of context",
            "End of conversation",
            "End of file",
            "End of session",
            "End of help",
            "End of assistance",
            "End of support",
            "End of service",
        ]
    }

    response = requests.post(LLAMA_SERVER_URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    cleaned = clean_model_output(data["content"])
    cleaned = collapse_repeated_phrases(cleaned)
    return cleaned


def rewrite_with_history(question: str, history):
    if not history:
        return question

    history_text_parts = []
    for msg in history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_text_parts.append(f"{role.capitalize()}: {content}")

    history_text = "\n".join(history_text_parts)

    rewrite_prompt = f"""
You are helping reformulate a user's follow-up question into a standalone question.

Conversation history:
{history_text}

Current user question:
{question}

Rewrite the current user question so that it is fully understandable on its own.
If the current question is already standalone, return it unchanged.
Only return the rewritten standalone question.
"""

    return call_llama_server(rewrite_prompt, n_predict=80, temperature=0.0)


def run_chat_mode(question: str, history=None) -> str:
    if history is None:
        history = []

    history_text_parts = []
    for msg in history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_text_parts.append(f"{role.capitalize()}: {content}")

    history_text = "\n".join(history_text_parts)

    prompt = f"""{CHAT_SYSTEM_PROMPT}

Conversation history:
{history_text}

User: {question}
Assistant:"""

    return call_llama_server(prompt, n_predict=80, temperature=0.0)


def run_rag_mode(question: str, context: str) -> str:
    prompt = f"""{RAG_SYSTEM_PROMPT.format(context=context)}

Question: {question}

Answer:"""

    return call_llama_server(prompt, n_predict=120, temperature=0.0)


def answer_question(question: str, history=None):
    if history is None:
        history = []

    if is_memory_question(question):
        last_user_questions = [
            msg["content"] for msg in history if msg.get("role") == "user"
        ]
        if last_user_questions:
            return (f"Your previous question was: {last_user_questions[-1]}", [])
        return ("You have not asked a previous question yet.", [])

    override = simple_chat_override(question)
    if override:
        return (override, [])

    if looks_like_gibberish(question):
        return ("I’m not sure what you meant. Try asking a full insurance-related question.", [])

    if not is_insurance_question(question):
        answer = run_chat_mode(question, history)
        return answer, []

    retriever = load_retriever()

    standalone_question = rewrite_with_history(question, history)

    if is_definition_question(question):
        standalone_question = standalone_question + " glossary definition insurance term"

    retrieved_docs = retriever.invoke(standalone_question)
    context = format_docs(retrieved_docs)

    answer = run_rag_mode(standalone_question, context)

    return answer, retrieved_docs


