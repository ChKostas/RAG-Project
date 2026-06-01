from dotenv import load_dotenv
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    KNOWLEDGE_BASE_PATH,
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

load_dotenv(override=True)


def load_markdown_documents() -> list[Document]:
    documents = []

    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(f"Knowledge base path does not exist: {KNOWLEDGE_BASE_PATH}")

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue

        doc_type = folder.name

        for file_path in folder.glob("*.md"):
            try:
                text = file_path.read_text(encoding="utf-8").strip()

                if not text:
                    print(f"Skipping empty file: {file_path}")
                    continue

                metadata = {
                    "source": file_path.as_posix(),
                    "type": doc_type,
                    "file_name": file_path.name,
                }

                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    split_docs = []

    for doc in tqdm(documents, desc="Chunking documents"):
        chunks = splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content.strip()

            if not chunk_text:
                continue

            chunk_metadata = dict(chunk.metadata)
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_id"] = f"{chunk_metadata['file_name']}_{i}"

            split_docs.append(
                Document(
                    page_content=f"passage: {chunk_text}",
                    metadata=chunk_metadata,
                )
            )

    print(f"Created {len(split_docs)} chunks")
    return split_docs


def build_vectorstore(chunks: list[Document]) -> None:
    if not chunks:
        raise ValueError("No chunks were created. Cannot build vectorstore.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    try:
        vectordb.delete_collection()
        print(f"Deleted previous collection '{COLLECTION_NAME}'")
    except Exception:
        print("No previous collection found or delete was skipped.")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
    )

    try:
        vectordb.persist()
    except Exception:
        pass

    print(f"Vector DB created at: {DB_PATH}")
    print(f"Stored {len(chunks)} chunks in collection '{COLLECTION_NAME}'")


def ingest_pipeline():
    documents = load_markdown_documents()
    if not documents:
        raise ValueError("No markdown documents found in knowledge-base.")

    chunks = split_documents(documents)
    build_vectorstore(chunks)

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_pipeline()