from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def process_documents(documents):
   
    if not documents:
        raise ValueError("No documents were provided for processing.")

    #Combining documents.
    print("\n[DOCUMENT] Combining documents...")
    print(f"[DOCUMENT] Total original documents: {len(documents)}")

    #Splitting documents into chunks.
    print("\n[CHUNKING] Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"[CHUNKING] Total chunks created: {len(chunks)}")

    # Displaying the first five chunks.
    for i, chunk in enumerate(chunks[:5], start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.page_content)
        print("Metadata:", chunk.metadata)
        print("Characters:", len(chunk.page_content))

    # Loading the embedding model.
    print("\n[EMBEDDING] Loading HuggingFace embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    print("[EMBEDDING] Model loaded successfully!")

    # Extracting text from the chunks.
    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # Generating vector embeddings.
    print("\n[EMBEDDING] Generating vector embeddings...")

    vectors = embedding_model.embed_documents(chunk_texts)

    print(f"[EMBEDDING] Total chunks embedded: {len(vectors)}")

    # Displaying details of the first embedding.
    if vectors:
        print(
            f"[EMBEDDING] Embedding dimensions: {len(vectors[0])}"
        )

        print(
            "[EMBEDDING] First 10 vector values:",
            vectors[0][:10],
        )

        print("\nFirst chunk:")
        print(chunks[0].page_content[:50])

        print("\nFirst chunk vector:")
        print(vectors[0][:10])

    return chunks, vectors, embedding_model