
from pathlib import Path

from langchain_chroma import Chroma

DB_DIR = Path(__file__).resolve().parent / "chroma_db"

COLLECTION_NAME = "aviral_documents"


def store_chunks(chunks, embedding_model):
  
    if not chunks:
        raise ValueError("No chunks received for database storage.")

    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Creating/connecting to the persistent vector database
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(DB_DIR),
    )

    # Stable IDs for this ordered chunk list
    chunk_ids = [
        f"aviral_chunk_{i:06d}"
        for i in range(len(chunks))
    ]

    # Adding chunks in batches to limit memory usage
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start:start + batch_size]
        batch_ids = chunk_ids[start:start + batch_size]

        vector_store.add_documents(
            documents=batch_chunks,
            ids=batch_ids,
        )

        print(
            f"Stored {min(start + batch_size, len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    print("\nVector database storage completed!")
    print("Database path:", DB_DIR)
    print("Collection:", COLLECTION_NAME)
    print("Chunks submitted:", len(chunks))
    print("Records in collection:", vector_store._collection.count())

    return vector_store
