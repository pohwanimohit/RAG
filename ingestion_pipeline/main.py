from pathlib import Path
from pdf_processor import process_pdf
from video_processor import process_video
from document_processor import process_documents
from vector_store import store_chunks

DATA_DIR = Path("data")

files_to_process = [
    DATA_DIR / "SAHKAR-SEVA-Cooperative-Gig-Services-Platform.pdf",
    DATA_DIR / "TIC_pitch.mp4",
]

def main():
    print("=" * 60)
    print("AVIRAL - DOCUMENT PROCESSING PIPELINE")
    print("=" * 60)

    all_documents = []

    #Loading files and routing them to the correct processor.
    for file_path in files_to_process:

        print(f"\n[MAIN] Loading file: {file_path.name}")

        if not file_path.exists():
            print(f"[MAIN] File not found: {file_path}")
            continue

        file_extension = file_path.suffix.lower()

        # Routing PDF files.
        if file_extension == ".pdf":

            print("[MAIN] PDF detected.")

            documents = process_pdf(str(file_path))

        # Routing video files.
        elif file_extension in [
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
        ]:

            print("[MAIN] Video detected.")

            documents = process_video(str(file_path))

        else:

            print(
                f"[MAIN] Unsupported file format: {file_extension}"
            )

            continue

        # Adding processed documents to the shared collection.
        all_documents.extend(documents)

        print(
            f"[MAIN] Documents extracted from this file: "
            f"{len(documents)}"
        )

    # Checking whether any documents were loaded.
    if not all_documents:
        print("\n[MAIN] No documents were loaded. Exiting.")
        return

    print("\n" + "=" * 60)
    print("[MAIN] ALL FILES LOADED SUCCESSFULLY")
    print("=" * 60)

    print(f"Total original documents: {len(all_documents)}")

    # Running the shared chunking and embedding pipeline.
    chunks, vectors, embedding_model = process_documents(
        all_documents
    )
    #Storing chunks and generating their embeddings through Chroma
    vector_store = store_chunks(chunks, embedding_model)


    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)

    print(f"Original documents: {len(all_documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total embeddings: {len(vectors)}")

    if vectors:
        print(f"Embedding dimensions: {len(vectors[0])}")

    # The variables are now available for later use.
    return chunks, vectors, embedding_model

if __name__ == "__main__":
    result = main()