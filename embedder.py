import chromadb
import uuid
import pymupdf4llm
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import Any


class Embedder():
    """Handles PDF ingestion, chunking, and embedding into ChromaDB.

    Extracts text from PDFs using pymupdf4llm, splits into chunks using
    LangChain's RecursiveCharacterTextSplitter, and stores embeddings
    in a persistent ChromaDB collection using sentence-transformers.

    Attributes:
        collection: ChromaDB collection for storing and querying embeddings.
        splitter: LangChain text splitter for chunking page text.
        pdf_dir: Path to directory containing source PDF files.
        cache_dir: Path to directory for caching extracted JSON files.
        manifest: List of already-ingested PDF stems.
    """

    def __init__(self, db_path: Path, collection_name: str, device: str):
        """Initialize the Embedder with ChromaDB client and embedding function.

        Sets up a persistent ChromaDB client, creates or connects to an existing
        collection with a sentence-transformers embedding function, and loads
        the ingestion manifest to track previously embedded documents.

        Args:
            db_path: Path to the persistent ChromaDB storage directory.
                Will be created if it does not exist.
            collection_name: Name of the ChromaDB collection to use or create.
            data_dir: Root directory containing pdf/ and cache/ subdirectories.
            device: Device for sentence-transformers inference.
                Use 'cuda' for GPU-accelerated ingestion, 'cpu' for query time.

        Raises:
            FileNotFoundError: If data_dir does not exist.
            ValueError: If device is not 'cuda' or 'cpu'.
            RuntimeError: If ChromaDB client fails to initialize.
        """
        

        if device not in ("cuda", "cpu"):
            raise ValueError(f"Invalid device '{device}'. Must be 'cuda' or 'cpu'.")

        try:
            self.client = chromadb.PersistentClient(path=str(db_path))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB at {db_path}: {e}") from e


        try:
            ef = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2",
                device=device
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model or create collection: {e}") from e
        


        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64
        )
        
        self.BATCH_SIZE  = 500
        
        self.manifest_file = Path("manifest.json")
        try:
            self.manifest = json.loads(
                self.manifest_file.read_text(encoding="utf8")
            ) if self.manifest_file.exists() else []
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Manifest file is corrupted at {self.manifest_file}: {e}") from e


    def _split_chunks(self, chunks: dict[str, Any]) -> list[dict]:
            split_chunks = []
            for chunk in chunks:
                page_text = chunk["text"]
                page_number = chunk["metadata"]["page_number"]
                source = chunk["metadata"]["file_path"]

                sub_chunks = self.splitter.split_text(page_text)

                for i, sub_chunk in enumerate(sub_chunks):
                    split_chunks.append({
                        "text": sub_chunk,
                        "page": page_number,
                        "source": source,
                        "chunk_index": i
                    })
            return split_chunks
    

    # Iterate through every file in pdf/ if it has existing cache, good. if not, chunk it
    def _process_docs_to_chunks(self, data_dir: Path) -> list[tuple]:

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        

        self.pdf_dir = data_dir / "pdf"
        self.cache_dir = data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)


        cached_stems = {f.stem for f in self.cache_dir.glob("*.json")}

        final_chunks = []

        for pdf_file in self.pdf_dir.glob("*.pdf"):
            cache_file = self.cache_dir / (pdf_file.stem + ".json")

            if pdf_file.stem not in cached_stems:
                chunks = pymupdf4llm.to_markdown(str(pdf_file), page_chunks=True)
                split_chunks = self._split_chunks(chunks)
                cache_file.write_text(
                    json.dumps(split_chunks, ensure_ascii=False, indent=2),
                    encoding="utf8"
                )
                print(f"Cached {pdf_file.name}")
            else:
                print(f"{pdf_file.name} Already cached.")

            chunks = json.loads(cache_file.read_text(encoding="utf8"))
            final_chunks.append((pdf_file.stem, chunks))

        return final_chunks



    def _embed_all_chunks(self, all_chunks: list):
        for stem, chunk_list in all_chunks:
            if stem in self.manifest:
                print(f"{stem} already embedded, skipping")
                continue

            ids = [str(uuid.uuid4()) for _ in chunk_list]
            documents = [chunk["text"] for chunk in chunk_list]
            metadatas = [{
                "chunk_index": chunk["chunk_index"],
                "page": chunk["page"],
                "source": chunk["source"]
            } for chunk in chunk_list]

            for i in range(0, len(chunk_list), self.BATCH_SIZE):
                self.collection.add(
                    ids=ids[i:i + self.BATCH_SIZE],
                    documents=documents[i:i + self.BATCH_SIZE],
                    metadatas=metadatas[i:i + self.BATCH_SIZE]
                )
                print(f"Added chunks {i} to {min(i + self.BATCH_SIZE, len(chunk_list))}")

            self.manifest.append(stem)
            self.manifest_file.write_text(json.dumps(self.manifest, indent=2), encoding="utf8")
            print(f"Marked {stem} as ingested")

        print(f"Total chunks stored: {self.collection.count()}")



    def ingest(self, data_dir: str):
        data_dir = Path(data_dir)
        all_chunks = self._process_docs_to_chunks(data_dir)
        self._embed_all_chunks(all_chunks)


    def query(self, query: list[str], n_results: int) -> chromadb.QueryResult:
        
        return self.collection.query(query_texts=query,n_results=n_results)




if __name__ == "__main__":


    embedder = Embedder(db_path=Path("chroma_data/"), collection_name="textbooks", device="cpu")
    embedder.ingest(data_dir="textbooks/")


    results = embedder.query(query=["How much morphine will cause an overdose?"],n_results=5)

    # access the results for
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # print them out together
    for doc, meta, distance in zip(docs, metas, distances):
        print(f"Distance: {distance:.4f}")
        print(f"Page: {meta['page']} | Source: {meta['source']}")
        print(f"Text: {doc}")
        print("---")