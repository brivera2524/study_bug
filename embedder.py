import chromadb
import uuid
import pymupdf4llm
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import Any


class Embedder():
    """Ingests PDFs into ChromaDB and handles similarity search."""

    def __init__(self, db_path: str | Path, collection_name: str, device: str):
        """Set up ChromaDB client, embedding model, and manifest.
        
        Args:
            db_path: Path to persistent ChromaDB storage directory.
            collection_name: Name of the collection to use or create.
            device: 'cuda' or 'cpu' for sentence-transformers inference.
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

        self.manifest_file = Path(db_path) / "manifest.json"
        try:
            self.manifest = json.loads(
                self.manifest_file.read_text(encoding="utf8")
            ) if self.manifest_file.exists() else []
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Manifest file is corrupted at {self.manifest_file}: {e}") from e


    def _split_chunks(self, chunks: list[dict[str, Any]]) -> list[dict]:
        """Splits page-level chunks into smaller sub-chunks for embedding.
        
        Args:
            chunks: List of page dicts from pymupdf4llm, each containing
                    'text' and 'metadata' keys.
        
        Returns:
            List of sub-chunk dicts with 'text', 'page', 'source', and 'chunk_index'.
        """
        
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
    


    def _load_chunks(self, data_dir: Path) -> list[tuple[str, list[dict]]]:
        """Loads and caches chunks from all PDFs in data_dir/pdf/.

        Extracts and splits new PDFs, writing results to data_dir/cache/.
        Already-cached PDFs are loaded directly from disk.

        Args:
            data_dir: Root directory containing pdf/ and cache/ subdirectories.

        Returns:
            List of (stem, chunks) tuples where stem is the PDF filename
            without extension and chunks is the list of sub-chunk dicts.
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        

        pdf_dir = data_dir / "pdf"
        cache_dir = data_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)


        cached_stems = {f.stem for f in cache_dir.glob("*.json")}

        final_chunks = []

        for pdf_file in pdf_dir.glob("*.pdf"):
            cache_file = cache_dir / (pdf_file.stem + ".json")

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



    def _upsert_chunks(self, all_chunks: list[tuple[str, list[dict]]]):
        """Embeds and upserts chunks into ChromaDB, skipping already-ingested documents.

        Args:
            all_chunks: List of (stem, chunks) tuples as returned by _load_chunks.
        """
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



    def ingest(self, data_dir: Path | str):
        """Ingests all PDFs from data_dir into ChromaDB.

        Args:
            data_dir: Root directory containing pdf/ and cache/ subdirectories.
        """
        data_dir = Path(data_dir)
        all_chunks = self._load_chunks(data_dir)
        self._upsert_chunks(all_chunks)


    def query(self, query: list[str], n_results: int) -> chromadb.QueryResult:
        """Searches the collection for chunks similar to the query.

        Args:
            query: List of query strings to search for.
            n_results: Number of results to return per query.
        """
        return self.collection.query(query_texts=query,n_results=n_results)


