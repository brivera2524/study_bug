import chromadb
import uuid
import pymupdf4llm
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class Embedder():
    

    def __init__(self, db_path: Path, collection_name: str, data_dir: Path, device: str):
        self.client = chromadb.PersistentClient(path=db_path)
        ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=device
            )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef
            )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64
        )
        self.pdf_dir = data_dir / "pdf"
        self.cache_dir = data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.BATCH_SIZE = 500
        self.manifest_file = Path("manifest.json")
        self.manifest = json.loads(self.manifest_file.read_text(encoding="utf8")) if self.manifest_file.exists() else []

    
    def process_docs_to_chunks(self): 
        cached_stems = {f.stem for f in self.cache_dir.glob("*.json")}

        all_chunks = []

        for pdf_file in self.pdf_dir.glob("*.pdf"):
            cache_file = self.cache_dir / (pdf_file.stem + ".json")

            if pdf_file.stem not in cached_stems:
                chunks = pymupdf4llm.to_markdown(str(pdf_file), page_chunks=True)
                cache_file.write_text(
                    json.dumps(chunks, ensure_ascii=False, indent=2),
                    encoding="utf8"
                )
                print(f"Cached {pdf_file.name}")
            else:
                print(f"{pdf_file.name} Already cached.")

            chunks = json.loads(cache_file.read_text(encoding="utf8"))
            all_chunks.append((pdf_file.stem, chunks))

        return all_chunks



    def embed_all_chunks(self, all_chunks: list):


        for stem, chunk_list in all_chunks:
            if stem in self.manifest:
                print(f"{stem} already embedded, skipping")
                continue

            final_chunks = []
            for chunk in chunk_list:
                page_text = chunk["text"]
                page_number = chunk["metadata"]["page_number"]
                source = chunk["metadata"]["file_path"]

                sub_chunks = self.splitter.split_text(page_text)

                for i, sub_chunk in enumerate(sub_chunks):
                    final_chunks.append({
                        "text": sub_chunk,
                        "page": page_number,
                        "source": source,
                        "chunk_index": i
                    })

            ids = [str(uuid.uuid4()) for _ in final_chunks]
            documents = [chunk["text"] for chunk in final_chunks]
            metadatas = [{
                "chunk_index": chunk["chunk_index"],
                "page": chunk["page"],
                "source": chunk["source"]
            } for chunk in final_chunks]

            for i in range(0, len(final_chunks), self.BATCH_SIZE):
                self.collection.add(
                    ids=ids[i:i + self.BATCH_SIZE],
                    documents=documents[i:i + self.BATCH_SIZE],
                    metadatas=metadatas[i:i + self.BATCH_SIZE]
                )
                print(f"Added chunks {i} to {min(i + self.BATCH_SIZE, len(final_chunks))}")

            self.manifest.append(stem)
            self.manifest_file.write_text(json.dumps(self.manifest, indent=2), encoding="utf8")
            print(f"Marked {stem} as ingested")

        print(f"Total chunks stored: {self.collection.count()}")


if __name__ == "__main__":

    embedder = Embedder(db_path=Path("chroma_data/"), collection_name="textbooks", data_dir=Path("textbooks/"), device="cpu")
    all_chunks = embedder.process_docs_to_chunks()
    embedder.embed_all_chunks(all_chunks)

    query = ["What are the nursing interventions for a patient with hyperkalemia"]

    results = embedder.collection.query(
        query_texts=query,
        n_results=10
    )

    # access the results for
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # print them out together
    for doc, meta, distance in zip(docs, metas, distances):
        print(f"Distance: {distance:.4f}")
        print(f"Page: {meta['page']} | Source: {meta['source']}")
        print(f"Text: {doc[:300]}")
        print("---")