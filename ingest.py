import chromadb
import uuid
import pymupdf4llm
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction



# create the embedding function with GPU for ingest
ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cuda"
)

# pass it to the collection
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="textbook",
    embedding_function=ef
)


textbook_names = ["textbook_0.pdf"]
pdf_dir = Path("textbooks/pdf")
cache_dir = Path("textbooks/cache")

cache_dir.mkdir(parents=True, exist_ok=True)


# Check first for cached version, load and cache if doesn't exist
for textbook_name in textbook_names:
    pdf_file = pdf_dir / textbook_name
    cache_file = cache_dir / (Path(textbook_name).stem + ".json")

    if not cache_file.exists():
        chunks = pymupdf4llm.to_markdown(str(pdf_file), page_chunks=True)
        cache_file.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf8"
        )

    chunks = json.loads(cache_file.read_text(encoding="utf8"))


# Currently one chunk per page, reduce further for more specific retrieval
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64
)

final_chunks = []
for chunk in chunks:
    page_text = chunk["text"]
    page_number = chunk["metadata"]["page_number"]
    source = chunk["metadata"]["file_path"]
    
    sub_chunks = splitter.split_text(page_text)
    
    for i, sub_chunk in enumerate(sub_chunks):
        final_chunks.append({
            "text": sub_chunk,
            "page": page_number,
            "source": source,
            "chunk_index": i
        })


# add in batches to respect ChromaDB's limit
BATCH_SIZE = 500

ids = [str(uuid.uuid4()) for _ in final_chunks]
documents = [chunk["text"] for chunk in final_chunks]
metadatas = [{
    "chunk_index": chunk["chunk_index"],
    "page": chunk["page"],
    "source": chunk["source"]
} for chunk in final_chunks]

for i in range(0, len(final_chunks), BATCH_SIZE):
    collection.add(
        ids=ids[i:i + BATCH_SIZE],
        documents=documents[i:i + BATCH_SIZE],
        metadatas=metadatas[i:i + BATCH_SIZE]
    )
    print(f"Added chunks {i} to {min(i + BATCH_SIZE, len(final_chunks))}")

print(f"Total chunks stored: {collection.count()}")

query = "What is the reversal agent for opiods?"

results = collection.query(
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