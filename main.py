from embedder import Embedder
embedder = Embedder(db_path="chroma_data/", collection_name="textbooks", device="cpu")
embedder.ingest(data_dir="textbooks/")


results = embedder.query(query=["How much morphine will cause an overdose?"],n_results=5)

# access the results
docs = results["documents"][0]
metas = results["metadatas"][0]
distances = results["distances"][0]

# print them out together
for doc, meta, distance in zip(docs, metas, distances):
    print(f"Distance: {distance:.4f}")
    print(f"Page: {meta['page']} | Source: {meta['source']}")
    print(f"Text: {doc}")
    print("---")