from embedder import Embedder
from unittest.mock import MagicMock, patch



def test_query():
    with patch("embedder.chromadb.PersistentClient") as mock_client, patch("embedder.SentenceTransformerEmbeddingFunction") as mock_ef:
        
        embedder = Embedder(db_path="fake/path", collection_name="fake_collection", device="cpu")

        embedder.collection.query.return_value = {
            "documents": [["some text"]],
            "ids": [["abc123"]],
            "distances": [[0.25]],
            "metadatas": [[{"source": "textbook.pdf", "page": 1, "chunk_index": 0}]]
        }

        result = embedder.query("test query", n_results=1)

        assert result[0]["document"] == "some text"
        assert result[0]["id"] == "abc123"
        assert result[0]["distance"] == 0.25
        assert result[0]["source"] == "textbook.pdf"
        assert result[0]["page"] == 1
        assert len(result) == 1
