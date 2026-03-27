from embedder import Embedder
from chat import Chat

embedder = Embedder(db_path="chroma_data/", collection_name="textbooks", device="cpu")

chat = Chat(embedder=embedder)

while True:
    prompt = input("Prompt:\n")
    response = chat.get_model_response(prompt)
    print(response)