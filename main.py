from embedder import Embedder
from chat import Chat

embedder = Embedder(db_path="chroma_data/", collection_name="textbooks", device="cpu")
embedder.ingest(data_dir="textbooks")


chat = Chat(embedder=embedder)

system_prompt = """
You are an LLM  specializing in CA DFW regulations. 
If a question is relating to DFW regulations, use the RAG tool to retrieve relevant context. 
Cite the source and page number of any context you use, by placing (source, page_number, distance: .xx) after the relevant portion of your response. 
You may incorporate general knowledge, but you must make it clear when using information not explicity from the context.
"""

while True:
    prompt = input("Prompt:\n")
    response = chat.get_model_response(prompt=prompt, system_prompt=system_prompt)
    print(response)