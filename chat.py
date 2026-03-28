from embedder import Embedder
import os
from dotenv import load_dotenv
from anthropic import Anthropic, beta_tool


class Chat():
    def __init__(self, embedder: Embedder):
        load_dotenv()

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable must be defined.")

        self.embedder = embedder
        self.client = Anthropic()
        self.rag_tool = self._make_rag_tool()
        self.messages = []
        


    def _make_rag_tool(self):
        embedder = self.embedder
    
        @beta_tool
        def RAG_query(query_text: str, n_results: int) -> list[dict]:
            """Query the fishing regulations for relevant content.
            
            Use this when the user asks a question that requires information
            from the regulations documents. Do not use for conversational follow-ups like
            rephrasing, shortening, or formatting previous responses.

            The context sources are written like a formal government regulations booklet.
            The language is dense, rule-based, and highly structured. It uses legal
            and administrative wording rather than conversational explanation. The document 
            repeatedly uses exact regulatory terms, section references, and formulaic phrasing, 
            so retrieval should prioritize literal terminology over loose paraphrase.

            To create the best query_text input, generate a hypothetical answer to the following
            question as if you were an expert. Be specific and use domain-appropriate terminology.
            Do not hedge or say you don't know. Commit to a plausible answer. Use this answer as query_text.

            Additionally, optimize n_results based on the query.
            
            Args:
                query_text: The search query to run against the textbook collection.
                n_reults: The number of unique pieces of context to return.
            
            Returns:
                A list of relevant text chunks with source and page metadata.
            """
            print(f"Model Calling RAG_query. Query: {query_text}, n_results:{n_results}")
            return str(embedder.query(query_text, n_results=n_results))
            
        
        return RAG_query

    
    def get_model_response(self, prompt: str, system_prompt:str) -> str:

        new_user_message = {"role": "user", "content": f"{prompt}"}
        self.messages.append(new_user_message)

        runner = self.client.beta.messages.tool_runner(
            max_tokens=1024,
            model="claude-sonnet-4-6",
            # placeholder system prompt, needs refinement
            system=system_prompt,
            tools=[self.rag_tool],
            messages=self.messages
        )

        assistant_text_response = (list(runner)[-1]).content[0].text
        
        new_assistant_message = {"role": "assistant", "content": f"{assistant_text_response}"}
        
        self.messages.append(new_assistant_message)

        return assistant_text_response


    
