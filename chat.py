from embedder import Embedder
import os
from dotenv import load_dotenv
from anthropic import Anthropic, beta_tool


class Chat():
    def __init__(self, embedder: Embedder):
        load_dotenv()
        self.embedder = embedder
        self.client = Anthropic()
        self.rag_tool = self._make_rag_tool()
        self.messages = []
        


    def _make_rag_tool(self):
        embedder = self.embedder
    
        @beta_tool
        def RAG_query(query_text: str) -> list[dict]:
            """Query the nursing textbook collection for relevant content.
            
            Use this when the user asks a question that requires information
            from the textbooks. Do not use for conversational follow-ups like
            rephrasing, shortening, or formatting previous responses.
            
            Args:
                query_text: The search query to run against the textbook collection.
            
            Returns:
                A list of relevant text chunks with source and page metadata.
            """

            return str(embedder.query(query_text, n_results=5))
            
        
        return RAG_query

    
    def get_model_response(self, prompt: str) -> str:

        new_user_message = {"role": "user", "content": f"{prompt}"}
        self.messages.append(new_user_message)

        runner = self.client.beta.messages.tool_runner(
            max_tokens=1024,
            model="claude-haiku-4-5",
            # placeholder system prompt, needs refinement
            system="You are an LLM with access to an RAG tool to gain context for your answer. Always use this tool, and cite the source and page number if you reference any context.",
            tools=[self.rag_tool],
            messages=self.messages
        )

        assistant_text_response = (list(runner)[-1]).content[0].text
        
        new_assistant_message = {"role": "assistant", "content": f"{assistant_text_response}"}
        
        self.messages.append(new_assistant_message)

        return assistant_text_response

    
