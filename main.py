import asyncio
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os



Settings.embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-4B")
Settings.llm = Ollama(
        model="llama3.2",
        request_timeout=360.0,
        context_window=8000
    )


# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("src/data/").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model
)

query_engine = index.as_query_engine(
    llm=Settings.llm)



async def search_documents(query: str) -> str:
    """Useful for answering questions about the car company Aether Motors."""
    response = await query_engine.aquery(query)
    return str(response)


# Create an agent
agent = FunctionAgent(
    tools=[search_documents],
    llm=Ollama(
        model="llama3.2",
        request_timeout=360.0,
        context_window=8000
    ),
    system_prompt="You are a helpful assistant that can search through documents to answer questions.",
)


async def main():
    print("Chat with the agent! Type 'quit' or 'exit' to stop.")
    ctx = Context(agent)

    while True:
        prompt_embedding = []
        prompt = input("\nPrompt: ").strip()

        prompt_embedding.append(prompt)
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not prompt:
            continue
            
        try:
            # Run the agent with user's prompt
            response = await agent.run(prompt, ctx=ctx)
            print(f"Agent: {str(response)}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    asyncio.run(main())