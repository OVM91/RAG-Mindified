import asyncio
import time
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os



Settings.embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-4B")
Settings.llm = Ollama(
        model="llama3.2",
        request_timeout=360.0,
        context_window=8000
    )


# Optimize: Load from storage if exists, otherwise create and persist
if os.path.exists("storage"):
    print("Loading existing index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(
        storage_context,
        embed_model=Settings.embed_model
    )
    print("Index loaded successfully!")
else:
    print("Creating new index from documents...")
    documents = SimpleDirectoryReader("src/data/").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=Settings.embed_model
    )
    index.storage_context.persist("storage")
    print("Index created and saved to storage!")

query_engine = index.as_query_engine(
    llm=Settings.llm,
    similarity_top_k=3,
    streaming=True
    )



async def search_documents(query: str) -> str:
    """Useful for answering questions about the car company Aether Motors."""
    print(f"\n Starting document search for: '{query}'")
    
    # Time the embedding + search process
    start_time = time.time()
    response = await query_engine.aquery(query)
    search_time = time.time() - start_time
    
    print(f"Document search took: {search_time:.2f} seconds")
    return str(response)


# Create an AgentWorkflow (supports streaming!)
agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Ollama(
        model="llama3.2",
        request_timeout=360.0,
        context_window=8000
    ),
    system_prompt="You are a helpful assistant that can search through documents to answer questions about Aether Motors.",
    timeout=120,
    verbose=False
)


async def main():
    print("Chat with the agent! Type 'quit' or 'exit' to stop.")

    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not prompt:
            continue
            
        try:
            print(f"\n Starting agent processing...")
            total_start = time.time()
            
            handler = agent.run(user_msg=prompt)
            
            print("Agent: ", end="", flush=True)
            
            # Time the LLM response generation
            llm_start = time.time()
            
            # Stream the response
            #async for event in handler.stream_events():
                #if hasattr(event, 'delta') and event.delta:
                    #print(event.delta, end="", flush=True)
            
            # Get final result
            result = await handler
            llm_time = time.time() - llm_start
            total_time = time.time() - total_start
            
            print(f"\nLLM response took: {llm_time:.2f} seconds")
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Response: {result.response.content}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    asyncio.run(main())