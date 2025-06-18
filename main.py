import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context


# Create an agent
agent = FunctionAgent(
    tools=[],
    llm=Ollama(
        model="llama3.2",
        request_timeout=360.0,
        context_window=8000,
    ),
    system_prompt="You are a helpful assistant",
)

async def main():
    print("Chat with the agent! Type 'quit' or 'exit' to stop.")

    ctx = Context(agent)
    while True:
        
        prompt = input("\nPrompt: ").strip()
        
        
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