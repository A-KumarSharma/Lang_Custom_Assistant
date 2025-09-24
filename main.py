from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

@tool

def calculator(a: float , b: float) -> str:
    """useful for performing basic arithmeric calculations with no.""" 
    return f"the sum of {a} and {b} is {a + b}"


def main(): 
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    tools= [calculator]
    
    agent_executor = create_react_agent(model, tools)

    print("How can I assist you today?") 
    print("To exit, please type 'Quit'")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input == "Quit":
            break
        
        try:
            result = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
            print(f"\nAssistant: {result['messages'][-1].content}")
        except Exception as e:
            print(f"\nError: {e}")




if __name__ == "__main__":
    main()
