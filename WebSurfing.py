"""
WebSurfing.py
=============
User input handler for the MultipleInputs agent.
Allows interactive input for values, name, and web search queries.
"""

from MultipleInputs import app

def get_user_input():
    """Get user input for the agent"""
    print("\n" + "="*50)
    print("LangGraph Web Surfing Agent")
    print("="*50)
    
    # Get name input
    name = input("\nEnter your name: ").strip()
    if not name:
        name = "User"
    
    # Get values input
    values_input = input("Enter numbers to sum (comma-separated, e.g., 1,2,3,4): ").strip()
    try:
        values = [int(x.strip()) for x in values_input.split(",") if x.strip()]
        if not values:
            values = [1, 2, 3]
            print(f"  → Using default values: {values}")
    except ValueError:
        values = [1, 2, 3]
        print(f"  → Invalid input. Using default values: {values}")
    
    # Get search query
    search_query = input("\nEnter a web search query (leave blank to skip): ").strip()
    
    return {
        "name": name,
        "values": values,
        "search_query": search_query
    }

def run_agent(user_data):
    """Run the agent with user input"""
    print("\n" + "-"*50)
    print("Running agent...\n")
    
    state = {
        "values": user_data["values"],
        "name": user_data["name"],
        "result": "",
        "messages": [],
        "search_query": user_data["search_query"]
    }
    
    try:
        result = app.invoke(state)
        
        # Display result
        print(f"\n✓ {result['result']}")
        
        # Display search results if available
        if result.get("messages"):
            for msg in result["messages"]:
                if msg.content and msg.content != "No search query provided":
                    print(f"\n📍 {msg.content}")
        
        print("\n" + "-"*50)
        
    except Exception as e:
        print(f"\n✗ Error running agent: {e}")

def main():
    """Main entry point"""
    while True:
        user_data = get_user_input()
        run_agent(user_data)
        
        # Ask if user wants to run again
        again = input("\nRun again? (yes/no): ").strip().lower()
        if again not in ["yes", "y"]:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
