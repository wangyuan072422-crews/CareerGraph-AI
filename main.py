"""
Main entry point demonstrating correct checkpointer usage for DIRECT PYTHON EXECUTION.

⚠️ IMPORTANT - DEPLOYMENT CONTEXT:
This file is for running agents directly with Python (not langgraph dev/Studio).
When running directly, you MUST pass a checkpointer to compile().

✅ This file: Direct Python execution → USE checkpointer
❌ graphs/*.py: LangGraph Studio → NO checkpointer
  
KEY POINTS:
1. Thread ID is passed in config during invocation, NOT in state
2. For direct Python: You MUST pass checkpointer (MemorySaver or MySQL)
3. For LangGraph Studio: See graphs/multi_agent_system.py (no checkpointer)
"""
import uuid
from langchain_core.messages import HumanMessage
from agents.business_agent import create_business_agent_graph
from agents.database_agent import create_database_agent_graph
from utils.checkpointer import get_memory_saver, get_mysql_saver

# 建一张业务图 → 给它配一个 checkpointer → 用同一个 thread_id 连问三句 → 打印回答，顺便演示"换个 thread_id 就失忆"。

def run_agent_with_memory(use_mysql: bool = False):
    """
    Demonstrate correct usage of checkpointer with conversational memory.

    Args:
        use_mysql: If True, use MySQL checkpointer. If False, use InMemorySaver.
    """
    # Create the workflow
    workflow = create_business_agent_graph()

    # ✅ CORRECT FOR DIRECT PYTHON EXECUTION - PASS CHECKPOINTER
    # This is different from graphs/multi_agent_system.py which is for Studio
    if use_mysql:
        # For MySQL: MUST pass checkpointer
        print("Using MySQL Checkpointer")
        checkpointer = get_mysql_saver("mysql://user:pass@localhost:3306/langgraph")
        graph = workflow.compile(checkpointer=checkpointer)
    else:
        # For InMemory: Pass checkpointer for direct Python execution
        print("Using InMemory Checkpointer")
        checkpointer = get_memory_saver()
        graph = workflow.compile(checkpointer=checkpointer)

        # Note: For LangGraph Studio, you would NOT pass checkpointer
        # See graphs/multi_agent_system.py for Studio usage

    # Create a thread ID for this conversation
    # IMPORTANT: Thread ID goes in config, NOT in state
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n=== Starting conversation with thread_id: {thread_id} ===\n")

    # First query
    print("Query 1: What are the key supply chain metrics?")
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="What are the key supply chain metrics?")]},
        config=config  # Thread ID is here!
    )
    print(f"Response: {result1['messages'][-1].content}\n")

    # Follow-up query - this should maintain context
    print("Query 2: Can you explain the first one in more detail?")
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Can you explain the first one in more detail?")]},
        config=config  # Same thread ID maintains context
    )
    print(f"Response: {result2['messages'][-1].content}\n")

    # Another follow-up to verify context is maintained
    print("Query 3: What about the others?")
    result3 = graph.invoke(
        {"messages": [HumanMessage(content="What about the others?")]},
        config=config  # Same thread ID
    )
    print(f"Response: {result3['messages'][-1].content}\n")

    print("=== Conversation complete ===")
    print(f"All messages maintained context using thread_id: {thread_id}")
    # 返回带记忆的图对象 和 刚才用的id
    return graph, thread_id

# 换一个新thread_id 问一个和第三次一模一样的问题
def demonstrate_new_conversation(graph, old_thread_id):
    """
    Demonstrate that a new thread ID starts a fresh conversation.
    """
    new_thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": new_thread_id}}

    print(f"\n=== New conversation with thread_id: {new_thread_id} ===\n")
    print("Query: What about the others?")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What about the others?")]},
        config=config
    )
    print(f"Response: {result['messages'][-1].content}\n")
    print("Notice: The agent doesn't have context because it's a new thread_id")


if __name__ == "__main__":
    # Run with InMemory checkpointer
    graph, thread_id = run_agent_with_memory(use_mysql=False)

    # Demonstrate new conversation
    demonstrate_new_conversation(graph, thread_id)

    print("\n" + "="*60)
    print("To use MySQL checkpointer:")
    print("1. Set up MySQL database")
    print("2. Run: python main.py --mysql")
    print("3. Update connection string in utils/checkpointer.py")
    print("="*60)
