"""
Example demonstrating MySQL checkpointer configuration.

CRITICAL POINTS FROM THE VIDEO DISCUSSION:
1. The error "LangGraph already has inbuilt Memory saver, it will be ignored"
   occurs when you pass a checkpointer unnecessarily
2. For InMemorySaver: DON'T pass checkpointer (or pass explicitly for shared memory)
3. For MySQL: MUST pass checkpointer to compile()
4. Thread ID must be in config, not in state
"""
import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from agents.business_agent import create_business_agent_graph


def setup_mysql_checkpointer():
    """
    Correct way to set up MySQL checkpointer.
    """
    # Step 1: Create connection string
    # Format: mysql://username:password@host:port/database
    connection_string = "mysql://root:password@localhost:3306/langgraph_db"

    # Step 2: 创建 MySQL checkpointer 对象，返回给变量 checkpointer
    checkpointer = PyMySQLSaver.from_conn_string(connection_string)

    # Step 3: Create workflow
    workflow = create_business_agent_graph()

    # Step 4: MUST compile with checkpointer
    # This is the correct way - no error will occur
    graph = workflow.compile(checkpointer=checkpointer)

    return graph


def run_with_mysql():
    """
    Run agent with MySQL checkpointer.
    """
    print("Setting up MySQL checkpointer...")

    try:
        graph = setup_mysql_checkpointer()
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        print("\nMake sure:")
        print("1. MySQL is running")
        print("2. Database 'langgraph_db' exists")
        print("3. Connection credentials are correct")
        print("\nTo create database:")
        print("  mysql -u root -p")
        print("  CREATE DATABASE langgraph_db;")
        return

    # Create thread ID
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nConversation thread: {thread_id}")

    # First query
    print("\n1. First query...")
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="What is supply chain optimization?")]},
        config=config
    )
    print(f"Response: {result1['messages'][-1].content[:100]}...")

    # Follow-up query
    print("\n2. Follow-up query (should maintain context)...")
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What are the key benefits?")]},
        config=config
    )
    print(f"Response: {result2['messages'][-1].content[:100]}...")

    print("\n✓ Context maintained across queries!")
    print("✓ Conversation persisted to MySQL database")


def common_mistakes():
    """
    Demonstrate common mistakes and how to fix them.
    """
    print("\n" + "="*60)
    print("COMMON MISTAKES AND FIXES")
    print("="*60)

    print("\n❌ MISTAKE 1: Not passing checkpointer for MySQL")
    print("""
    # This won't use MySQL - it will use built-in MemorySaver
    from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
    checkpointer = PyMySQLSaver.from_conn_string(conn_str)
    graph = workflow.compile()  # Missing checkpointer parameter!
    """)

    print("\n✓ FIX 1: Pass checkpointer to compile()")
    print("""
    from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
    checkpointer = PyMySQLSaver.from_conn_string(conn_str)
    graph = workflow.compile(checkpointer=checkpointer)  # Correct!
    """)

    print("\n" + "-"*60)

    print("\n❌ MISTAKE 2: Passing MemorySaver unnecessarily")
    print("""
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    # Warning: "LangGraph already has inbuilt Memory saver, it will be ignored"
    """)

    print("\n✓ FIX 2: Don't pass MemorySaver (or pass for shared memory)")
    print("""
    # Option A: Don't pass anything
    graph = workflow.compile()

    # Option B: Pass for shared memory across instances
    memory = MemorySaver()
    graph1 = workflow.compile(checkpointer=memory)
    graph2 = workflow.compile(checkpointer=memory)  # Shares memory with graph1
    """)

    print("\n" + "-"*60)

    print("\n❌ MISTAKE 3: Putting thread_id in state")
    print("""
    # This won't work - thread_id is not part of state
    result = graph.invoke({
        "messages": [HumanMessage(...)],
        "thread_id": thread_id  # Wrong!
    })
    """)

    print("\n✓ FIX 3: Put thread_id in config")
    print("""
    # Correct way - thread_id goes in config
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(...)]},
        config=config  # Correct!
    )
    """)

    print("\n" + "-"*60)

    print("\n❌ MISTAKE 4: Using wrong state type")
    print("""
    # This won't maintain conversation properly
    class State(TypedDict):
        messages: list[BaseMessage]  # Missing reducer!
    """)

    print("\n✓ FIX 4: Use add_messages reducer")
    print("""
    from typing import Annotated
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]  # Correct!
    """)


if __name__ == "__main__":
    # Show common mistakes
    common_mistakes()

    # Run example (uncomment when MySQL is set up)
    # run_with_mysql()

    print("\n" + "="*60)
    print("To run this example:")
    print("1. Install: pip install langgraph-checkpoint-mysql")
    print("2. Set up MySQL database")
    print("3. Uncomment run_with_mysql() above")
    print("="*60)
