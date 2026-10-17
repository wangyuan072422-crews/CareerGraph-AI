"""
Example client for interacting with the FastAPI server.
Demonstrates remote graph invocation with conversational memory.
"""
import requests
import json

# 通过 HTTP 去调 api/server.py 起的那台 FastAPI 服务，
# 用三个 demo 演示"记忆在 HTTP 层怎么工作"。

# API base URL 请求的是 POST /query
BASE_URL = "http://localhost:8000"


def query_agent(message: str, thread_id: str = None, agent_type: str = "business"):
    """
    Send a query to the agent API.

    Args:
        message: The user's query
        thread_id: Thread ID for conversation continuity (optional)
        agent_type: "business" or "database"

    Returns:
        Response dict with response, thread_id, and agent_type
    """
    url = f"{BASE_URL}/query"
    # 先放入 message 和 agent_type
    payload = {
        "message": message,
        "agent_type": agent_type
    }
    # 有thread_id再放入thread_id
    if thread_id:
        payload["thread_id"] = thread_id
    # 把 Python 字典序列化成 HTTP JSON
    response = requests.post(url, json=payload)
    # 用来把 HTTP 4xx 或 5xx 转换成异常。
    # 否则即使服务端返回错误，客户端也可能继续把错误响应当成正常结果处理。
    response.raise_for_status()
    # 把服务端 JSON 响应转回 Python 字典
    return response.json()


def get_conversation_history(thread_id: str, agent_type: str = "business"):
    """
    Retrieve conversation history for a thread.
    """
    url = f"{BASE_URL}/conversation/{thread_id}"
    response = requests.get(url, params={"agent_type": agent_type})
    response.raise_for_status()
    return response.json()


def demo_conversation():
    """
    Demonstrate a full conversation with context preservation.
    """
    print("="*60)
    print("DEMO: Multi-turn Conversation with Context")
    print("="*60)

    # First query - no thread_id, will create new conversation
    print("\n1. Starting new conversation...")
    result1 = query_agent("What are the main supply chain challenges?")

    print(f"   Response: {result1['response'][:100]}...")
    print(f"   Thread ID: {result1['thread_id']}")

    # Save thread_id for follow-up queries
    thread_id = result1['thread_id']

    # Follow-up query - use same thread_id
    print("\n2. Follow-up question (with context)...")
    result2 = query_agent(
        "Can you explain the first one in detail?",
        thread_id=thread_id
    )

    print(f"   Response: {result2['response'][:100]}...")
    print(f"   Thread ID: {result2['thread_id']}")

    # Another follow-up
    print("\n3. Another follow-up (with context)...")
    result3 = query_agent(
        "What solutions would you recommend?",
        thread_id=thread_id
    )

    print(f"   Response: {result3['response'][:100]}...")

    # Get full conversation history
    print("\n4. Retrieving conversation history...")
    history = get_conversation_history(thread_id)

    print(f"   Total messages: {len(history['messages'])}")
    print("\n   Full conversation:")
    for i, msg in enumerate(history['messages'], 1):
        role = msg['role'].upper()
        content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        print(f"   {i}. [{role}] {content}")

    print("\n✓ Context maintained across all queries!")

    return thread_id


def demo_multi_agent():
    """
    Demonstrate switching between different agents.
    """
    print("\n" + "="*60)
    print("DEMO: Multi-Agent System")
    print("="*60)

    # Query business agent
    print("\n1. Querying Business Agent...")
    result1 = query_agent(
        "What are the company policies for returns?",
        agent_type="business"
    )
    print(f"   Response: {result1['response'][:100]}...")
    business_thread = result1['thread_id']

    # Query database agent (different thread)
    print("\n2. Querying Database Agent...")
    result2 = query_agent(
        "Show me the total orders from last month",
        agent_type="database"
    )
    print(f"   Response: {result2['response'][:100]}...")
    database_thread = result2['thread_id']

    # Continue business conversation
    print("\n3. Continuing Business Agent conversation...")
    result3 = query_agent(
        "What about refunds?",
        thread_id=business_thread,
        agent_type="business"
    )
    print(f"   Response: {result3['response'][:100]}...")

    print("\n✓ Each agent maintains separate conversation context!")


def demo_new_vs_existing_thread():
    """
    Demonstrate difference between new and existing conversations.
    """
    print("\n" + "="*60)
    print("DEMO: New vs Existing Conversation")
    print("="*60)

    # Start conversation
    print("\n1. Starting conversation about inventory...")
    result1 = query_agent("What is inventory management?")
    thread_id = result1['thread_id']
    print(f"   Response: {result1['response'][:100]}...")

    # Continue with SAME thread_id
    print("\n2. Follow-up with SAME thread_id...")
    result2 = query_agent("What are the best practices?", thread_id=thread_id)
    print(f"   Response: {result2['response'][:100]}...")
    print("   ✓ Agent has context from previous message")

    # New query with DIFFERENT thread_id (or None)
    print("\n3. Same question but with NEW thread_id...")
    result3 = query_agent("What are the best practices?")  # No thread_id = new conversation
    print(f"   Response: {result3['response'][:100]}...")
    print("   ✗ Agent doesn't have context (new conversation)")


if __name__ == "__main__":
    print("\nStarting API Client Examples...")
    print("Make sure the API server is running: python -m api.server\n")

    try:
        # Check if server is running
        response = requests.get(BASE_URL)
        print(f"✓ Server is running: {response.json()}\n")

        # Run demos
        demo_conversation()
        demo_multi_agent()
        demo_new_vs_existing_thread()

        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API server")
        print("\nPlease start the server first:")
        print("  cd langgraph_multi_agent")
        print("  python -m api.server")
    except Exception as e:
        print(f"❌ Error: {e}")
