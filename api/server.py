"""
FastAPI server for remote graph invocation.

This demonstrates how to:
1. Deploy LangGraph agents as API endpoints
2. Maintain conversational memory across HTTP requests
3. Handle thread_id for conversation persistence
"""
import uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agents.business_agent import create_business_agent_graph
from agents.database_agent import create_database_agent_graph
from agents.supervisor import create_supervisor_graph, SupervisorState
from utils.checkpointer import get_memory_saver
from langgraph.graph import StateGraph, START, END
from typing import Literal


# Initialize FastAPI app
app = FastAPI(title="LangGraph Multi-Agent API")

# Initialize checkpointer
# 只创建一次、跨请求复用
checkpointer = get_memory_saver()

# create_business_agent_graph()搭建了一个business_query节点 但这个函数返回的是 return workflow 而不是编译好的图
# 这个项目的核心设计决策就是图定义和编译分离 checkpointer的使用权取决于调用方
# 紧接着 .compile 就是将workflow编译成一个可执行的对象 business_graph
business_graph = create_business_agent_graph().compile(checkpointer=checkpointer)
database_graph = create_database_agent_graph().compile(checkpointer=checkpointer)


# Initialize multi-agent system with supervisor (for automatic routing)
def route_supervisor(state: SupervisorState) -> Literal["business_agent", "database_agent", "__end__"]:
    """Route based on supervisor's decision."""
    next_choice = state.get("next", "").lower()
    if next_choice == "finish":
        return "__end__"
    elif next_choice == "business_agent":
        return "business_agent"
    elif next_choice == "database_agent":
        return "database_agent"
    else:
        return "__end__"

# business_graph和database_compiled都是生产一个CompiledStateGraph对象
# 只是二者用于不同的地方(用途不一样) 本质没有区别

# Create multi-agent system with supervisor
supervisor_workflow = create_supervisor_graph()
business_workflow = create_business_agent_graph()
database_workflow = create_database_agent_graph()

# Compile sub-agents
supervisor_compiled = supervisor_workflow.compile(checkpointer=checkpointer)
business_compiled = business_workflow.compile(checkpointer=checkpointer)
database_compiled = database_workflow.compile(checkpointer=checkpointer)

# 创建一个父图构造器 SupervisorState参数是状态 schema 里面有message和next两个字段
parent_workflow = StateGraph(SupervisorState)
parent_workflow.add_node("supervisor", supervisor_compiled)
parent_workflow.add_node("business_agent", business_compiled)
parent_workflow.add_node("database_agent", database_compiled)
parent_workflow.add_edge(START, "supervisor")
parent_workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "business_agent": "business_agent",
        "database_agent": "database_agent",
        "__end__": END
    }
)
parent_workflow.add_edge("business_agent", END)
parent_workflow.add_edge("database_agent", END)

# 变成可运行的 CompiledStateGraph
multi_agent_graph = parent_workflow.compile(checkpointer=checkpointer)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for direct agent queries (manual routing)."""
    message: str
    thread_id: Optional[str] = None
    agent_type: str = "business"  # "business" or "database"


class AutoQueryRequest(BaseModel):
    """Request model for automatic routing via supervisor."""
    message: str
    thread_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str
    thread_id: str
    agent_type: str


class ConversationHistory(BaseModel):
    """Model for conversation history."""
    messages: List[Dict[str, str]]
    thread_id: str


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "LangGraph Multi-Agent API",
        "routing_modes": {
            "automatic": "POST /query/auto - Supervisor routes to appropriate agent",
            "manual": "POST /query - Client specifies agent_type"
        },
        "available_agents": ["business", "database", "supervisor"]
    }


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query a specific agent directly (manual routing).

    Client specifies which agent to use via agent_type parameter.

    IMPORTANT:
    - If thread_id is provided, continues existing conversation
    - If thread_id is None, starts new conversation with generated thread_id
    - Thread ID is passed in config, not in state
    """
    try:
        # Generate thread_id if not provided
        thread_id = request.thread_id or str(uuid.uuid4())

        # Select the appropriate agent
        if request.agent_type == "business":
            graph = business_graph
        elif request.agent_type == "database":
            graph = database_graph
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_type: {request.agent_type}. Must be 'business' or 'database'"
            )

        # IMPORTANT: Thread ID goes in config
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

        # Extract the response
        response_message = result["messages"][-1].content

        return QueryResponse(
            response=response_message,
            thread_id=thread_id,
            agent_type=request.agent_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/auto", response_model=QueryResponse)
async def query_agent_auto(request: AutoQueryRequest):
    """
    Query with automatic routing via supervisor (like LangGraph Studio).

    The supervisor analyzes the query and automatically routes to the
    appropriate agent (business or database).

    IMPORTANT:
    - Supervisor makes routing decision based on query content
    - Same behavior as LangGraph Studio
    - If thread_id is provided, continues existing conversation
    - If thread_id is None, starts new conversation with generated thread_id
    """
    try:
        # 客户端带来thread_id就用这个会话id聊 没带就生成一个。
        thread_id = request.thread_id or str(uuid.uuid4())

        # 根据thread_id来找到上一次的快照。
        config = {"configurable": {"thread_id": thread_id}}

        # multi_agent_graph(schema)里的schema决定了这个图状态长什么样，invoke返回的就是这个状态的最终值
        # 比如有(message，next)两个字段，返回的dict(字典)就有这两个键
        result = multi_agent_graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

        # Extract the response (last message)
        response_message = result["messages"][-1].content

        # Determine which agent was used (from state)
        # 根据返回的状态(result)来确定用哪个agent
        # .get(key, default) 是 dict 的方法：键存在就返回它的值，键不存在就返回默认值，这里是 "unknown"
        routed_agent = result.get("next", "unknown")


        # FastAPI 按 QueryResponse 序列化成 JSON 返回。
        return QueryResponse(
            response=response_message,
            thread_id=thread_id,
            agent_type=routed_agent  # Shows which agent supervisor chose
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_agent_stream(request: QueryRequest):
    """
    Stream responses from agent (for real-time updates).
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    if request.agent_type == "business":
        graph = business_graph
    elif request.agent_type == "database":
        graph = database_graph
    else:
        raise HTTPException(status_code=400, detail="Invalid agent_type")

    config = {"configurable": {"thread_id": thread_id}}

    # Stream the graph execution
    async def event_generator():
        for event in graph.stream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        ):
            yield f"data: {event}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/conversation/{thread_id}")
async def get_conversation_history(thread_id: str, agent_type: str = "business"):
    """
    Retrieve conversation history for a specific thread.
    """
    try:
        # Select agent
        if agent_type == "business":
            graph = business_graph
        elif agent_type == "database":
            graph = database_graph
        else:
            raise HTTPException(status_code=400, detail="Invalid agent_type")

        # Get state for this thread
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)

        if not state or not state.values.get("messages"):
            return {
                "messages": [],
                "thread_id": thread_id,
                "note": "No conversation history found"
            }

        # 将某个会话中的整段消息历史转换成模型API更容易接受的dict格式
        messages = []
        for msg in state.values["messages"]:
            messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })

        return ConversationHistory(
            messages=messages,
            thread_id=thread_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversation/{thread_id}")
async def delete_conversation(thread_id: str):
    """
    Delete conversation history for a specific thread.
    """
    # Note: MemorySaver doesn't have a delete method
    # For production with MySQL, you would delete from database
    return {
        "message": f"Conversation {thread_id} deletion requested",
        "note": "With MemorySaver, conversations are cleared on restart. Use MySQL for persistent storage."
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting LangGraph Multi-Agent API Server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
