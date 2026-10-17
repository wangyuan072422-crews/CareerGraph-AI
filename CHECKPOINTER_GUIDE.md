# Checkpointer Usage Guide: When to Pass a Checkpointer

## Overview

The **#1 cause of memory issues** in LangGraph is passing custom checkpointers when the deployment platform already manages persistence automatically.

**Key Rule**: Only pass a custom checkpointer when running graphs in **your own custom application code**.

---

## Quick Reference

| Deployment Method | Checkpointer? | Why |
|------------------|--------------|-----|
| **LangSmith Deployments** (Studio, Cloud, Agent Server) | ❌ **NO** | Platform handles it automatically |
| **Custom Application** (your own Python script/FastAPI) | ✅ **YES** | You manage persistence |

---

## When NOT to Pass a Checkpointer

### ❌ LangSmith Deployments (Studio, Cloud, Agent Server)

If you're using any LangSmith deployment platform, **do not pass a custom checkpointer**:

- `langgraph dev` (LangGraph Studio)
- LangSmith Deployments (formerly LangGraph Cloud)
- Agent Server

**These platforms handle checkpointing automatically behind the scenes.**

### ✅ CORRECT - Platform Deployments
```python
# graphs/multi_agent_system.py

workflow = create_business_agent_graph()

# DO NOT pass checkpointer - platform handles it
graph = workflow.compile()

__all__ = ["graph"]
```

### ❌ WRONG - Platform Deployments
```python
# graphs/multi_agent_system.py

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)  # ❌ Conflicts!

# Error: "LangGraph already has inbuilt Memory saver, it will be ignored"
```

**Why?**

LangSmith Deployments provide managed persistence. When you pass a custom checkpointer, you're telling the platform to use two different persistence mechanisms, which causes a conflict.

**Reference**: [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/persistence) - "Agent Server handles checkpointing automatically"

---

## When TO Pass a Checkpointer

### ✅ Custom Applications (Self-Hosted)

Only pass a custom checkpointer when you're building **your own application** that embeds LangGraph:

- Direct Python scripts (`python main.py`)
- Your own FastAPI/Flask server
- Self-hosted custom applications

In these cases, **you are responsible for persistence**.

### ✅ CORRECT - Custom Python Script
```python
# main.py - your custom script

from langgraph.checkpoint.memory import MemorySaver

workflow = create_business_agent_graph()

# DO pass checkpointer - you manage persistence
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

### ✅ CORRECT - Custom Self-Hosted Server
```python
# server.py - your custom FastAPI server

from langgraph.checkpoint.memory import MemorySaver
# or for production:
# from langgraph.checkpoint.postgres import PostgresSaver

workflow = create_business_agent_graph()

# DO pass checkpointer - you manage persistence
checkpointer = MemorySaver()  # or PostgresSaver for production
graph = workflow.compile(checkpointer=checkpointer)
```

### ❌ WRONG - Custom Applications
```python
# main.py or server.py

workflow = create_business_agent_graph()

# No checkpointer = no persistence!
graph = workflow.compile()  # ❌ Conversations not saved
```

**Why?**

Custom applications don't have built-in persistence infrastructure. You must explicitly provide a checkpointer for conversational memory to work.

---

## Common Mistakes

### Mistake 1: Passing Checkpointer to Platform Deployments

**Symptom**: Error message "LangGraph already has inbuilt Memory saver, it will be ignored"

**Cause**: Passing a custom checkpointer when using LangSmith Deployments (Studio, Cloud, Agent Server)

```python
# graphs/multi_agent_system.py (loaded by langgraph dev)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)  # ❌
```

**Fix**: Remove the checkpointer - platform handles it automatically
```python
# graphs/multi_agent_system.py
graph = workflow.compile()  # ✅ Platform handles persistence
```

---

### Mistake 2: Not Using Checkpointer in Custom Applications

**Symptom**: Conversations lost, no memory persistence

**Cause**: Not passing a checkpointer when building your own custom application

```python
# main.py or your custom server.py
graph = workflow.compile()  # ❌ No persistence
```

**Fix**: Pass a checkpointer explicitly
```python
# main.py or your custom server.py
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)  # ✅
```

---

### Mistake 3: Using Same Graph Definition for Both

**Problem**: One graph file needs to work for both platform deployments and custom applications

**Solution**: Always compile without checkpointer in your graph definition. Add checkpointer separately in custom applications.

```python
# graphs/multi_agent_system.py (shareable)
def create_multi_agent_graph():
    """Return uncompiled workflow"""
    workflow = create_business_agent_graph()
    return workflow

# For platform deployment - compile without checkpointer
graph = create_multi_agent_graph().compile()
__all__ = ["graph"]
```

```python
# main.py (custom application)
from graphs.multi_agent_system import create_multi_agent_graph
from langgraph.checkpoint.memory import MemorySaver

# Compile with checkpointer for custom application
workflow = create_multi_agent_graph()
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

---

## File Organization

```
your_project/
├── graphs/
│   └── multi_agent_system.py   # Graph definition (no checkpointer)
└── main.py                      # Custom app (add checkpointer here)
```

### graphs/multi_agent_system.py
- **Purpose**: Graph definition for any deployment
- **Checkpointer**: ❌ NO
- **Why**: Platform handles it; custom apps add it separately

### main.py (if building custom application)
- **Purpose**: Your custom Python script/server
- **Checkpointer**: ✅ YES (MemorySaver or PostgresSaver)
- **Why**: You manage persistence

---

## Testing Your Setup

### Test Platform Deployment (Studio)
```bash
# Should work without errors
langgraph dev

# Open http://localhost:8123
# Send message: "Hello"
# Send follow-up: "What did I say?"
# Should maintain context ✅ (Studio handles persistence automatically)
```

### Test Custom Application
```bash
# If you built your own script with checkpointer
python main.py

# Should see conversation with context maintained ✅
# Checkpointer you passed handles persistence
```

---

## Debugging

### Error: "LangGraph already has inbuilt Memory saver, it will be ignored"

**Full Error Message:**
```
ValueError: Heads up! Your graph '{name}' from '{path}' includes a custom 
checkpointer (type <class '...'). With LangGraph API, persistence is handled 
automatically by the platform, so providing a custom checkpointer here isn't 
necessary and will be ignored when deployed.
```

**What this means:**
You're running `langgraph dev` (LangGraph Studio) AND passing an explicit checkpointer in your graph definition. This creates a conflict because Studio provides its own persistence layer.

**Cause**: Using checkpointer with Studio

**Why it happens:**
- Your graph file (e.g., `graphs/multi_agent_system.py`) includes `checkpointer=MemorySaver()` or similar
- You're running with `langgraph dev` command
- Studio tries to use its built-in persistence, but you've also provided a checkpointer
- The conflict causes a ValueError that blocks execution

**Fix**: Remove checkpointer from graph file loaded by `langgraph dev`

```python
# In graphs/multi_agent_system.py (or wherever Studio loads your graph)

# ❌ REMOVE THIS:
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)  # Don't do this!

# ✅ USE THIS:
graph = workflow.compile()  # No checkpointer - Studio handles it
```

---

### "Context not maintained" / Follow-up questions don't work

**Symptoms:**
- First question works fine
- Follow-up questions don't maintain context
- Agent doesn't remember previous conversation
- Each query is treated as a new conversation

**Cause**: No checkpointer in custom application, OR missing thread_id in config

**Common scenarios:**
1. Running your custom application without passing checkpointer
2. Forgot to pass `config` with `thread_id` during invocation (applies to all deployments)

**Fix 1** (if running custom application): Add checkpointer to `compile()` call
```python
# main.py - your custom application
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

**Fix 2** (applies to all deployments): Ensure you're passing thread_id in config
```python
# Thread ID must be in config, not state
config = {"configurable": {"thread_id": "conversation-123"}}
result = graph.invoke({"messages": [...]}, config=config)
```

---

### "Memory lost on server restart"

**Symptoms:**
- Conversations work during a session
- After restarting the server/application, all conversation history is lost
- Thread IDs don't retrieve previous conversations

**Cause**: Using MemorySaver in custom application (stores in RAM only)

**Why it happens:**
MemorySaver stores checkpoints in memory, which is cleared when the process restarts. For production custom applications, you need persistent storage.

**Note**: This only applies to **custom applications**. If using LangSmith Deployments (Studio, Cloud, Agent Server), the platform provides persistent storage automatically.

**Fix** (for custom applications): Switch to PostgresSaver for persistent storage

```python
# Replace MemorySaver with PostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@host:port/database"
)
graph = workflow.compile(checkpointer=checkpointer)
```

**Setup PostgreSQL:**
```bash
# 1. Install package
pip install langgraph-checkpoint-postgres

# 2. Create database
psql -U postgres
CREATE DATABASE langgraph_db;

# 3. First run - call setup()
checkpointer.setup()  # Creates required tables
```

---

### Database Connection Errors (Custom Applications Only)

**Note**: This section only applies if you're building a custom application with a persistent checkpointer (PostgresSaver, etc.). If using LangSmith Deployments, the platform handles database connections automatically.

**Symptoms:**
- Certificate errors when connecting to database
- "SSL connection error" or similar
- Connection timeouts

**Common causes:**
1. **SSL/TLS certificate issues**
2. **Connection string format incorrect**
3. **Database not created or accessible**
4. **Firewall/network restrictions**

**Fix for SSL/certificate errors:**
```python
# Add SSL parameters to connection string if needed
connection_string = (
    "postgresql://user:password@host:port/database"
    "?sslmode=require"
    "&sslrootcert=/path/to/ca.pem"
)
```

**Or disable SSL verification for testing (not recommended for production):**
```python
connection_string = (
    "postgresql://user:password@host:port/database"
    "?sslmode=disable"
)
```

**Test connection independently:**
```python
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        user='postgres',
        password='password',
        database='langgraph_db'
    )
    print("✓ Database connection successful!")
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

---

## Summary

**The Key Rule:**

- **LangSmith Deployments** (Studio, Cloud, Agent Server) = ❌ Don't pass checkpointer
- **Custom Applications** (your own Python scripts/servers) = ✅ Pass checkpointer

**Different deployment methods have different persistence requirements!**

---

## See Also

- [README.md](README.md) - Full project documentation
- [graphs/multi_agent_system.py](graphs/multi_agent_system.py) - Platform deployment example
- [main.py](main.py) - Custom application example
- [LangGraph Persistence Docs](https://docs.langchain.com/oss/python/langgraph/persistence) - Official documentation
