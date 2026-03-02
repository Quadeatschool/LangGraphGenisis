# LangGraph Boilerplate

A minimal, extensible starting point for building stateful LLM agents with [LangGraph](https://github.com/langchain-ai/langgraph).

---

## Project Structure

```
your-project/
├── .venv/                      # Virtual environment (created by you, not committed)
├── .env                        # API keys (never commit this)
├── .gitignore
├── langgraph_boilerplate.py    # Main boilerplate
└── README.md                   # This file
```

---

## Setup

### 1. Create a Virtual Environment

A virtual environment keeps this project's dependencies isolated from your system Python.

```bash
# Create the venv
python -m venv .venv

# Activate it
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

You'll see `(.venv)` in your terminal prompt when it's active. Always activate before working on this project.

### 2. Install Dependencies

```bash
pip install langgraph langchain-openai langchain-core
```

### 3. Set Your API Key

**Option A — Environment variable (quick):**
```bash
export OPENAI_API_KEY=your_key_here    # macOS/Linux
set OPENAI_API_KEY=your_key_here       # Windows
```

**Option B — `.env` file (recommended):**
```bash
pip install python-dotenv
```
Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```
Then load it at the top of your script:
```python
from dotenv import load_dotenv
load_dotenv()
```

> ⚠️ Never commit your `.env` file or API key. Add `.env` to your `.gitignore`.

### 4. Run It

```bash
python langgraph_boilerplate.py
```

### 5. Deactivate When Done

```bash
deactivate
```

---

## Graph Structure

```
START → preprocess → call_llm → should_continue() → postprocess → END
                                                   ↘ END (if "DONE" in response)
```

| Node | Purpose |
|------|---------|
| `preprocess` | Validate or enrich input before the LLM sees it |
| `call_llm` | Sends messages to GPT, appends the response to state |
| `postprocess` | Parse, log, or act on the LLM's response |
| `should_continue` | Conditional router — decides what happens after the LLM responds |

---

## Extending the Boilerplate

### Add a new node
```python
def my_node(state: State) -> dict:
    # Do something with state
    return {"messages": [...]}  # Return only what changed

graph.add_node("my_node", my_node)
graph.add_edge("call_llm", "my_node")
```

### Add state fields
```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str       # add any fields you need
    step_count: int
```

### Add tools to the LLM
```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """Describe what this tool does."""
    return "result"

llm_with_tools = llm.bind_tools([my_tool])
```

### Visualize the graph
```python
# Prints a Mermaid diagram of your graph
print(app.get_graph().draw_mermaid())
```

---

## File Safety

Your local files are **safe by default**. Agents can only access your file system if you explicitly give them a file-related tool. No tool = no access.

| Scenario | Safety |
|----------|--------|
| LLM + structured output only | ✅ Safe |
| Agent with scoped `read_file` tool | ⚠️ Use with care |
| Agent with shell / exec tools | 🚫 Dangerous — avoid unless sandboxed |

---

## .gitignore Recommendation

```
.venv/
.env
__pycache__/
*.pyc
```

---

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [OpenAI API Docs](https://platform.openai.com/docs/api-reference)
