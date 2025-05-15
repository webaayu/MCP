
# 🧠 Understanding the Workflow: MCP + OpenAI + Gradio Chatbot

This application integrates a local tool server (via MCP), OpenAI function calling (GPT-4o), and a Gradio UI. Below is a complete breakdown of how the system works.

---

## 🔁 Overview

This chatbot does the following:

1. Starts a local server (e.g., `hr.py`) that defines tools.
2. Lists available tools via MCP.
3. Sends a user query to OpenAI with these tools.
4. Lets OpenAI choose and call a tool.
5. Executes that tool on the MCP server.
6. Sends the result back to OpenAI for a final response.
7. Displays everything in a Gradio interface.

---

## 🔧 Detailed Workflow

### 1. Load Environment and OpenAI Client

```python
load_dotenv()
api_key = os.getenv("openai_api_key")
openai_client = OpenAI(api_key=api_key)
```

This loads your API key and initializes the OpenAI client.

---

### 2. MCPClient Class

Manages tool server interaction through MCP.

---

### 3. Connecting to Server

```python
await client.connect_to_server("hr.py")
```

- Launches an MCP-compatible tool server.
- Connects to it via stdin/stdout using `stdio_client`.
- Initializes a session to interact with the tools.

---

### 4. Processing User Query

#### a. List Available Tools

```python
tool_response = await self.session.list_tools()
```

Each tool includes name, description, and input schema.

#### b. Convert Tools to OpenAI Format

```python
tool_specs = [{
  "type": "function",
  "function": {
    "name": tool.name,
    "description": tool.description,
    "parameters": tool.inputSchema
  }
} for tool in tools]
```

#### c. First GPT Call

```python
response = openai_client.chat.completions.create(...)
```

GPT gets the user input + tool list.

#### d. Extract Tool Call

```python
fn_name = tool_call.function.name
fn_args = eval(tool_call.function.arguments)
```

#### e. Call Tool on MCP Server

```python
tool_result = await self.session.call_tool(fn_name, fn_args)
```

#### f. Send Tool Result Back to GPT

```python
messages.append({ ... tool result ... })
final_response = openai_client.chat.completions.create(...)
```

GPT now gives the final answer using tool results.

---

### 5. Gradio UI

```python
gr.Interface(
    fn=ask_openai,
    inputs=gr.Textbox(...),
    ...
).launch()
```

The chatbot UI sends input to `ask_openai()`, which wraps the async query handler.

---

## ✅ Summary

```text
User input → GPT (with tools) → GPT chooses tool → Tool is executed → Result → GPT final response → UI
```

---

## 💡 Benefits

- Dynamic tool usage via GPT function calling.
- Plug-and-play architecture using MCP tools.
- Fully async + user-friendly UI via Gradio.

---

Happy building! 🛠️
