
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
