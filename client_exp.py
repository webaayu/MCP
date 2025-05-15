import asyncio
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastmcp import Client
import gradio as gr

load_dotenv()
api_key = os.getenv("openai_api_key")
openai_client = OpenAI(api_key=api_key)


from fastmcp.client import Client
from fastmcp.client.transports import SSETransport  # ✅ explicitly use SSE
import asyncio



async def process_query(query: str) -> str:
    async with Client(transport=SSETransport("http://localhost:8000/sse")) as client:
        tools = await client.list_tools()
        tool_specs = []
        for tool in tools:
            tool_specs.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            })
        messages = [{"role": "user", "content": query}]
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tool_specs
        )
        choice = response.choices[0].message
        messages.append(choice)
        if choice.tool_calls:
            tool_call = choice.tool_calls[0]
            fn_name = tool_call.function.name
            fn_args = eval(tool_call.function.arguments)
            # Call the tool
            tool_result = await client.call_tool(fn_name, fn_args)
            print("Tool result:",tool_result)
            # If tool_result is a list, extract the first element or join them
            if isinstance(tool_result, list):
            # Join content from each result if they have a 'content' attribute
                content = "\n".join([r.content for r in tool_result if hasattr(r, "content")])
                print("content:",content)
            else:
                content = tool_result.content  # fallback if it's a single result
                print("content:",content)

            # Append tool response to messages
            messages.append({
               "role": "tool",
               "tool_call_id": tool_call.id,
               "content": tool_result
            })

            final_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            result = final_response.choices[0].message.content
            return result
        return "No tool call was made by the model."

def ask_openai(user_input):
    return asyncio.run(process_query(user_input))

demo = gr.Interface(
    fn=ask_openai,
    inputs=gr.Textbox(lines=2, placeholder="Ask about HR tools or automation..."),
    outputs="text",
    title="MCP + OpenAI Chatbot",
    description="Chat with tools exposed via MCP using OpenAI function calling."
)

demo.launch()

