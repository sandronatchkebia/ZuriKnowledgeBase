from rag_builder import load_index, add_new_document
from llama_index.core.query_engine import RetrieverQueryEngine
import openai
import gradio as gr
from dotenv import load_dotenv
import os
import json

load_dotenv(override=True)
client = openai.OpenAI()

index = load_index()
retriever = index.as_retriever()

system_prompt = """
You are a concise, no-fluff assistant that answers questions based strictly on the content of academic papers in the knowledge base.

- Provide direct, specific answers — avoid introductions, summaries, or high-level overviews.
- Do not restate the question. Do not suggest the user read the paper.
- Only include information explicitly found in the source documents.
- If the answer is not in the context, say "I couldn’t find that in the papers I have access to."
- When listing or comparing, use bullet points or numbered lists.
- Do not speculate. Be clear when information is missing or uncertain.

If a user asks to add a new paper, call the appropriate function tool with the exact file path.
"""

add_new_paper_tool = {
    "type": "function",
    "function": {
        "name": "add_new_paper",
        "description": "Add an uploaded paper to the knowledge base by file name.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Exact filename of the uploaded paper, e.g. 'transformers.pdf'"
                }
            },
            "required": ["filename"]
        }
    }
}

rag_search_tool = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "Retrieve relevant academic content to answer the user's question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to answer using academic documents."
                }
            },
            "required": ["query"]
        }
    }
}

tools = [add_new_paper_tool, rag_search_tool]

uploaded_files = {}

def upload_file(file):
    try:
        os.makedirs("data", exist_ok=True)
        source_path = file.name  # already a full path
        destination_path = os.path.join("data", os.path.basename(file.name))

        with open(source_path, "rb") as src, open(destination_path, "wb") as dst:
            dst.write(src.read())

        uploaded_files[os.path.basename(file.name)] = destination_path
        return f"File uploaded: {os.path.basename(file.name)}"
    except Exception as e:
        return f"Error: {str(e)}"

def chat_with_rag(message, history):
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({
        "role": "user",
        "content": message
    })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )

    msg = response.choices[0].message

    # Handle tool call if triggered
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        if tool_name == "add_new_paper":
            print("calling add_new_paper")
            filename = tool_args.get("filename")
            path = uploaded_files.get(filename)
            if not path:
                return f"I couldn’t find the file '{filename}'. Make sure it was uploaded."

            result = add_new_document(path)
            tool_response = f"Paper indexed: {result}"
            print(tool_response)

            messages.append({
                "role": "assistant",
                "tool_calls": msg.tool_calls
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response
            })

            followup = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return followup.choices[0].message.content

        elif tool_name == "rag_search":
            print("calling rag_search")
            query = tool_args.get("query")
            context_nodes = retriever.retrieve(query)
            context = "\n\n".join([n.text for n in context_nodes])

            # Append the tool_call and its response
            messages.append({
                "role": "assistant",
                "tool_calls": msg.tool_calls
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": context
            })

            # Final model call to generate the answer using the context
            followup = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return followup.choices[0].message.content

    return msg.content

with gr.Blocks(
    title="Zuri Knowledge Base",
    css="""
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f4ff;
        }

        .gradio-container {
            background-color: #f0f4ff;
        }

        .gr-button {
            background-color: #2563eb !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600;
        }

        .gr-button:hover {
            background-color: #1e40af !important;
        }

        .gr-box, .gr-chatbot {
            border: 1px solid #dbeafe;
            background-color: white !important;
            border-radius: 1rem !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }

        .gr-textbox textarea {
            background-color: #eff6ff !important;
            border-radius: 0.5rem !important;
            border: 1px solid #bfdbfe !important;
            padding: 0.75rem;
        }

        .gr-markdown h1 {
            color: #1e3a8a;
            font-weight: 700;
        }

        .gr-markdown {
            margin-bottom: 1.5rem;
        }
    """
) as demo:
    gr.Markdown("""
    # 🔷 Zuri Knowledge Base  
    _Ask AI research questions, powered by your uploaded papers._
    """)

    with gr.Row():
        with gr.Column(scale=1):
            uploader = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
            upload_output = gr.Textbox(label="Upload Status", interactive=False)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 Knowledge Base Chat")
            user_input = gr.Textbox(
                label="Ask a Question",
                placeholder="Ask something about the papers...",
                lines=2
            )
            submit_btn = gr.Button("Submit 🚀")

    state = gr.State([])

    def handle_upload(file):
        return upload_file(file)

    def handle_submit(message, history):
        try:
            response = chat_with_rag(message, history)
            history.append((message, response))
            return "", history, history
        except Exception as e:
            return "", history, history + [(message, f"Error: {str(e)}")]

    uploader.change(fn=handle_upload, inputs=uploader, outputs=upload_output)
    submit_btn.click(fn=handle_submit, inputs=[user_input, state], outputs=[user_input, chatbot, state])
    user_input.submit(fn=handle_submit, inputs=[user_input, state], outputs=[user_input, chatbot, state])

if __name__ == "__main__":
    demo.launch()