
import gradio as gr
from rag_pipelineChat import load_pdf_and_create_chain, ask_question

with gr.Blocks() as app:
    gr.Markdown("## Chatbot PDF avec mémoire (RAG)")
    gr.Markdown("**Uploade un PDF et pose tes questions.**")
     
    pdf_file = gr.File(label="Upload ton PDF", file_types=[".pdf"], type="filepath")
    chatbot_ui = gr.Chatbot(label="Chat", type="messages")
    msg_input = gr.Textbox(placeholder="Pose une question...", label="Input")

    def handle_pdf_upload(file):
        status = load_pdf_and_create_chain(file)
        return [], status

    def user_ask(message, history):
        return "", ask_question(message)

    pdf_file.change(handle_pdf_upload, inputs=pdf_file, outputs=[chatbot_ui, gr.Textbox(visible=False)])
    msg_input.submit(user_ask, inputs=[msg_input, chatbot_ui], outputs=[msg_input, chatbot_ui])

app.launch(server_name="0.0.0.0", server_port=7860)
