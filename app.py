# pip install -r requirements.txt
import sys
import io
import locale
import os
import gradio as gr

from rag_pipeline import get_retriever, retriever_qa
import torch
print(torch.cuda.get_device_name(0)) 


rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Markdown(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

rag_application.launch(server_name="0.0.0.0", server_port= 7860)

# with gr.Blocks() as app:
#     gr.Markdown("## Chatbot PDF avec mémoire (RAG)")
#     gr.Markdown("**Uploade un PDF et pose tes questions. Le chatbot se souvient du contexte !**")
     
#     # Sélection du fichier
#     pdf_file = gr.File(label="📄 Upload ton PDF", file_types=[".pdf"], type="filepath")

#     # Composant Chatbot
#     chatbot_ui = gr.Chatbot(type="messages")
#     msg_input = gr.Textbox(placeholder="Pose une question...", label="💬 Ta question")
    
#     # Envoyer une question
#     def user_ask(message, history, file):
#         response = retriever_chatbot(file, message)
#         history.append((message, response))
#         return "", history

#     msg_input.submit(user_ask, [msg_input, chatbot_ui, pdf_file], [msg_input, chatbot_ui])

# app.launch(server_name="0.0.0.0", server_port=7860)