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
        gr.Textbox(label="Input Query", lines=2, placeholder="Pose une question...")
    ],
    outputs=gr.Markdown(label="Output"),
    title="RAG Chatbot",
    description="Uploade un PDF et pose tes questions."
)

rag_application.launch(server_name="0.0.0.0", server_port= 7860)
