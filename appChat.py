
import gradio as gr
from rag_pipelineChat import load_pdf_and_create_chain, ask_question

with gr.Blocks() as app:
    gr.Markdown("## Chatbot PDF avec mémoire (RAG)")
    gr.Markdown("**Uploade un PDF et pose tes questions. Le chatbot se souvient du contexte !**")
     
    # Upload PDF
    pdf_file = gr.File(label="📄 Upload ton PDF", file_types=[".pdf"], type="filepath")

    # Composants
    chatbot_ui = gr.Chatbot(label="Chatbot", type="messages")
    msg_input = gr.Textbox(placeholder="Pose une question...", label="💬 Ta question")

    # Charger le fichier
    def handle_pdf_upload(file):
        print('---------start--------')
        status = load_pdf_and_create_chain(file)
        return [], status  # Vide l'historique, retourne un message d'état

    # Pose une question
    def user_ask(message, history):
        response = ask_question(message)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return "", history

    pdf_file.change(handle_pdf_upload, inputs=pdf_file, outputs=[chatbot_ui, gr.Textbox(visible=False)])
    msg_input.submit(user_ask, inputs=[msg_input, chatbot_ui], outputs=[msg_input, chatbot_ui])

app.launch(server_name="0.0.0.0", server_port=7860)
