
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.memory import ConversationBufferMemory
import torch
import re

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
db = None
retriever = None
qa_chain = None
pipe = None
history = []
def doc_process(texte):
    lignes = texte.split('\n')
    resultat = []
    for i, ligne in enumerate(lignes):
        ligne_actuelle = ligne.strip()
        if ligne_actuelle == "":
            resultat.append("")
            continue
        if ligne_actuelle.isupper() or re.match(r"^\d+(\.\d+)*", ligne_actuelle):
            resultat.append(ligne_actuelle)
            continue
        if (
            resultat
            and not resultat[-1].endswith(('.', ':', '!', '?'))
            and not resultat[-1].isupper()
        ):
            resultat[-1] += " " + ligne_actuelle
        else:
            resultat.append(ligne_actuelle)
    return '\n'.join(resultat)

def get_local_llm():
    global pipe
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=1024,
    )
    return HuggingFacePipeline(pipeline=pipe)

def get_retriever(pdf_file):
    global db, retriever
    loader = PyMuPDFLoader(pdf_file.name)
    data = loader.load()

    cleaned_documents = []
    for doc in data:
        texte_nettoye = doc_process(doc.page_content)
        doc_nettoye = Document(page_content=texte_nettoye, metadata=doc.metadata)
        cleaned_documents.append(doc_nettoye)

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)

    db = Chroma(collection_name="full_documents", embedding_function=embeddings_model)
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=db,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    retriever.search_kwargs = {"k": 4}
    retriever.add_documents(cleaned_documents)

def build_chat_messages(context, question, history_pairs):
    messages = [
        {"role": "system", "content": "Tu es un assistant expert scientifique. Tu rédiges des réponses complètes, précises et chiffrées si possible. Utilise des listes à puces si nécessaire, et n'hésite pas à inclure des formules LaTeX si cela est pertinent."}
    ]
    for user, assistant in history_pairs:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    # Dernière question avec contexte
    messages.append({
        "role": "user",
        "content": f"Contexte extrait du document :\n{context}\n\nQuestion : {question}"
    })
    return messages

def load_pdf_and_create_chain(file):
    global retriever, pipe, history
    history = []
    get_local_llm()
    get_retriever(file)
    return "PDF chargé et prêt à répondre aux questions."

def ask_question(query):
    global history, pipe
    if pipe is None or retriever is None:
        return "Veuillez d'abord uploader un PDF."

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs[:2]])

    messages = build_chat_messages(context, query, history)
    
    output = pipe(messages, max_new_tokens=512)
    response = output[0]["generated_text"][-1]

    history.append({"role": "user", "content": query})
    history.append(response)
    print("\n\n")
    print(messages)
    print("\n\n")
    return history
