
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline, BitsAndBytesConfig, LlamaTokenizer, MistralForCausalLM
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.memory import ConversationBufferMemory
import torch
import re

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
db = None
retriever = None
qa_chain = None
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
    model_id = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = MistralForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        max_new_tokens=1024,
        temperature=0.8,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
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

custom_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=""" 
Tu es un assistant expert scientifique. Tu rédiges des réponses complètes et précises. Utilise des listes à puces si nécessaire, et n'hésite pas à inclure des formules LaTeX si cela est pertinent.

Contexte extrait du document :
{context}

Question de l'utilisateur :
{question}

Réponse en français :
"""
)

def load_pdf_and_create_chain(file):
    global retriever, qa_chain, memory
    get_retriever(file)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_local_llm(),
        chain_type="stuff", 
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h : h,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": custom_prompt_template}
    )
    print('OK------------')
    return "PDF chargé et prêt à répondre aux questions."

def ask_question(query):
    global qa_chain, history
    if qa_chain is None:
        return "Veuillez d'abord uploader un PDF."
    result = qa_chain.invoke({"question":query}, {"chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]
