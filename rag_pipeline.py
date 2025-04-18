# http://localhost:7860/?

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline, BitsAndBytesConfig

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, MistralForCausalLM
import torch

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

import re

# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.prompts import PromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain


def doc_process(texte):
    lignes = texte.split('\n')
    resultat = []

    for i, ligne in enumerate(lignes):
        ligne_actuelle = ligne.strip()

        # Ligne vide (paragraphe ou section)
        if ligne_actuelle == "":
            resultat.append("")
            continue

        # Si c'est un titre (tout en majuscules ou numérotation)
        if ligne_actuelle.isupper() or re.match(r"^\d+(\.\d+)*", ligne_actuelle):
            resultat.append(ligne_actuelle)
            continue

        # Fusion avec ligne précédente s’il n’y a pas de ponctuation
        if (
            resultat
            and not resultat[-1].endswith(('.', ':', '!', '?'))
            and not resultat[-1].isupper()
        ):
            resultat[-1] += " " + ligne_actuelle
        else:
            resultat.append(ligne_actuelle)

    return '\n'.join(resultat)

# Globals for session
# deepseek-ai/DeepSeek-Coder-1.3B-Instruct
db = None
retriever = None

def get_local_llm():
    # model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # model_id = "HuggingFaceH4/zephyr-7b-beta"
    # model_id = "deepseek-ai/deepseek-llm-7b-chat"
    model_id = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     offload_folder="D:\LLMOffLoad"
    # )
    tokenizer = LlamaTokenizer.from_pretrained('NousResearch/Nous-Hermes-2-Mistral-7B-DPO', trust_remote_code=True)
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

    # pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")

    return HuggingFacePipeline(pipeline=pipe)

def get_retriever(pdf_file):
    global db, retriever
    # loader = UnstructuredPDFLoader(pdf_file.name)
    loader = PyMuPDFLoader(pdf_file.name)
    data = loader.load()
    
    # Nettoyage du texte contenu dans chaque Document
    cleaned_documents = []
    for doc in data:
        texte_nettoye = doc_process(doc.page_content)
        doc_nettoye = Document(page_content=texte_nettoye, metadata=doc.metadata)
        cleaned_documents.append(doc_nettoye)

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len
    # )

    # chunks = text_splitter.split_documents(cleaned_documents)
    # embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # db = FAISS.from_documents(chunks, embeddings_model)
    # retriever = db.as_retriever(search_kwargs={"k": 5})
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
    


def retriever_qa(file, query):
    global retriever
    get_retriever(file)
    if retriever is None:
        return "Veuillez d'abord uploader et traiter un PDF."
    
#     prompt_template = """<!-- Tu es un expert scientifique. Rédige des réponses bien structurées et techniquement précises. Utilise des listes à puces si nécessaire et des formules mathématiques en format LaTeX si utiles.

# Contexte extrait du document :

# {context} -->

# Question : {question}

# Réponse en français :
#     """

    prompt_template = """<!-- <|im_start|>system
Tu es un assistant scientifique expert. Tu rédiges des réponses complètes et précises. Utilise des listes à puces si nécessaire, et n'hésite pas à inclure des formules LaTeX si cela est pertinent.
<|im_end|>

<|im_start|>user
Voici un extrait d'un document : 
 {context} 

Question : {question}
<|im_end|> 

<|im_start|>assistant --> """

    

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=get_local_llm(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )
    return qa.invoke(query)['result']

