# RAG QA on any PDF document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

from dotenv import load_dotenv
load_dotenv(override=True)

import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings("ignore")

## LLM
def get_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=256,
    )

## Document loader
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

## Text splitter
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(data)

## Embedding model
def openai_embedding():
    return OpenAIEmbeddings(model="text-embedding-3-small")

## Vector db
def vector_database(chunks):
    embedding_model = openai_embedding()

    # texts = []
    # metadatas = []

    # for chunk in chunks:
    #     text = chunk.page_content.strip()
    #     if text:
    #         texts.append(text)
    #         metadatas.append(chunk.metadata)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        # metadatas=metadatas,
    )
    return vectordb

## Retriever
def retriever(file_path):
    docs = document_loader(file_path)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

## QA Chain
def retriever_qa(file_path, query):
    llm = get_llm()
    retriever_obj = retriever(file_path)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )
    response = qa.invoke(query)
    return response["result"]

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
    ],
    outputs=gr.Textbox(label="Response Here...", lines=5),
    title="QA on your uploaded PDF",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
)

rag_application.launch()
