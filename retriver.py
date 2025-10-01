from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
splits = text_splitter.split_documents(data)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectordb = Chroma.from_documents(documents=splits,embedding=embeddings)

question = "What are the approaches to Task Decomposition"

llm = ChatOllama(model="llama3.1:latest", temperature=0)

retriver_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriver_from_llm.invoke(question)

print(len(unique_docs))

