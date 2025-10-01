from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines"""

    def parse(self,text:str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None,lines))



loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
splits = text_splitter.split_documents(data)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
vectordb = Chroma.from_documents(documents=splits,embedding=embeddings)

#question = "What are the approaches to Task Decomposition"
question = "What does the course say about regression?"

llm = ChatOllama(model="llama3.1:latest", temperature=0)

output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

llm_chain = QUERY_PROMPT | llm | output_parser

retriver = MultiQueryRetriever(
    retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriver.invoke(question)

print(len(unique_docs))

