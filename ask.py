import chromadb
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama


load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="growing_vegetables")

user_query = input("what do you want to know about growing vegetables")

#Search on a small subset of similar documents
# results = collection.query(
#     query_texts=[user_query],
#     n_results=4
# )

#Search the Entire correction
results = collection.get()

print(results['documents'])

llm = ChatOllama(model="llama3.1:latest",temperature=0)

system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
--------------------
The data:
"""+str(results['documents'])+"""
"""

messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_query}    
    ]
response = llm.invoke(messages)

print("\n\n---------------------\n\n")

print(response.content)