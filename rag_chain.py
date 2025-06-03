from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from vector_store import search_document
from pinecone_db import search_document
from langchain_groq import ChatGroq


model = ChatGroq(model='llama3-70b-8192')


# prompt = ChatPromptTemplate.from_messages([
#     ("system","You are a law expert of Nepal. Answer the question given by the user using the following context. \n {context}"),
#     ("human","{input}")
# ])

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a medical expert for emergencies. Answer the question given by the user using the following context in a simple and understandable format. \n {context}"),
    ("human","{input}")
])


def formatter(query):
    similar_docs = search_document(query)
    output = ""
    for docs in similar_docs:
        output += docs.page_content
        output += " \n\n"
    return output



chain = (
    {"context":RunnableLambda(formatter),"input":RunnablePassthrough()}|
    prompt |
    model |
    StrOutputParser()
)


resp = chain.invoke("What should i do if the person seems to be bitten by a snake")

print(resp)