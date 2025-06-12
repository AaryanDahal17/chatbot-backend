from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from vector_store import search_document
from pinecone_db import search_document
from langchain_groq import ChatGroq


# model = ChatGroq(model='llama3-70b-8192')

model = ChatGroq(model='llama-3.3-70b-versatile')


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


def run(prompt):
    resp = chain.invoke(prompt)
    
    msgs = [SystemMessage(content=f""" You are a medical expert, your task is to review the answer provided by the user.
    Make sure the user's answer is correct, make some changes if required and give me the updated answer in a proper format with suitable spacing and
    paragraphing when required. If possible, try to give a step by step answer for the question.
    Here is the question : {prompt}
    """),
    
    HumanMessage(content=resp)
    ]

    # print("CONTEXT : ")
    # print(msgs)
    # print()

    new_resp = model.invoke(msgs)

    return new_resp.content




# our_resp = run("What to do in case of a heart attack?")

# print("AI:")
# print()
# print(our_resp)
