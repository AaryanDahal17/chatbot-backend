import os
from getpass import getpass
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
# from document_split import return_splits

load_dotenv()

# Get Pinecone API key from environment or prompt
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-pinecone-embeddings"

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,  # dimension depends on Pinecone embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Initialize Pinecone embeddings model
embeddings = PineconeEmbeddings(model="multilingual-e5-large")

# Initialize vector store with Pinecone embeddings
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# # Example documents
# documents = return_splits()

# # Add documents to Pinecone vector store
# vector_store.add_documents(documents)

# Query
# query = "what to do when a person is going through cardiac arrest"
# results = vector_store.similarity_search(query, k=3)


def search_document(query):
    return vector_store.similarity_search(query, k=3)

# for doc in results:
#     print(doc.page_content)
#     print("-" * 100)
