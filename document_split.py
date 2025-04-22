from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pickle
import os

PDFPATH = "./files/US-constitution.pdf"
PDFPATH = "./files/nepali-law.pdf"


def get_or_create_splits(cache_file="splits_cache.pkl"):
    # Check if cache exists
    if os.path.exists(cache_file):
        print("Loading splits from cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    # If no cache, create new splits
    print("Creating new splits...")
    loader = PyPDFLoader(PDFPATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )

    splits = text_splitter.split_documents(docs)
    
    # Save to cache
    with open(cache_file, "wb") as f:
        pickle.dump(splits, f)
    
    return splits



def return_splits():
    return get_or_create_splits()




