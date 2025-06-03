from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pickle
import os

PDFPATH = "./files/US-constitution.pdf"
PDFPATH = "./files/nepali-law.pdf"
PDFPATH = "./files/Emergency_Care.pdf"


# splits_cache => nepali-law
# emer_splits_cache.pkl => emergency care


def get_or_create_splits(cache_file="emer_splits_cache.pkl"):
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
        chunk_size=2700,    #1000
        chunk_overlap=400,   #200 for general
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




