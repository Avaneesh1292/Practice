from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_chunk_pdf():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)