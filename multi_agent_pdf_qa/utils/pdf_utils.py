import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file) -> str:
    """Extract full text from a single PDF file using PyMuPDF."""
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks using LangChain's recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

