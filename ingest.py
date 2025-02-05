import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv()

# Configuration
DOC_DIR = os.getenv('DOC_DIR')
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"

def load_documents():
    # List all files in directory
    all_files = []
    for root, _, files in os.walk(DOC_DIR):
        for file in files:
            if file.endswith(('.pdf', '.txt', '.md')):
                all_files.append(os.path.join(root, file))
    print(f"Found files: {all_files}")

    documents = []
    for file_path in all_files:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)  # Changed to PyPDFLoader
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            elif file_path.endswith('.md'):
                loader = UnstructuredMarkdownLoader(file_path)
            
            print(f"Loading {file_path}")
            doc = loader.load()
            documents.extend(doc)
            print(f"Successfully loaded {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return documents

def main():
    print(f"Scanning directory: {DOC_DIR}")
    documents = load_documents()
    
    if not documents:
        print(f"No documents found in {DOC_DIR}")
        return

    print(f"Loaded {len(documents)} documents")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Create vector store
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=CHROMA_PATH
    )

if __name__ == "__main__":
    main()