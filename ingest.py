import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv()

# Configuration
DOC_DIR = os.getenv('DOC_DIR')
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"

def main():
    # Load documents (excluding .png files)
    loader = DirectoryLoader(
        DOC_DIR,
        glob="**/*.md",  # Only load markdown files
        show_progress=True
    )
    documents = loader.load()
    
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