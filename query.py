from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
import subprocess

# Configuration
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-r1"

def query_ollama(prompt):
    print("\nSending prompt to Ollama:", prompt[:200] + "..." if len(prompt) > 200 else prompt)
    cmd = ["ollama", "run", LLM_MODEL, prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print("Error from Ollama:", result.stderr)
    return result.stdout

def rag_query(question: str) -> str:
    # Load vector DB
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL)
    )
    
    print("\nSearching for relevant context...")
    
    # Retrieve context
    results = vector_store.similarity_search(question, k=3)
    if not results:
        return "No relevant documents found in the database."
        
    print(f"\nFound {len(results)} relevant documents")
    context = "\n".join([doc.page_content for doc in results])
    print("\nContext length:", len(context), "characters")
    
    # Build RAG prompt
    template = """Answer the question based on the following context:

{context}

Question: {question}

Answer the question using only the information provided in the context above. If you cannot find the answer in the context, say "I cannot find the answer in the provided context."
"""
    
    prompt = PromptTemplate.from_template(template).format(
        context=context,
        question=question
    )
    
    # Get response
    response = query_ollama(prompt)
    if not response.strip():
        return "Warning: Received empty response from Ollama"
    return response

def main():
    # Get user query
    query = input("Enter your question: ")
    response = rag_query(query)
    print("\nAnswer:", response)

if __name__ == "__main__":
    main()