import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ----------------------------------------------------
# 1. Load Documents
# ----------------------------------------------------
def load_all_documents(folder_path="./sample_docs"):
    """sample_docs folder se saare .txt aur .pdf files load karta hai."""
    documents = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' nahi mila tha, naya bana diya gaya hai.")

    # Text files load karne ke liye
    txt_loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())
    
    # PDF files load karne ke liye
    pdf_loader = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())
    
    return documents

# ----------------------------------------------------
# 2. Text Chunking
# ----------------------------------------------------
def split_documents(docs, chunk_size=300, chunk_overlap=50):
    """Bade documents ko chote chunks me divide karta hai."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

# ----------------------------------------------------
# 3 & 4. Embeddings & ChromaDB Storage
# ----------------------------------------------------
def create_vector_db(chunks, db_directory="./chroma_db"):
    """SentenceTransformers embedding model use karke ChromaDB me store karta hai."""
    print("Embedding model load ho raha hai (all-MiniLM-L6-v2)...")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    print("ChromaDB me embeddings store ho rahe hain...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_directory
    )
    return vector_db

# ----------------------------------------------------
# 5, 6 & 7. Query Conversion & Top 3 Chunk Retrieval
# ----------------------------------------------------
def search_similar_chunks(vector_db, query, top_k=3):
    """User ke question ka embedding banakar top 3 most relevant chunks dhoondta hai."""
    print(f"\n==================================================")
    print(f"SEARCH QUERY: '{query}'")
    print(f"==================================================")
    
    results = vector_db.similarity_search_with_score(query, k=top_k)
    
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\n[Rank {rank}] | Distance Score: {score:.4f}")
        print(f"Source Document: {doc.metadata.get('source', 'Unknown')}")
        print(f"Chunk Text:\n{doc.page_content.strip()}")
        print("-" * 50)

# ----------------------------------------------------
# Execution Flow
# ----------------------------------------------------
if __name__ == "__main__":
    DOCS_PATH = "./sample_docs"
    
    # Step 1: Documents load karein
    raw_docs = load_all_documents(DOCS_PATH)
    print(f"Total documents loaded: {len(raw_docs)}")
    
    if len(raw_docs) == 0:
        print("\n[ERROR] 'sample_docs' folder me koi file nahi mili ya files khali hain!")
        print("Khabardar: 'sample_docs' folder me kam se kam 1-2 text files (.txt) bana kar unme kuch lines likh kar save karein.")
        exit()

    # Step 2: Chunks banayein
    doc_chunks = split_documents(raw_docs)
    print(f"Total chunks created: {len(doc_chunks)}")

    if len(doc_chunks) == 0:
        print("\n[ERROR] Documents se koi chunk nahi ban saka. Make sure text files me data mojood ho.")
        exit()

    # Step 3 & 4: Embeddings & Vector DB
    db = create_vector_db(doc_chunks)
    print("Vector database successfully build ho gaya!")
    
    # Step 5, 6 & 7: Test Query Run Karein
    user_query = "What are the rules or policies mentioned in the documents?"
    search_similar_chunks(db, query=user_query, top_k=3)