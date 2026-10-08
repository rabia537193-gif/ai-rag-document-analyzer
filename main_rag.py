import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[ERROR] .env file me GEMINI_API_KEY nahi mili! Check karein ki .env file saved hai.")
    exit()

# ----------------------------------------------------
# 1. Load Documents safely
# ----------------------------------------------------
def load_documents_safely(docs_path="./sample_docs"):
    documents = []
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        return documents

    for file_name in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file_name)
        if file_name.endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                loaded = loader.load()
                if loaded and loaded[0].page_content.strip():
                    documents.extend(loaded)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        elif file_name.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                loaded = loader.load()
                if loaded and any(p.page_content.strip() for p in loaded):
                    documents.extend(loaded)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    return documents

def setup_vector_db():
    print("Documents load ho rahe hain...")
    documents = load_documents_safely("./sample_docs")
    
    if not documents:
        print("\n[ERROR] Documents khali hain! Pehle 'python create_docs.py' chalayein.")
        exit()

    print(f"Total valid documents loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f"Total chunks generated: {len(chunks)}")
    print("Embedding model load ho raha hai (all-MiniLM-L6-v2)...")
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("ChromaDB me embeddings store ho rahi hain...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    return vector_db

# ----------------------------------------------------
# 2. RAG Query Execution with Gemini
# ----------------------------------------------------
def ask_gemini(vector_db, question):
    print("\nRelevant chunks vector search se retrieve ho rahe hain...")
    results = vector_db.similarity_search(question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    template = """You are a Technical Document Assistant. Answer the question strictly using only the provided context below.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    formatted_prompt = prompt.format(context=context_text, question=question)
    
    # Updated to gemini-3.8-flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key
    )
    
    print("Gemini API ko prompt send kiya ja raha hai...")
    response = llm.invoke(formatted_prompt)
    
    print("\n" + "="*50)
    print(f"QUESTION: {question}")
    print("="*50)
    print(f"GEMINI RESPONSE:\n{response.content}")

if __name__ == "__main__":
    print("Vector database setup shuru ho raha hai...")
    db = setup_vector_db()
    
    user_query = "What are the rules regarding remote work and unregistered devices?"
    ask_gemini(db, user_query)