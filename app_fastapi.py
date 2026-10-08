import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="RAG Document QA System")

templates_dir = "./templates"
docs_dir = "./sample_docs"

os.makedirs(templates_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

templates = Jinja2Templates(directory=templates_dir)

def load_documents_safely():
    documents = []
    if not os.path.exists(docs_dir):
        return documents
    for file_name in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file_name)
        if file_name.endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        elif file_name.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    return documents

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    files = os.listdir(docs_dir) if os.path.exists(docs_dir) else []
    return templates.TemplateResponse(request=request, name="index.html", context={
        "request": request, "files": files, "answer": None, "chunks": None, "msg": None
    })

@app.post("/upload", response_class=HTMLResponse)
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    uploaded_names = []
    for file in files:
        if file.filename:
            file_path = os.path.join(docs_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            uploaded_names.append(file.filename)
    all_files = os.listdir(docs_dir)
    return templates.TemplateResponse(request=request, name="index.html", context={
        "request": request, "files": all_files, "msg": f"Uploaded: {', '.join(uploaded_names)}", "answer": None, "chunks": None
    })

@app.post("/index-db", response_class=HTMLResponse)
async def index_db(request: Request):
    docs = load_documents_safely()
    all_files = os.listdir(docs_dir)
    if not docs:
        return templates.TemplateResponse(request=request, name="index.html", context={
            "request": request, "files": all_files, "msg": "No documents found!", "answer": None, "chunks": None
        })
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    
    return templates.TemplateResponse(request=request, name="index.html", context={
        "request": request, "files": all_files, "msg": f"Indexed {len(docs)} files into {len(chunks)} chunks in ChromaDB!", "answer": None, "chunks": None
    })

@app.post("/query", response_class=HTMLResponse)
async def query_rag(request: Request, question: str = Form(...), technique: str = Form("role")):
    all_files = os.listdir(docs_dir)
    
    if not api_key:
        return templates.TemplateResponse(request=request, name="index.html", context={
            "request": request, "files": all_files, "msg": "GEMINI_API_KEY is missing in .env file!", "answer": None, "chunks": None
        })

    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
        
        results = vector_db.similarity_search_with_score(question, k=4)
        context_text = "\n\n".join([doc.page_content for doc, _ in results])
        
        formatted_chunks = [
            {"content": doc.page_content, "score": round(float(score), 4)}
            for doc, score in results
        ]

        if technique == "role":
            sys_prompt = "You are an expert Technical Document Analyst. Answer the question accurately using only the provided context."
        elif technique == "fewshot":
            sys_prompt = "Answer accurately based on context."
        else:
            sys_prompt = "Answer the question using the provided context."

        template = f"""{sys_prompt}

Context:
{{context}}

Question: {{question}}

Answer:"""

        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        formatted_prompt = prompt.format(context=context_text, question=question)

        # Updated to gemini-3.8-flash model
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key)
        response = llm.invoke(formatted_prompt)

        clean_answer = response.content
        if isinstance(clean_answer, list) and len(clean_answer) > 0:
            if isinstance(clean_answer[0], dict) and 'text' in clean_answer[0]:
                clean_answer = clean_answer[0]['text']
            else:
                clean_answer = str(clean_answer[0])

        return templates.TemplateResponse(request=request, name="index.html", context={
            "request": request, "files": all_files, "msg": "Query processed successfully!", "question": question, "answer": clean_answer, "chunks": formatted_chunks
        })

    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", context={
            "request": request, "files": all_files, "msg": f"Error: {str(e)}", "question": question, "answer": None, "chunks": None
        })