import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import openai
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Please set GROQ_API_KEY environment variable")

# Configure OpenAI client for Groq
openai.api_key = groq_api_key
openai.base_url = "https://api.groq.com/v1"

def get_embeddings(texts):
    """Get embeddings for a list of texts using Groq API."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        raise

def process_uploaded_pdf(uploaded_file):
    """Save uploaded PDF and return its path."""
    if not os.path.exists("pdfs"):
        os.makedirs("pdfs")
    file_path = f"pdfs/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def refresh_vectorstore():
    """Process PDFs and rebuild the FAISS vector store."""
    print("🔄 Rebuilding FAISS Vectorstore...")

    # Load all PDFs
    loaders = [PyPDFLoader(f"pdfs/{f}") for f in os.listdir("pdfs") if f.endswith(".pdf")]
    if not loaders:
        print("⚠️ No PDFs found in 'pdfs/' directory.")
        return None

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    print(f"📄 Loaded {len(documents)} pages from PDFs.")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📌 Total document chunks created: {len(chunks)}")
    
    # Get embeddings using Groq
    texts = [doc.page_content for doc in chunks]
    embeddings_list = get_embeddings(texts)
    
    # Create FAISS index
    dimension = len(embeddings_list[0])
    import faiss
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings_list))
    
    # Create FAISS database
    class GroqEmbeddings:
        def embed_documents(self, texts):
            return get_embeddings(texts)
        
        def embed_query(self, text):
            return get_embeddings([text])[0]

    faiss_db_local = FAISS(
        embeddings=GroqEmbeddings(),
        index=index,
        docstore=chunks
    )

    # Save FAISS index
    if not os.path.exists("vectorstore"):
        os.makedirs("vectorstore")
    if not os.path.exists("vectorstore/db_faiss"):
        os.makedirs("vectorstore/db_faiss")
    faiss_db_local.save_local("vectorstore/db_faiss")
    print("✅ FAISS vectorstore successfully rebuilt.")
    
    return faiss_db_local

def cleanup_all_resources():
    """Clean up all PDFs and vector database resources"""
    try:
        # Clean up PDFs directory
        if os.path.exists("pdfs"):
            for file in os.listdir("pdfs"):
                file_path = os.path.join("pdfs", file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            try:
                os.rmdir("pdfs")
            except Exception as e:
                print(f"Error removing pdfs directory: {e}")

        # Clean up vector database
        if os.path.exists("vectorstore/db_faiss"):
            for file in os.listdir("vectorstore/db_faiss"):
                file_path = os.path.join("vectorstore/db_faiss", file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            try:
                os.rmdir("vectorstore/db_faiss")
            except Exception as e:
                print(f"Error removing db_faiss directory: {e}")
            try:
                os.rmdir("vectorstore")
            except Exception as e:
                print(f"Error removing vectorstore directory: {e}")

        print("✨ All resources cleaned up successfully")
        return True
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return False

def cleanup_pdf(pdf_path):
    """Clean up PDF and its associated resources"""
    try:
        # Remove PDF file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        # Remove associated text file if exists
        text_path = pdf_path.replace('.pdf', '.txt')
        if os.path.exists(text_path):
            os.remove(text_path)
        
        # Check if pdfs directory is empty
        if os.path.exists("pdfs") and not os.listdir("pdfs"):
            try:
                os.rmdir("pdfs")
            except Exception as e:
                print(f"Error removing empty pdfs directory: {e}")
        
        # Rebuild vectorstore without the removed PDF
        refresh_vectorstore()
        
        return True
    except Exception as e:
        print(f"Error cleaning up PDF: {e}")
        return False

# Load or refresh the FAISS vector store
if os.path.exists("vectorstore/db_faiss"):
    print("📥 Loading existing FAISS index...")
    try:
        class GroqEmbeddings:
            def embed_documents(self, texts):
                return get_embeddings(texts)
            
            def embed_query(self, text):
                return get_embeddings([text])[0]

        faiss_db = FAISS.load_local(
            "vectorstore/db_faiss",
            GroqEmbeddings(),
            allow_dangerous_deserialization=True
        )
        print("✅ Loaded existing FAISS index...")
        if getattr(faiss_db.index, "ntotal", 0) == 0:
            raise ValueError("⚠️ FAISS index is empty. Rebuilding...")
    except Exception as e:
        print(f"❌ FAISS Load Failed: {str(e)}")
        print("🔄 Rebuilding FAISS from scratch...")
        faiss_db = refresh_vectorstore()
else:
    print("🚀 No FAISS index found. Creating a new one...")
    faiss_db = refresh_vectorstore()
