import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure the Google API key is set as an environment variable.
# (Replace 'YOUR_GOOGLE_API_KEY' with your actual key or use another method to supply the key.)
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyADCKnydIYN5CZiYfuNaswxGB5ZjspeOh8"

from pypdf import PdfReader                              # PyPDF for PDF text extraction:contentReference[oaicite:4]{index=4}
import easyocr                                           # EasyOCR for image (JPG) text extraction:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.faiss import FAISS    # FAISS vector store for embeddings:contentReference[oaicite:7]{index=7}
import gradio as gr

# Global variables to hold the models and chains
embed_model = None
llm = None
chat_llm = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
vector_store = None
qa_chain = None

def initialize_models():
    """Initialize the Google Gemini models with error handling."""
    global embed_model, llm, chat_llm
    try:
        print("Initializing Google Gemini models...")
        embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        llm = GoogleGenerativeAI(model="gemini-2.0-flash")
        chat_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        print("Models initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyPDF."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_image(file_path):
    """Extract English text from an image (JPG) using EasyOCR."""
    reader = easyocr.Reader(['en'], gpu=False)
    # detail=0 returns only the detected text strings:contentReference[oaicite:11]{index=11}
    results = reader.readtext(file_path, detail=0)
    text = " ".join(results)
    return text

def process_file(uploaded_file):
    """Handle file upload: extract text, split into chunks, embed, build FAISS index, and compute summary and clauses."""
    global vector_store, qa_chain, embed_model, llm, chat_llm
    
    # Initialize models if not already done
    if embed_model is None or llm is None or chat_llm is None:
        if not initialize_models():
            return "Error: Could not initialize AI models. Please check your Google API key.", ""
    
    try:
        file_path = uploaded_file.name
        # Determine file type by extension
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            text = extract_text_from_image(file_path)
        else:
            return "Unsupported file format", ""

        if not text.strip():
            return "No text could be extracted from the file.", ""

        # Split text into chunks for embedding and retrieval
        chunks = text_splitter.split_text(text)
        # Convert chunks into Documents for summarization
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Create or recreate the FAISS vector store with Gemini embeddings
        vector_store = FAISS.from_texts(chunks, embedding=embed_model)
        # Create a Retriever for RAG (could use max marginal relevance for diversity)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        # Build the RetrievalQA chain using the chat LLM
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.prompts import PromptTemplate
        from langchain.chains import create_retrieval_chain

        prompt_summary = PromptTemplate.from_template("Summarize this legal document:\n\n{context}")
        summarize_chain = create_stuff_documents_chain(llm, prompt_summary)
        summary = summarize_chain.invoke({"context": documents})

        # ↓ Insert the QA / Clause extraction block here ↓

        clause_query = "List the main clauses in the above document and briefly describe each clause."
        prompt_qa = PromptTemplate.from_template(
            "Here is the context:\n\n{context}\n\nQuestion: {input}"
        )
        combine_chain = create_stuff_documents_chain(llm, prompt_qa)
        qa_chain = create_retrieval_chain(retriever, combine_chain)
        qa_result = qa_chain.invoke({"input": clause_query})
        clauses = qa_result.get("answer") or str(qa_result)

        # → Finally return both results
        return summary, clauses

    except Exception as e:
        return f"Error processing file: {str(e)}", ""
if __name__ == "__main__":
    import sys

    # Path to your local PDF file
    local_pdf_path = "1.pdf"  # <-- Replace with your actual file path

    if not os.path.exists(local_pdf_path):
        print(f"File not found: {local_pdf_path}")
        sys.exit(1)

    class UploadedFileMock:
        def __init__(self, file_path):
            self.name = file_path

    # Mock the uploaded file as Gradio provides it
    uploaded_file = UploadedFileMock(local_pdf_path)

    summary, clauses = process_file(uploaded_file)

    print("\n====== 📄 Document Summary ======\n")
    print(summary)

    print("\n====== 🧾 Main Clauses ======\n")
    print(clauses)
