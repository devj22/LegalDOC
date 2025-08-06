import os
import asyncio
import gradio as gr
from pypdf import PdfReader
import easyocr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

# ✅ Set API key FIRST
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "Enter your api key"

# ✅ Fix for OpenMP libomp error on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ Ensure event loop for async libs like Gradio + Gemini
try:
    asyncio.get_event_loop()

except RuntimeError:
    asyncio.get_event_loop(asyncio.new_event_loop())

# Globals
embed_model = None
llm = None
chat_llm = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
vector_store = None
qa_chain = None

def initialize_models():
    """Ensure event loop exists in the current thread and initialize models."""
    global embed_model, llm, chat_llm

    try:
        import threading

        # Ensure event loop in current thread (especially for AnyIO threads)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

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
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_image(file_path):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(file_path, detail=0)
    return " ".join(results)

from langchain_core.documents import Document

def process_file(uploaded_file):
    """Handle file upload: extract text, split into chunks, embed, build FAISS index, and compute summary and clauses."""
    global vector_store, qa_chain, embed_model, llm, chat_llm
    
    # Initialize models if not already done
    if embed_model is None or llm is None or chat_llm is None:
        if not initialize_models():
            return "Error: Could not initialize AI models. Please check your Google API key.", ""
    
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

def answer_question(user_input, chat_history):
    global qa_chain
    if not qa_chain:
        return "", chat_history

    try:
        result = qa_chain.invoke({"input": user_input})
        answer = result.get("answer") or result.get("output") or str(result)

        # Append in Gradio-friendly format (list of dicts with role/content)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})

        return "", chat_history
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return "", chat_history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Legal Document Analyzer (Gemini)")
    gr.Markdown("Upload a **PDF or Image** of a legal document to get a **summary**, **clauses**, and ask **questions**.")

    status = gr.Textbox(value="Ready. Upload a document.", interactive=False, label="Status")

    with gr.Row():
        file_input = gr.File(label="Upload Document (PDF or JPG)", type="filepath")
        summary_box = gr.Textbox(label="📑 Document Summary", lines=6)
        clauses_box = gr.Textbox(label="📌 Extracted Clauses", lines=6)

        file_input.change(fn=process_file, inputs=[file_input], outputs=[summary_box, clauses_box])

    gr.Markdown("## 💬 Ask Questions About the Document")
    chatbot = gr.Chatbot(label="Gemini Chat", type="messages")
    msg = gr.Textbox(placeholder="Ask a question...")
    msg.submit(fn=answer_question, inputs=[msg, chatbot], outputs=[msg, chatbot])

# Launch the app
print("Launching Gradio app...")
demo.launch(share=False, server_name="0.0.0.0", server_port=7897)
