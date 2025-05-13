import streamlit as st
import os
import hashlib
import json
from dotenv import load_dotenv
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env


PDF_FILES = [
    os.path.join("data", "Women’s Rights in India complete_compressed.pdf"),
    os.path.join("data", "Majlis_Legal-rights-of-women.pdf"),
]

FAISS_INDEX_PATH = "faiss_index_multiple_pdfs"
INDEX_METADATA_PATH = "index_metadata.json"

def compute_file_hash(file_path):
    """Compute SHA256 hash of a file to detect changes."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error computing file hash for {file_path}: {str(e)}")
        return None

def load_existing_index():
    """Check if FAISS index and metadata exist and are up-to-date."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(INDEX_METADATA_PATH):
        return None

    try:
        with open(INDEX_METADATA_PATH, "r") as f:
            stored_metadata = json.load(f)
        
        current_hashes = {pdf: compute_file_hash(pdf) for pdf in PDF_FILES}

        if stored_metadata == current_hashes:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
            logger.info("FAISS index is up-to-date. Loading existing index.")
            return db
        else:
            logger.info("PDF files have changed. Rebuilding FAISS index.")

    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")

    return None

def create_new_index():
    """Create FAISS index for given PDFs and save metadata."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    all_documents = []
    for pdf_path in PDF_FILES:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(docs)
            all_documents.extend(documents)

        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            st.error(f"Error loading PDF {pdf_path}")

    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local(FAISS_INDEX_PATH)

    file_hashes = {pdf: compute_file_hash(pdf) for pdf in PDF_FILES}
    with open(INDEX_METADATA_PATH, "w") as f:
        json.dump(file_hashes, f)

    logger.info("FAISS index created and saved.")
    return db

def get_faiss_index():
    """Load existing FAISS index or create a new one if needed."""
    db = load_existing_index()
    if db is None:
        db = create_new_index()
    return db

def main():
    st.set_page_config(
        page_title="Women's Rights Legal Query System",
        page_icon="⚖️",
        layout="wide"
    )

    st.title("JusticeMitra")

    if "previous_searches" not in st.session_state:
        st.session_state.previous_searches = []

    st.sidebar.header("Previous Searches")
    if st.session_state.previous_searches:
        for idx, query in enumerate(st.session_state.previous_searches):
            st.sidebar.write(f"{idx + 1}. {query}")
    else:
        st.sidebar.write("No previous searches.")

    db = get_faiss_index()
    retriever = db.as_retriever()

    # Fetch the API key from Streamlit's Secrets management system
    groq_api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    prompt_template = PromptTemplate(
        template=""" 
        Answer the question based on the provided context only. 
        Provide a detailed and accurate response. 
        <context> 
        {context} 
        </context> 
        Question: {input} 
        """,
        input_variables=["context", "input"]
    )

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    query = st.text_input("Please describe your situation or ask your question:")

    if query:
        st.session_state.previous_searches.append(query)

        with st.spinner("Searching for relevant information..."):
            response = retrieval_chain.invoke({"input": query})

            if response:
                st.write("**Important Information:**")
                st.write(response["answer"])
                st.write("---")
                st.write("**Emergency Contacts and Helplines:**")
                st.markdown("""
                    - **Police Emergency**: 100
                    - **Women's Helpline**: 1091
                    - **National Commission for Women**: 011-26944880, 26944883
                    - **Domestic Violence Helpline**: 181
                    - **Legal Services Authority**: 1516
                    *Please save these numbers for future reference.*
                """)

if __name__ == "__main__":
    main()
