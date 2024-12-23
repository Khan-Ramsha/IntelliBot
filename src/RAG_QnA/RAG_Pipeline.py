from src.RAG_QnA.document_loader import DocumentLoader
from src.RAG_QnA.doc_processing import DocumentProcessor
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.RAG_QnA.retriever import VectorStore
from langchain_community.llms import HuggingFaceHub
from src.RAG_QnA.answer_generator import RAGQAEngine
import os
from dotenv import load_dotenv

load_dotenv()


TOKEN = os.getenv("HUGGINGFACE")

def run_rag_pipeline(file_path, query):
    # Load document
    loader = DocumentLoader(file_path)
    content = loader.load_document()

    # Process the document into chunks
    processor = DocumentProcessor(content)
    chunked_documents = processor.process_documents()

    # Create embeddings for document chunks and set up vector store
    embedding_model = HuggingFaceEmbeddings() 
    vector_store = VectorStore(chunked_documents, embedding_model)
    retriever = vector_store.get_retriever()

    pipeline = HuggingFaceHub(
        repo_id="google/gemma-2-2b-it", 
        model_kwargs={"temperature": 0.7},
        huggingfacehub_api_token=TOKEN
    )

    rag_qa_engine = RAGQAEngine(retriever, pipeline)
    answer = rag_qa_engine.generate_answer(query)
    
    return answer