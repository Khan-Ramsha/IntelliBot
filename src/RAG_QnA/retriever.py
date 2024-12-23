from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self, documents, embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model
        self.vector_store = FAISS.from_documents(documents=self.documents, embedding=self.embedding_model)

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": 2})
