from langchain.chains import RetrievalQA

class RAGQAEngine:
    def __init__(self, retriever, llm_pipeline):
        self.retriever = retriever
        self.llm_pipeline = llm_pipeline
    
    def generate_answer(self, query):
        qa = RetrievalQA.from_chain_type(llm=self.llm_pipeline, retriever=self.retriever, chain_type="stuff")
        return qa.run(query)
    

