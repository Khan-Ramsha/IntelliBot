from langchain_community.document_loaders import PyPDFLoader
class DocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_document(self):
        loader = PyPDFLoader(self.file_path)
        content = loader.load()
        return content
