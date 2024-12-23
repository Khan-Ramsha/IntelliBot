from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, content):
        self.content = content

    def process_documents(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_documents = splitter.split_documents(self.content)
        return chunked_documents
