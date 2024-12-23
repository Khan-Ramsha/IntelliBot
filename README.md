## IntelliBot

A multifunctional AI-powered smart assistant designed to offer personalized conversational experiences with memory and tone customization. This system allows users to interact with the assistant in different tones, upload documents for summarization, ask questions based on the document's content using a Retrieval-Augmented Generation (RAG) pipeline. Powered by cutting-edge Large Language Models (LLMs) like Cohere Command, Gemma, and Mistral 7B.


Features:

- Tone Customization & Memory Functionality: Tailor your conversations with personalized tone options, including formal, casual, and humorous tones, which the assistant maintains throughout the chat. Retains memory of past interactions, allowing the assistant to evolve and provide more personalized and context-aware responses over time.

- Document QnA: User can upload PDFs and ask questions based on a corpus of documents. IntelliBot will quickly analyze and retrieve relevant answers using its RAG-based pipeline and utilizes Gemma for generating answers based on the relevant document chunks. 

- Summarize Documents: Easily extract key points from large documents with the help of Mistral 7B for concise and accurate summaries.

Installation

Clone the repository:

```
https://github.com/Khan-Ramsha/IntelliBot.git
cd intellibot

```

Install the dependencies:

```
pip install -r requirements.txt
```

Create a .env file and add your API keys (for models like Cohere, Hugging Face, etc.):

```
COHERE_API_KEY=your_cohere_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```
Run the app:

```
python app.py
```
The app will be available at http://127.0.0.1:8000.

Engage in conversation, ask questions from PDF, or summarize your document.
