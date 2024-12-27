
from fastapi import FastAPI, Request, HTTPException
from src.conversation.handler import ConversationHandler
from src.utils.exception import AppException
from src.utils.logger import Logger
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.RAG_QnA.RAG_Pipeline import run_rag_pipeline
from src.RAG_QnA.extract_helpful_ans import extract_answer
from fastapi import File, UploadFile
from src.RAG_QnA.document_loader import DocumentLoader
from src.doc_summary.summarizer import MeetingAssistant
from src.doc_summary.data_preprocessing import preprocessing,clean_numbers
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

app = FastAPI(title="ConverseQueryAI")
logger = Logger.get_logger()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates"), name="static")

conversation_handler = ConversationHandler()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

conversation_history_store = {}

@app.post("/chat")
async def conversation_endpoint(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id")  
        question = data.get("question", "").strip()
        tone = data.get("tone", "formal")


        history = conversation_history_store.get(user_id, [])

        response_text = conversation_handler.generate_response(question, history, tone)

        conversation_history_store[user_id] = history

        result = {
            "response": response_text,
            "history": [{"question": history[i]["content"], "answer": history[i+1]["content"]} for i in range(0, len(history), 2)]
        }
        logger.info(f"Tone User Selected : {tone}")
        logger.info(f"Input from User: {question}")
        logger.info(f"Response according to the selected tone: {response_text}")

        return result
    except Exception as e:
        logger.error(f"Error during conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/doc_processing", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("document.html", {"request": request})

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
        file_location = f"artifacts/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        return {"message": "File uploaded successfully", "file_path": file_location}
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload file")

@app.post("/QnA")
async def qna(request: Request):
    try:
        data = await request.json()
        file_path = data.get("file_path")
        query = data.get("query", "").strip()

        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Invalid file path.")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
        raw_answer = run_rag_pipeline(file_path, query)
        answer = extract_answer(raw_answer)
        logger.info(f"Answer of the asked question: {answer}")
        return {"response": answer}
    except Exception as e:
        logger.error(f"Error during QnA: {str(e)}")
    
        if '500' in str(e):  
            raise HTTPException(status_code=500, detail="The model is busy, please try again later.")
        raise HTTPException(status_code=500, detail="Failed to process the document, Try Again")
        
@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    file_path = data.get("file_path")
    loader = DocumentLoader(file_path)
    content = loader.load_document()
    content = preprocessing(content)
    assistant = MeetingAssistant(HUGGINGFACE_TOKEN, "mistralai/Mistral-7B-Instruct-v0.3")
    assistant.login_to_huggingface()
    logger.info("Logging into HuggingFace")
    assistant.load_model()
    logger.info("Loading model...")
    summary = assistant.generate_summary(content)
    logger.info("Summary Generated..")
    summary = clean_numbers(summary)
    return {"summary":summary}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)