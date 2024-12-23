from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from src.conversation.prompts import PromptSelector
from langchain_openai import ChatOpenAI  
import cohere
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("KEY")
class ConversationHandler:
    def __init__(self):
        self.prompt = PromptSelector()
        self.history = []
    def generate_response(self,user_input, history, tone):
        prompt_selector = PromptSelector()
        
        prompt_template = prompt_selector.get_prompt(tone)

        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        prompt = prompt_template.format(history=conversation_history, input=user_input)

        co_client = cohere.Client(API_KEY)  

        response = co_client.generate(
            model="command-r-08-2024",  
            prompt=prompt,  
            max_tokens=100,
            temperature=0.7
        )
        
        generated_text = response.generations[0].text.strip()
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": generated_text})
        
        return generated_text
