from langchain_core.prompts.prompt import PromptTemplate

class PromptSelector:
    def get_prompt(self, tone: str) -> str:
        if tone == "humorous":
            template = """
            You are a witty assistant. Inject humor into your responses while staying relevant.
            Current conversation:
            {history}
            Human: {input}
            Your AI Assistant Response:
            """
        elif tone == "casual":
            template = """
            You are a friendly and casual assistant. Respond to all queries like you're chatting with a friend.
            Current conversation:
            {history}
            Human: {input}
            Your AI Assistant Response:
            """
        elif tone == "formal":
            template = """
            You are a highly professional assistant. Respond to all queries in a formal and polite manner.
            Current conversation:
            {history}
            Human: {input}
            Your AI Assistant Response:
            """
        else:
            raise ValueError("Invalid tone provided")
        return PromptTemplate(input_variables=["history", "input"], template=template)
