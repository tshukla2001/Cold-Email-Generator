from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_api = os.getenv("GROQ_API_KEY")


class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_key = groq_api
    
    def call(self):
        chat_groq = ChatGroq(
            model_name=self.model_name, 
            temperature=0.2,
            max_retries=2,
            api_key=self.api_key)
        
        return chat_groq
    
if __name__ == "__main__":
    cold_email_llm = Model(
        model_name = "llama-3.3-70b-versatile"
    )
    cold_email_llm = cold_email_llm.call()
    res = cold_email_llm.invoke("First Man to walk on moon was ")
    print(res.content)