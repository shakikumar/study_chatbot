import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URL = os.getenv("MONGODB_URL") 

# 2. Database Connection (MongoDB) [cite: 15, 16]
client = MongoClient(MONGO_URL)
db = client["study_db"]
collection = db["chat_history"]

# 3. Initialize FastAPI [cite: 4, 10]
app = FastAPI(title="Study Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    question: str

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Study Assistant. Help users with academic questions and remember previous context."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
chain = prompt | llm


def get_history_from_db(user_id: str):
   
    docs = collection.find({"user_id": user_id}).sort("timestamp", 1).limit(10)
    history = []
    for doc in docs:
        if doc["role"] == "user":
            history.append(HumanMessage(content=doc["message"]))
        else:
            history.append(AIMessage(content=doc["message"]))
    return history

@app.get("/")
def home():
    return {"status": "Study Bot API is running"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
   
    history = get_history_from_db(request.user_id)
    

    response = chain.invoke({"history": history, "question": request.question})
    bot_message = response.content

 
    timestamp = datetime.utcnow()
    collection.insert_many([
        {
            "user_id": request.user_id, 
            "role": "user", 
            "message": request.question, 
            "timestamp": timestamp
        },
        {
            "user_id": request.user_id, 
            "role": "assistant", 
            "message": bot_message, 
            "timestamp": timestamp
        }
    ])

    return {
        "user_id": request.user_id,
        "response": bot_message
    }


