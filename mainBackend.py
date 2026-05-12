from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fake_news import analyze

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class NewsRequest(BaseModel):
    url: str


@app.get("/")
def home():
    return {
        "message": "Fake News Detector API Running"
    }


@app.post("/analyze")
def check_news(news: NewsRequest):
    result = analyze(news.url)
    return result