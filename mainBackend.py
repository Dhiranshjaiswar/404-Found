# main.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import os
import shutil
import random

# -----------------------------
# APP CONFIG
# -----------------------------

app = FastAPI(
    title="404-Found AI Backend",
    description="Fake News + Deepfake Detection API",
    version="1.0.0"
)

# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD NLP MODEL
# -----------------------------

print("Loading AI model...")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("Model Loaded Successfully!")

# -----------------------------
# REQUEST MODELS
# -----------------------------

class NewsRequest(BaseModel):
    text: str

# -----------------------------
# HOME ROUTE
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "404-Found Backend Running Successfully"
    }

# -----------------------------
# FAKE NEWS DETECTION
# -----------------------------

@app.post("/predict-news")
def predict_news(request: NewsRequest):

    text = request.text.strip()

    # validation
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    # AI prediction
    result = classifier(text)[0]

    label = result["label"]
    confidence = round(result["score"] * 100, 2)

    # simple mapping
    if label == "NEGATIVE":
        prediction = "Fake News"
    else:
        prediction = "Real News"

    return {
        "prediction": prediction,
        "confidence": f"{confidence}%"
    }

# -----------------------------
# DEEPFAKE DETECTION
# -----------------------------

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/detect-deepfake")
async def detect_deepfake(file: UploadFile = File(...)):

    # allowed formats
    allowed_extensions = ["jpg", "jpeg", "png", "mp4"]

    file_extension = file.filename.split(".")[-1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format"
        )

    # save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -----------------------------
    # PLACEHOLDER AI LOGIC
    # -----------------------------
    # replace with real deepfake model later

    prediction = random.choice(["Deepfake", "Real"])

    confidence = round(random.uniform(75, 99), 2)

    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": f"{confidence}%"
    }

# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
