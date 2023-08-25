from fastapi import FastAPI
from pydantic import BaseModel
from .sentiment_classifier import SentimentClassifier


app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: dict

@app.get("/")
def read_root():
    return "Sentiment Analysis server is running!"

@app.post("/predict")
def predict(request: SentimentRequest):
    classifier = SentimentClassifier()
    sentiment_score = classifier.get_sentiment_label_and_score(text=request.text)
    return sentiment_score#SentimentResponse(sentiment=sentiment_score)