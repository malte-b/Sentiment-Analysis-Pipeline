from .model import Model
from scipy.special import softmax
import numpy as np
import torch

class SentimentClassifier:
  def __init__(self):
    self.model = Model.load_model()
    self.tokenizer = Model.load_tokenizer()
    self.model.load_state_dict(
        torch.load('/code/models/sentiment_bert.pt', map_location=self.model.device)
    )
    self.model = self.model.eval()

  def get_sentiment_label_and_score(self, text: str):
    result = {}
    labels = ["negative", "positive"]
    encoded_input = self.tokenizer(text, 
                                   return_tensors='pt', 
                                   padding=True,
                                   truncation=True)
    output = self.model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    result["label"] = str(labels[ranking[0]])
    result["score"] = np.round(float(scores[ranking[0]]), 4)
    return result