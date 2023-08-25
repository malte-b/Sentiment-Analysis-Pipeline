from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Model:
  """A model class to lead the model and tokenizer"""

  def __init__(self) -> None:
    if torch.cuda.is_available():
        self.device = torch.device("cuda")  
    elif torch.backends.mps.is_available():
        self.device = torch.device("mps")
    else: 
        self.device = torch.device("cpu")
  
  def load_model():
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    return model

  def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer