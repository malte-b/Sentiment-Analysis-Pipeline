import nltk
import torch
from datasets import load_dataset
from preprocessing import preprocessing
from training import training
from eval import evaluation

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

if torch.cuda.is_available():
    device = torch.device("cuda")  
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")

dataset = load_dataset("imdb")
train_dataloader, eval_dataloader = preprocessing(dataset, device)
model = training(train_dataloader, device)
metrics = evaluation(model, eval_dataloader, device)