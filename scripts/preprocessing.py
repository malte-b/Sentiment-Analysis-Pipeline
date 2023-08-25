from datasets import Dataset
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

stopwords = set(stopwords.words('english'))

def preprocessing(dataset, device):

    #Switch to pandas for pre-processing
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()

    #Removing the html strips
    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    #Removing the square brackets
    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)

    #Removing the noisy text
    def denoise_text(text):
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        return text

    #Removing special characters
    def remove_special_characters(text, remove_digits=True):
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern,'',text)
        return text

    #Reduce all words to their stems
    def stem(text):
        stemmer = PorterStemmer()
        text= ' '.join([stemmer.stem(w) for w in text.split()])
        return text

    #Removing English stopwords
    def remove_stopwords(text):
        tokens = word_tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_words = [w for w in tokens if w not in stopwords]
        filtered_text = ' '.join(filtered_words)    
        return filtered_text

    #Combine all pre-processing steps into one function
    def preprocessing_combined(df):
        for func in [denoise_text, remove_special_characters, stem, remove_stopwords]:
            df = df.apply(func)
        return df


    df_train['text'] = preprocessing_combined(df_train['text'])
    df_test['text'] = preprocessing_combined(df_test['text'])

    dataset['train'] = Dataset.from_pandas(df_train, split='train')
    dataset['test'] = Dataset.from_pandas(df_test, split='test')

    #Switch back for tokenization
    dataset.reset_format()


    #Start Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt").to(device)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    #For demo purposes I select a subset of the train and test datasets
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #Some renaming
    small_train_dataset = small_train_dataset.remove_columns(["text"])
    small_train_dataset = small_train_dataset.rename_column("label", "labels")
    small_train_dataset.set_format("torch")
    small_eval_dataset = small_eval_dataset.remove_columns(["text"])
    small_eval_dataset = small_eval_dataset.rename_column("label", "labels")
    small_eval_dataset.set_format("torch")

    #Loading the data into PyTorch Dataloaders
    train_dataloader = DataLoader(
        small_train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        small_eval_dataset, batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader

