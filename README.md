# Sentiment Analysis Pipeline

Deploy a pre-trained BERT model for Sentiment Analysis as a REST API using FastAPI with Docker

## Demo

The model is trained to classify sentiment (negative or positive) on a custom dataset from movie reviews on IMDB. Here's a sample request to the API:

```bash
http POST http://0.0.0.0:7860/predict text="Best movie ever
```

The response you'll get looks something like this:

```js
{
    "label": "positive",
    "score": 0.7835
}
```


## Installation

Clone this repo:

```sh
git clone git@github.com:malte-b/Sentiment-Analysis-Pipeline.git
cd Sentiment-Analysis-Pipeline
```

Build Docker container:

``sh
docker build -f Dockerfile -t sentiment .
``

Run Docker container:

``sh
docker run  -it -p 7860:7860 sentiment 
``


## Training and evaluating the model

For training the BERT model on device, one has first to install the dependencies on their local machine:

``sh
pip install -r requirements.txt 
``

Then they only have to run the pipeline.py script and the program will to the rest.

``sh
cd scripts
python3 pipeline.py
``


## Why did I choose this particular dataset and model

Sentiment analysis is one of the most accessible NLP problems which reduces training time while still being a very relevant technique for many practical applications. Similarly, I used the DistilBERT as a smaller, distilled version of the famous encoder-only BERT model which can achieve comparable performance while needing significantly fewer resources. BERT is generally well suited for text classification tasks and can easily be fine-tuned to mode-specific use cases. 

## Challenges I faced and how I overcame them

Since I do data preprocessing in pandas, it was a bit tricky to convert the dataframes even with tokenized tensors to a format that would work for my model. Ultimately, I decided to use the transformers library as an abstraction with an easy convert function to pandas and back.
Regarding the API server, I also decided to go with a simpler variant in the final version since it was not super straightforward to wrap the transformers model in a more abstract class.

## How can my application be updated or maintained in the future?

Regarding libraries, they can be updated in the requirement.txt file. Different foundation models can be used as a drop-in replacement but the model classes as well as the initalization strings need to be adjusted in the code accordingly. Also, the imdb dataset I used for finetuning can be changed with another in the pipeline.py function. Lastly, the size of the dataset and possible augmentations can be tuned in the preprocessing.py script within the code. 

## License

MIT
