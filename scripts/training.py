from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
import torch

def training(train_dataloader, device, save_model=True):

    #Using a pre-trained BERT model
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device) 

    #Initializing optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    #Setting up the leraning rate scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    #Training the model
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    #optionally save the model
    if save_model:
        torch.save(model.state_dict(), '../models/sentiment_bert.pt')

    print('Training Complete!')
    return model