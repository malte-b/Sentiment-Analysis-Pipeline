import evaluate
import torch


def evaluation(model, eval_dataloader, device):
    metrics = []

    metrics.append(evaluate.load('accuracy'))
    metrics.append(evaluate.load('precision'))
    metrics.append(evaluate.load('recall'))
    metrics.append(evaluate.load('f1'))

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        for k in metrics:
            k.add_batch(predictions=predictions, references=batch["labels"])
        
    results = []
    for k in metrics:
        results.append(k.compute())
    
    print(results)
    return results