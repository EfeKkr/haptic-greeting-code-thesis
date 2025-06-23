# Evaluation utility for computing validation metrics.
# Calculates accuracy, precision, recall, and F1 score 
# on the given dataset using the provided model.

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, lengths in data_loader:
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            outputs = model(inputs, lengths)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "f1": f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }
