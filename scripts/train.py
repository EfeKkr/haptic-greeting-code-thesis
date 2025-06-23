# Training loop for LSTM-based models using pre-extracted video features.
# Tracks loss, accuracy, and macro F1 across epochs, and evaluates on validation set (if provided).
# Saves the best-performing model (based on val F1) with a name indicating the model type and frame ratio.

import torch
from scripts.evaluate import validate
from sklearn.metrics import accuracy_score, f1_score

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, scheduler=None):
    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for inputs, labels, lengths in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Training metrics
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        print(f"Train     - Acc: {train_acc:.2f}, F1: {train_f1:.2f}")

        if val_loader is not None:
            metrics = validate(model, val_loader, device)
            print(f"Validation - Acc: {metrics['accuracy']:.2f}, F1: {metrics['f1']:.2f}")
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save(model.state_dict(), "ViT30.pt") # Saves best model (name depends on video length and pipeline)
                print("!Saved best model!")

        if scheduler:
            scheduler.step()