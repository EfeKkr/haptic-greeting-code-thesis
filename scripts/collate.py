# This script specifies the custom `pad_collate` function used in `DataLoader`batching. 
# Video feature sequences have different lengths and this function pads them to the same length as 
# the longest one batch, so that batch processing with pyTorch is possible. It also gives the original 
# length of the sequences back. Served to both CNN+LSTM and ViT+LSTM training and evaluation pipeline.

from torch.nn.utils.rnn import pad_sequence
import torch

def pad_collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True)  # Pad sequences to have the same length across the batch: [B, T, D]
    labels = torch.tensor(labels)
    return padded_sequences, labels, lengths
