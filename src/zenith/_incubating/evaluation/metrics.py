import torch

def accuracy(preds, labels):
    _, preds_max = torch.max(preds, dim=1)
    correct = (preds_max == labels).sum().item()
    return correct / len(labels)