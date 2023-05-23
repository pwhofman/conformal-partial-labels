import torch

def accuracy(model, loader):
    """Compute accuracy of model on loader"""
    device = next(model.parameters()).device
    correct = 0
    total = 0
    for inputs, _, targets, _ in loader:
        outputs = model(inputs.to(device).float())
        correct += torch.sum(torch.argmax(outputs, dim=1) == targets.to(device))
        total += inputs.shape[0]
    return correct / total * 100