import torch
import torch.nn.functional as F
import numpy as np
from models import MNISTNet

def partialize(y, p, q):
    """Partialize the labels by selecting labels with probability p and
    adding labels according to the binomial distribution with probability q.
    If no label is added, add a random label."""
    new_y = y.clone()
    n, c = y.shape[0], y.shape[1]
    avgC = 0
    for i in range(n):
        if np.random.binomial(1, p, 1):
            row = new_y[i, :] 
            row[np.where(np.random.binomial(1, q, c)==1)] = 1
            while torch.sum(row) == 1:
                row[np.random.randint(0, c)] = 1
            avgC += torch.sum(row)
            new_y[i] = row / torch.sum(row) 
    avgC = avgC / n    
    return new_y

def instance_partialize(loader, dataset):
    """Partialize the labels according to a binomial distribution with distributions
    based on the softmax outputs of the nontarget labels of the MNISTmodel.
    This ensures at least one label is added"""
    if dataset == "cifar10":
        device = 'mps'
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True).to(device)
    else:
        device = 'cpu'
        model = MNISTNet().to(device)
        model.load_state_dict(torch.load("./instance/" + dataset + ".pt"))
        model.eval()
    for inputs, targets, trues, indexes in loader:
        outputs = F.softmax(model(inputs.to(device)), dim=1)
        for i in range(outputs.shape[0]):
            row = targets[i].clone()
            mx = torch.max(outputs[i,row!=1])
            if mx != 0:
                probs = outputs[i].detach().clone()/mx
                probs[probs>1] = 1
                bin = torch.from_numpy(np.random.binomial(1,probs.detach().cpu().numpy())).float()
                # make sure true label is in set
                bin[trues[i]] = 1
                # s = torch.sum(bin)
                # bin = bin / s
                loader.dataset.tensors[1][indexes[i],:] = bin.detach().float().cpu()

def mean_targets(targets, mode):
    """Compute mean label set size"""
    if mode == "all":
        sums = torch.sum(targets != 0, dim=1)
    elif mode == "amb":
        sums = torch.sum(targets != 0, dim=1)
        sums = sums[sums!=1]
    return torch.mean(sums.float())

    
    

