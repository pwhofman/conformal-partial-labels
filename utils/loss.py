import sys
import torch
import torch.nn.functional as F

def partial_loss(outputs, targets):
    '''Partial loss adopted from PRODEN https://github.com/Lvcrezia77/PRODEN'''
    outputs = F.softmax(outputs, dim=1)
    # loss can become unstable for real-world data
    eps = 1e-20
    l = targets * torch.log(outputs+eps)
    # l = targets * torch.log(outputs)


    loss = (-torch.sum(l)) / l.size(0)

    if torch.isnan(loss):
        sys.exit("loss became unstable")

    revisedY = targets.clone()
    revisedY[revisedY > 0]  = 1
    revisedY = revisedY * outputs
    revisedY = revisedY / revisedY.sum(dim=1).repeat(revisedY.size(1),1).transpose(0,1)

    new_target = revisedY


    return loss, new_target
