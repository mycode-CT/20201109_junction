import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def pairwisedice(output, target):
    s = (10e-20)
    output = output > 0.8
    output = output.type(torch.FloatTensor)

    target = target == 1
    target = target.type(torch.FloatTensor)

    intersect = torch.sum(output * target)

    dice = (2 * intersect) / (torch.sum(output) + torch.sum(target) + s)

    return dice


def soft_dice_loss(output, target):
    s = (10e-20)
    output = output.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)

    intersect = torch.sum(output * target)

    dice = (2 * intersect) / (torch.sum(output) + torch.sum(target) + s)

    return 1 - dice

def weighted_soft_dice_loss(output, target, weights):
    s = (10e-20)
    output = output.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)

    intersect = torch.sum(output * target * weights)
    dice = 2 * intersect / (torch.sum(output*weights) + torch.sum(target*weights) + s)
    print((intersect>0.5).sum(), (output*weights>0.5).sum(), (target*weights).sum(), weights.sum())
    return 1 - dice



def cross_entropy(pred, label, weight=None, reduction='mean',
                  avg_factor=None):
    # element-wise losses
    # print((pred[:,:]*weight).sum(), label.sum(), pred.shape)
    loss = F.cross_entropy(pred, label.type(torch.LongTensor), reduction='none')
    test = torch.nn.functional.softmax(pred, 1)[:, 1] > 0.5
    print((label * weight).sum(), label.sum(), (test * label * weight).sum(), (test * label).sum())
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = (loss * weight).sum()/weight.sum()

    return loss



def focal_loss(inputs, targets, alpha=1, gamma=0, weights=None, logits=True, reduce=True):

    if inputs.ndim == 5:
        inputs = inputs.permute(0,2,3,4,1)
        inputs = inputs.reshape([-1,2])
        targets = targets.flatten()

    if logits:
        BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        #BCE_loss = F.cross_entropy(inputs, targets, reduction = 'none') # chin

    else:
        BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        #BCE_loss = F.cross_entropy(inputs, targets, reduction='none')  # chin
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    test = (torch.nn.functional.softmax(inputs,1)>0.5)[:,1,...]

    # print((targets*weights.flatten()).sum(),weights.sum(),
    #       (test*targets*weights.flatten()).sum(),
    #       (test*weights.flatten()).sum())

    if reduce and weights is not None:
        temp = torch.sum(F_loss*weights.flatten())/weights.sum()
        if weights.sum() == 0:
            return torch.zeros(1) # 0 # chin
        return temp
    else:
        return F_loss


def smooth_l1_loss(pred, target, weights=None, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    #print(target.shape, pred.shape)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    #loss = diff
    if weights is None:
        #print('None')
        return torch.mean(loss)
        #return loss.mean()
    else:
        #print('Else')
        temp = (loss * weights).sum() / weights.sum()
        if weights.sum() == 0:
            return 0
        return temp

def smooth_l1_loss_chin_norm(pred, target, weights=None, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    if weights is None:
        pred = pred - pred.min()  # normalisation
        pred = pred / pred.max()  # normalisation

        target = target - target.min()  # normalisation
        target = target / target.max()  # normalisation

        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        # loss = diff
        return torch.mean(loss)
    else:
        pred = pred * weights
        target = target * weights

        pred = pred - pred.min()  # normalisation
        pred = pred / pred.max()  # normalisation

        target = target - target.min()  # normalisation
        target = target / target.max()  # normalisation

        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        # loss = diff
        temp = (loss * weights).sum() / weights.sum()
        if weights.sum() == 0:
            return 0
        return temp

def smooth_l2_loss(pred, target, weights=None, b1=1.0, b2=1.0):
    assert b1 > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss_fin = diff*diff

    if weights is None:
        return loss_fin.mean()
    else:
        #print(target[weights==1.0]) # chin
        #print(pred[weights==1.0]) # chin
        temp = (loss_fin * weights).sum()/weights.sum()
        if weights.sum() == 0:
            return 0
        return temp