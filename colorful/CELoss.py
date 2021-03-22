import torch
from torch.nn.functional import log_softmax


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        if weights is not None:
            self.weights_sum = self.weights.sum()

    def forward(self, predicted, target):
        n, _, w, h = predicted.shape
        m = w * h * n
        softmax = log_softmax(predicted, dim=1)
        norm = target.clone()
        norm[norm != 0] = torch.log(norm[norm != 0])
        loss = -torch.sum((softmax - norm) * target) / predicted.shape[0]
        if self.weights is not None:
            loss = loss * self.weights
            loss = loss.sum() / self.weights_sum
        return loss / m
