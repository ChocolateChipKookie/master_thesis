import torch
from torch.nn.functional import log_softmax


class MultinomialCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        if weights is not None:
            self.weights_sum = self.weights.sum()

    def forward(self, logits, target):
        # Log-softmax
        ls = log_softmax(logits, dim=1)
        # Cross-entropy
        ce = torch.sum(ls * target, dim=1)
        # Apply weights
        if self.weights is not None:
            am = torch.argmax(target, dim=1)
            weights = self.weights[am]
            ce *= weights

        loss = -torch.mean(ce)
        return loss
