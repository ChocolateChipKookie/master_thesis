import torch

class ClosestDecode:
    def __init__(self, mapping):
        self.mapping = mapping

    def to(self, device):
        self.mapping = self.mapping.to(device)

    def __call__(self, q_img):
        q_img = q_img.argmax(dim=1)
        decoded = self.mapping[q_img]
        # The input is the quantized bin image
        return decoded


class AnnealedMeanDecode:
    def __init__(self, mapping, T=0.38):
        self.T = T
        self.mapping = mapping

        # In case T == 0 the decoding degrades to finding the class with the highest probability
        # Cannot use the default method as it divides by the T factor
        if self.T == 0:
            self.decode_func = self._one_hot_decode
        else:
            self.decode_func = self._annealed_mean_decode

    def to(self, device):
        self.mapping = self.mapping.to(device)

    def __call__(self, q_img):
        return self.decode_func(q_img)

    def _one_hot_decode(self, q_img):
        q_img = q_img.argmax(dim=1)
        decoded = self.mapping[q_img]
        # The input is the quantized bin image
        return decoded

    def _annealed_mean_decode(self, q_img):
        """
            sum(exp(log(x)/T))
        """
        exp_log = torch.pow(q_img, 1/self.T)
        sum_exp_log = torch.sum(exp_log, dim=1)
        prob = exp_log / sum_exp_log

        a = torch.tensordot(prob, self.mapping[:, 0], ([1], [0]))
        b = torch.tensordot(prob, self.mapping[:, 1], ([1], [0]))

        return torch.stack((a, b), 3)

