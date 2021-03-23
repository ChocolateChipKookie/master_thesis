

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




