
class Logger:
    def __init__(self, logging_frequency):
        self.frequency = logging_frequency

    def log(self, iter, loss, network):
        raise NotImplementedError("Function not implemented!")

    def __call__(self, iter, loss, network):
        if iter % self.frequency == 0:
            self.log(iter, loss, network)
