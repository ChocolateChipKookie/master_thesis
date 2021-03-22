from colorful.log.logger import Logger
import sys

class LossLogger(Logger):
    def __init__(self, frequency=1, format="{0}.\t{1}\n", output = None):
        super(LossLogger, self).__init__(frequency)
        if output:
            self.output = open(output, 'w')
        else:
            self.output = sys.stdout
        self.format = format

    def log(self, iter, loss, network):
        self.output.write(self.format.format(iter, loss))