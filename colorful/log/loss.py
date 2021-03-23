from colorful.log.listener import Listener
import sys
import os

class LossLogger(Listener):
    def __init__(self, frequency=1, format="{0}.\t{1}\n", output = None):
        super(LossLogger, self).__init__(frequency)
        self.output = output
        if output:
            if os.path.exists(output):
                open(output, 'w')

        self.format = format

    def log(self, iter, loss, network):
        if self.output:
            if os.path.exists(self.output):
                output = open(self.output, 'a')
            else:
                output = open(self.output, 'w')
        else:
            output = sys.stdout
        output.write(self.format.format(iter, loss))