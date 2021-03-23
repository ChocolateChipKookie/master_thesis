from colorful.log.listener import Listener
import sys
import datetime
import os

class OutputLogger(Listener):
    def __init__(self, total_iterations, batch_size, frequency=1, output=None, append=False):
        super(OutputLogger, self).__init__(frequency)
        self.output = output
        if output:
            if not append:
                if os.path.exists(output):
                    open(output, 'w')

        self.batch_size = batch_size
        self.begin = datetime.datetime.now()
        self.total_iterations = total_iterations
        self.total_iterations_magnitude = len(str(self.total_iterations))

    def log(self, iter, loss, network):
        now = datetime.datetime.now()
        now_str = now.strftime("%d/%m/%Y %H:%M:%S")
        since_begin = now - self.begin
        total_seconds = since_begin.seconds
        hours = total_seconds // 3600
        minutes = (total_seconds - hours * 3600) // 60
        seconds = total_seconds - hours * 3600 - minutes * 60
        since_begin_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        iter_str = str(iter).rjust(self.total_iterations_magnitude)

        if self.output:
            if os.path.exists(self.output):
                output = open(self.output, 'a')
            else:
                output = open(self.output, 'w')
        else:
            output = sys.stdout

        output.write(
            f"[{now_str}] [{since_begin_str}] [{iter_str}/{self.total_iterations}] {loss:>10.4e}\n"
        )