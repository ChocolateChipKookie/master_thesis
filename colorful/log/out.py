from colorful.log.logger import Logger
import sys
import datetime

class OutputLogger(Logger):
    def __init__(self, total_iterations, batch_size, frequency=1, output = None):
        super(OutputLogger, self).__init__(frequency)
        if output:
            self.output = open(output, 'w')
        else:
            self.output = sys.stdout
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
        self.output.write(
            f"[{now_str}] [{since_begin_str}] [{iter_str}/{self.total_iterations}] {loss:>10.4e}\n"
        )