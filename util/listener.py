import sys
from abc import ABCMeta, abstractmethod
import os
import datetime
import torch
import json
import copy
from skimage import color

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class SolverListener:
    """
    Base solver listener class
    The update method is called every frequency iterations
    """
    __metaclass__ = ABCMeta

    def __init__(self, solver, logging_frequency):
        self.solver = solver
        self.frequency = logging_frequency

    @abstractmethod
    def update(self, iter, loss):
        pass

    def __call__(self, iter, loss):
        if iter % self.frequency == 0:
            self.update(iter, loss)


class OutputLogger(SolverListener):
    def __init__(self, solver, frequency=1, output=None):
        """
        Logs basic output for monitoring the solver status
        :param solver: The solver that is being monitored
        :param frequency: The frequency of logging
        :param output: Output file path, if none stdout is used as default
        """
        super(OutputLogger, self).__init__(solver, frequency)

        if output:
            self.output = open(output, 'a' if self.solver.restore else 'w')
        else:
            self.output = sys.stdout

        self.iterations = self.solver.iterations
        self.begin = datetime.datetime.now()
        self.iterations_magnitude = len(str(self.iterations))

    def update(self, iter, loss):
        # Get current time
        now = datetime.datetime.now()
        # Create string
        now_str = now.strftime("%d/%m/%Y %H:%M:%S")
        # Fetch hours, minutes and seconds since begin
        since_begin = now - self.begin
        total_seconds = since_begin.seconds
        hours = total_seconds // 3600
        minutes = (total_seconds - hours * 3600) // 60
        seconds = total_seconds - hours * 3600 - minutes * 60
        # Format
        since_begin_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        # Justify, output and flush
        iter_str = str(iter).rjust(self.iterations_magnitude)
        self.output.write(
            f"[{now_str}] [{since_begin_str}] [{iter_str}/{self.iterations}] {loss:>10.4e}\n"
        )
        self.output.flush()


class LossLogger(SolverListener):
    def __init__(self, solver, frequency=1, format="{0}.\t{1}\n", output = None):
        """
        Logs loss
        :param solver: Solver to be monitored
        :param frequency: Frequence of logging
        :param format: Loss format, {0} for iteration and {1} for loss
        :param output: Output file path, stdout if none is defined
        """
        super(LossLogger, self).__init__(solver, frequency)
        if output:
            if self.solver.restore:
                self.output = open(output, 'a')
            else:
                self.output = open(output, 'w')
        else:
            self.output = sys.stdout

        self.format = format

    def update(self, iter, loss):
        # Write and flush
        self.output.write(self.format.format(iter, loss))
        self.output.flush()


class Backup(SolverListener):
    def __init__(self, solver, frequency, save_dir, config_file='backup_config.json', save_file='backup_model.pth'):
        """
        Backs up the network in regular periods defined by the frequency
        :param solver: Solver to be monitored
        :param frequency: Frequency of backups
        :param save_dir: Base directory for backup files
        :param config_file: Config file name
        :param save_file: Model backup file
        """
        super(Backup, self).__init__(solver, frequency)
        self.save_dir = save_dir
        self.config_file = config_file
        self.save_file = save_file

        if not os.path.isdir(save_dir):
            raise RuntimeError(f"Directory '{save_dir}' does not exist!")

    def update(self, iter, loss):
        # Save state dict of model
        path = os.path.join(self.save_dir, self.save_file)
        torch.save(self.solver.network.state_dict(), path)

        with open(os.path.join(self.save_dir, self.config_file), 'w') as file:
            # Deepcopy the config, so it does not get modified
            config = copy.deepcopy(self.solver.config)
            # Create state
            state = {
                "time": str(datetime.datetime.now()),
                "iter": iter,
                "state_dict": path
            }
            # Update config
            config['state'] = state
            config['restore'] = True
            # Dump JSON
            json.dump(config, file, indent=2)


class Validator(SolverListener):
    def __init__(self, solver, frequency, output=None, save_every=None, snapshot_dir=None):
        super(Validator, self).__init__(solver, frequency)
        self.save = save_every is not None

        if self.save:
            if not snapshot_dir:
                raise RuntimeError("Destination directory not specified")
            self.save_every = save_every
            self.snapshot_dir = snapshot_dir

            if not os.path.exists(snapshot_dir):
                raise RuntimeError("Snapshot directory does not exist")
            if not os.path.isdir(snapshot_dir):
                raise RuntimeError("Defined snapshot directory path is not a directory")

        if output:
            self.output = open(output, 'a' if self.solver.restore else 'w')
        else:
            self.output = sys.stdout
        self.counter = 0

    def write(self, str):
        self.output.write(str)
        self.output.flush()

    def update(self, iter, loss):
        # If it is the first iteration, skip
        if iter == 0:
            return
        # Note the beginning of the validation
        begin = datetime.datetime.now()
        print("===============================================")
        print(f"Validating...")

        # For simpler code
        solver = self.solver

        # Fetch sampler and dataloader
        sampler = solver.val_sampler
        data_loader = DataLoader(solver.val_dataset, batch_size=solver.batch_size, num_workers=solver.loaders, sampler=sampler)

        # Define initial variables
        total_validated = 0
        total_loss = 0.
        total_i = 0.
        # Validate all
        for batch, _ in data_loader:
            total_i += 1
            total_validated += batch.shape[0]

            with torch.no_grad():
                # Calculate loss
                total_loss += self.solver.calculate_loss(batch).item()
                # Clean memory
                torch.cuda.empty_cache()

        # Calculate results
        avg_loss = total_loss / total_i
        end = datetime.datetime.now()
        total_duration = end - begin
        total_seconds = total_duration.total_seconds()

        print(f"Validated {total_validated} samples!")
        print(f"Time spent: {int(total_seconds)} sec")
        print(f"Iteration: {iter}")
        print(f"Average loss: {avg_loss}")

        self.counter += 1
        if self.save and self.counter % self.save_every == 0:
            now = datetime.datetime.now()
            now_str = now.strftime("%m_%d")
            name = f"{now_str}-{iter}.pth"
            path = os.path.join(self.snapshot_dir, name)
            torch.save(self.solver.network.state_dict(), path)
            print(f"Saved snapshot {name} to {self.snapshot_dir}")
        print("===============================================")

        self.write(f"{iter:<8} {avg_loss}\n")


class ColorizeLogger(SolverListener):
    def __init__(self, solver, frequency, directory):
        super().__init__(solver, frequency)
        self.dataset = solver.val_dataset
        self.dataloader = DataLoader(self.dataset, batch_size=1, sampler=self.solver.shuffled_val_sampler)
        self.dataloader_iter = iter(self.dataloader)
        self.directory = directory

    def get_image(self):
        try:
            image, _ = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            image, _ = next(self.dataloader_iter)
        return image

    def update(self, iter, loss):
        with torch.no_grad():
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

            image = self.get_image()
            image = image.to(self.solver.device)[0]

            l = image[:1, :, :]
            predicted = self.solver.network.forward_colorize(l.view(1, *l.shape))

            image_normal = color.lab2rgb(image.cpu().permute(1, 2, 0))

            image_gs = torch.cat(3 * [l.cpu()]).permute(1, 2, 0)
            image_gs /= 100

            image_colorized = color.lab2rgb(predicted.cpu())
            ax1.imshow(image_gs)
            ax2.imshow(image_normal)
            ax3.imshow(image_colorized)

            plt.savefig(os.path.join(self.directory, f"{iter}.png"))
            plt.close()

