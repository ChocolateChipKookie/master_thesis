from torch.nn import functional
from torch.utils.data import DataLoader
import torch
from colorful.log.listener import Listener
import datetime
import os


class Validator(Listener):
    def __init__(self, validate_every, solver, output, save_every=None, snapshot_dir=None, append=False):
        super(Validator, self).__init__(validate_every)
        self.solver = solver
        self.save = save_every is not None

        if self.save:
            if not snapshot_dir:
                raise RuntimeError("Destination directory not specified")
            self.save_every = save_every
            self.snapshot_dir = snapshot_dir

            if not os.path.exists(snapshot_dir):
                raise RuntimeError("Destination path does not exist")
            if not os.path.isdir(snapshot_dir):
                raise RuntimeError("Destination path is not a directory")

        self.output = output
        if not append:
            if os.path.exists(output):
                open(output, 'w')
        self.counter = 0

    def log(self, iter, loss, network):
        if iter == 0:
            return
        begin = datetime.datetime.now()
        print("===============================================")
        print(f"Validating...")

        sampler = self.solver.val_sampler
        data_loader = DataLoader(self.solver.val_dataset, batch_size=self.solver.batch_size, shuffle=False, num_workers=self.solver.loaders, sampler=sampler)
        total_validated = 0
        total_loss = 0.
        total_i = 0.
        for batch, _ in data_loader:
            total_i += 1
            total_validated += batch.shape[0]

            with torch.no_grad():
                # Fetch images
                batch = batch.type(self.solver.dtype)
                # Fetch input for the network
                x = batch[:, :1, :, :].to(self.solver.device)
                # Forward pass
                predicted = self.solver.network(x)
                # Fetch output and resize
                actual = functional.interpolate(batch[:, 1:, :, :], size=predicted.shape[2:]).to(self.solver.device)
                # Encode the outputs
                labels = self.solver.encoder(actual)

                # Calculate loss
                total_loss += self.solver.loss(predicted, labels).item()

                # Clean memory
                torch.cuda.empty_cache()

        avg_loss = total_loss / total_i
        end = datetime.datetime.now()
        total_duration = end - begin
        total_seconds = total_duration.total_seconds()

        print(f"Validated {total_validated} samples!")
        print(f"Time spent: {int(total_seconds)} sec")
        print(f"Iteration: {iter}")
        print(f"Average loss: {avg_loss}")

        if os.path.exists(self.output):
            output = open(self.output, 'a')
        else:
            output = open(self.output, 'w')

        output.write(f"{iter:<8} {avg_loss}\n")
        self.counter += 1
        if self.save and self.counter % self.save_every == 0:
            now = datetime.datetime.now()
            now_str = now.strftime("%m_%d")
            name = f"{now_str}-{iter}.pth"
            path = os.path.join(self.snapshot_dir, name)
            torch.save(self.solver.network.state_dict(), path)
            print(f"Saved snapshot {name} to {self.snapshot_dir}")

        print("===============================================")
