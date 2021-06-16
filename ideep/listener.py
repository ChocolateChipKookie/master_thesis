from torch.utils.data import DataLoader
from util.listener import SolverListener
import torch
import matplotlib.pyplot as plt
from skimage import color
import os
import datetime
import sys

class GlobalHintsColorizeLogger(SolverListener):
    def __init__(self, solver, frequency, directory):
        super().__init__(solver, frequency)
        self.dataset = solver.val_dataset
        self.dataloader = DataLoader(self.dataset, batch_size=1, sampler=self.solver.shuffled_val_sampler)
        self.dataloader_iter = iter(self.dataloader)
        self.directory = directory
        self.global_hints = self.solver.global_hints

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
            global_hints = self.global_hints(image).to(self.solver.device)
            image = image.to(self.solver.device)[0]

            l = image[:1, :, :]
            predicted = self.solver.network.forward_colorize(l.view(1, *l.shape), global_hints, False)

            image_normal = color.lab2rgb(image.cpu().permute(1, 2, 0))

            image_gs = torch.cat(3 * [l.cpu()]).permute(1, 2, 0)
            image_gs /= 100

            image_colorized = color.lab2rgb(predicted.cpu())
            ax1.imshow(image_gs)
            ax2.imshow(image_normal)
            ax3.imshow(image_colorized)

            plt.savefig(os.path.join(self.directory, f"{iter}.png"))
            plt.close()


class GlobalHintsValidator(SolverListener):
    def __init__(self, solver, frequency, output=None, save_every=None, snapshot_dir=None):
        super(GlobalHintsValidator, self).__init__(solver, frequency)
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
                total_loss += self.solver.loss(batch, False).item()
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

        if self.save and iter % (self.frequency * self.save_every) == 0:
            now = datetime.datetime.now()
            now_str = now.strftime("%m_%d")
            name = f"{now_str}-{iter}.pth"
            path = os.path.join(self.snapshot_dir, name)
            torch.save(self.solver.network.state_dict(), path)
            print(f"Saved snapshot {name} to {self.snapshot_dir}")
        print("===============================================")

        self.write(f"{iter:<8} {avg_loss}\n")
