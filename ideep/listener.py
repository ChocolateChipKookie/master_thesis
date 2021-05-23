from torch.utils.data import DataLoader
from util.listener import SolverListener
import torch
import matplotlib.pyplot as plt
from skimage import color
import os

class GlobalHintsColorizeLogger(SolverListener):
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
            global_hints = self.solver.get_global(image).to(self.solver.device)
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