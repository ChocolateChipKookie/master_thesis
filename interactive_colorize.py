from tkinter import *
from tkinter import filedialog

from PIL import ImageTk, Image
from skimage import color

from torchvision.transforms import PILToTensor, ToPILImage
import torch

import json
import sys
import os

from util import util
from ideep.solver import GlobalHints

class MainWindow:
    def __init__(self, root):
        self.root = root
        root.title("Colorizer")
        # Bind functions
        root.bind('<Control-s>', lambda e: self.save(e))
        root.bind('<Control-r>', lambda e: self.choose_reference(e))
        root.bind('<Control-c>', lambda e: self.remove_reference(e))
        root.bind('<Control-n>', lambda e: self.choose_new(e))

        # Load model
        with open("config/colorize_ideep.json", "r") as file:
            config = json.load(file)

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        # Create model and load parameters
        self.network = util.factory(config["colorizer"])
        state_dict = torch.load(config['state_dict'])
        self.network.load_state_dict(state_dict)
        self.network.eval()
        self.network.type(torch.float)
        self.network.to(self.device)
        self.network.train(False)

        # Init members
        self.global_hints = GlobalHints()
        self.p2t = PILToTensor()
        self.t2p = ToPILImage()

        self.img = None
        self.img_gs = None
        self.img_tk = None

        # Create canvas
        self.canvas = Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()
        # If there is a program argument, display image if it exists
        if len(sys.argv) > 1:
            self.display_new(sys.argv[1])

    def colorize(self, l, reference=None):
        if reference is not None:
            reference = reference.to(self.device)
        lab = self.network.forward_colorize(l.to(self.device), reference, False)
        lab = lab.cpu()
        rgb = torch.tensor(color.lab2rgb(lab)).permute(2, 0, 1)
        return rgb

    def display_new(self, path):
        # Check if image exists
        if not os.path.exists(path):
            print("Image does not exist!")
            return

        dir, file = os.path.split(path)
        root.title(f"Colorize: {file}")

        # Open image as pil
        pil_img = Image.open(path)
        # Create normalize to [0, 1]
        img_tensor = self.p2t(pil_img).type(torch.float) / 255

        if img_tensor.shape[0] == 1:
            # If it is already grayscale, fit to L* range [0, 100]
            self.img_gs = img_tensor * 100
        elif img_tensor.shape[0] >= 3:
            if img_tensor.shape[0] == 4:
                # If RGBA, remove alpha channel
                img_tensor = img_tensor[:3]
            # Convert to lab
            lab = color.rgb2lab(img_tensor.permute(1, 2, 0))
            # Get L* channel
            self.img_gs = torch.tensor(lab).permute(2, 0, 1)[:1]
        else:
            raise NotImplementedError

        # Expand batch dimension
        self.img_gs = self.img_gs[None, :, :, :]

        # Create new canvas with wanted size
        self.canvas.destroy()
        self.canvas = Canvas(self.root, width=pil_img.width, height=pil_img.height)
        self.canvas.pack()

        # Display grayscale
        self.img_tk = ImageTk.PhotoImage(self.t2p(self.img_gs[0, 0]/100))
        self.canvas.create_image(0, 0, anchor=NW, image=self.img_tk)
        # Update UI to display grayscale image
        self.root.update()

        # Colorize image
        self.img = self.colorize(self.img_gs)
        # Display image
        self.img_tk = ImageTk.PhotoImage(self.t2p(self.img))
        self.canvas.create_image(0, 0, anchor=NW, image=self.img_tk)
        print("Image loaded")

    def choose_new(self, event):
        filename = filedialog.askopenfilename()
        if filename:
            self.display_new(filename)

    def save(self, event):
        file = filedialog.asksaveasfile(mode='wb', defaultextension=".png")
        if file:
            pil_img = self.t2p(self.img)
            pil_img.save(file)
            print("Image saved")

    def choose_reference(self, event):
        filename = filedialog.askopenfilename()
        if filename:
            img_tensor = self.p2t(Image.open(filename)).type(torch.float)/255

            if img_tensor.shape[0] < 3:
                print("Reference image has to be colored")
            img_tensor = img_tensor[None, :3, :, :]
            gloabal_hints = self.global_hints(img_tensor)

            self.img = self.colorize(self.img_gs, gloabal_hints)
            self.img_tk = ImageTk.PhotoImage(self.t2p(self.img))
            self.canvas.create_image(0, 0, anchor=NW, image=self.img_tk)
            print("Hints added")

    def remove_reference(self, event):
        self.img = self.colorize(self.img_gs)
        self.img_tk = ImageTk.PhotoImage(self.t2p(self.img))
        self.canvas.create_image(0, 0, anchor=NW, image=self.img_tk)
        print("Hints added")


root = Tk()
main = MainWindow(root)
root.mainloop()
