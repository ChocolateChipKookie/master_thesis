from tkinter import *
from tkinter import filedialog

from PIL import ImageTk, Image
from skimage import color

from torchvision.transforms import PILToTensor, ToPILImage
import torch

import json
from util import util
from ideep.solver import GlobalHints

class MainWindow:
    def __init__(self, root):
        self.root = root
        root.bind('<Control-s>', lambda e: self.save(e))
        root.bind('<Control-r>', lambda e: self.choose_reference(e))
        root.bind('<Control-n>', lambda e: self.choose_new(e))

        with open("config/colorize_ideep.json", "r") as file:
            config = json.load(file)
        device = torch.device("cpu")

        self.canvas = Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()

        # Create model and load parameters
        self.network = util.factory(config["colorizer"])
        state_dict = torch.load(config['state_dict'])
        self.network.load_state_dict(state_dict)
        self.network.eval()
        self.network.to(device)
        self.network.type(torch.float)
        self.network.train(False)

        self.global_hints = GlobalHints()

        self.p2t = PILToTensor()
        self.t2p = ToPILImage()

        self.img = None
        self.img_gs = None
        self.img_tk = None

    def colorize(self, l, reference=None):
        lab = self.network.forward_colorize(l, reference, False)
        rgb = torch.tensor(color.lab2rgb(lab)).permute(2, 0, 1)
        return rgb

    def choose_new(self, event):
        filename = filedialog.askopenfilename()
        if filename:
            pil_img = Image.open(filename)
            img_tensor = self.p2t(pil_img).type(torch.float)/255

            if img_tensor.shape[0] == 1:
                self.img_gs = img_tensor
            elif img_tensor.shape[0] == 3:
                lab = color.rgb2lab(img_tensor.permute(1, 2, 0))
                self.img_gs = torch.tensor(lab).permute(2, 0, 1)[:1]
            elif img_tensor.shape[0] == 4:
                lab = color.rgb2lab(img_tensor[:3].permute(1, 2, 0))
                self.img_gs = torch.tensor(lab).permute(2, 0, 1)[:1]
            else:
                raise NotImplementedError

            self.img_gs = self.img_gs[None, :, :, :]
            self.img = self.colorize(self.img_gs)

            self.canvas.destroy()
            self.canvas = Canvas(self.root, width=pil_img.width, height=pil_img.height)
            self.canvas.pack()

            self.img_tk = ImageTk.PhotoImage(self.t2p(self.img))
            self.canvas.create_image(0, 0, anchor=NW, image=self.img_tk)
            print("Image loaded")

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


root = Tk()
main = MainWindow(root)
root.mainloop()