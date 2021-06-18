from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import util.util


def create_plot(l_=50., dim = 200):

    def to_xyz(lab):
        var_Y = (lab[0] + 16) / 116
        var_X = lab[1] / 500 + var_Y
        var_Z = var_Y - lab[2] / 200

        if var_Y ** 3 > 0.008856:
            var_Y = var_Y ** 3
        else:
            var_Y = ( var_Y - 16 / 116 ) / 7.787
        if var_X ** 3 > 0.008856:
            var_X = var_X ** 3
        else:
            var_X = ( var_X - 16 / 116 ) / 7.787
        if var_Z ** 3 > 0.008856:
            var_Z = var_Z ** 3
        else:
            var_Z = ( var_Z - 16 / 116 ) / 7.787

        x = var_X * 95.0489
        y = var_Y * 100
        z = var_Z * 108.8840
        return (x, y, z)

    def to_rgb(xyz):
        var_X = xyz[0] / 100
        var_Y = xyz[1] / 100
        var_Z = xyz[2] / 100

        var_R = var_X * 3.2406 + var_Y * -1.5372 + var_Z * -0.4986
        var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415
        var_B = var_X * 0.0557 + var_Y * -0.2040 + var_Z * 1.0570

        if var_R > 0.0031308:
            var_R = 1.055 * ( var_R ** ( 1 / 2.4 ) ) - 0.055
        else:
            var_R = 12.92 * var_R
        if var_G > 0.0031308:
            var_G = 1.055 * ( var_G ** ( 1 / 2.4 ) ) - 0.055
        else:
            var_G = 12.92 * var_G
        if var_B > 0.0031308:
            var_B = 1.055 * ( var_B ** ( 1 / 2.4 ) ) - 0.055
        else:
            var_B = 12.92 * var_B

        sR = var_R * 255
        sG = var_G * 255
        sB = var_B * 255
        return (sR, sG, sB)

    def in_rgb(lab):
        color_xyz = to_xyz(lab)
        color_rgb = to_rgb(color_xyz)
        return all(0 < c < 255 for c in color_rgb)

    a = np.arange(-dim//2, dim//2).reshape((1, dim, 1)).repeat(dim, 0)/1.
    b = np.arange(-dim//2, dim//2).reshape((dim, 1, 1)).repeat(dim, 1)/1.
    l = np.full((dim, dim, 1), l_)

    lab = np.concatenate((l, a, b), 2)
    rgb = color.lab2rgb(lab)

    for x in range(dim):
        for y in range(dim):
            if not in_rgb(lab[x, y]):
                rgb[x, y] = [1, 1, 1]

    plt.imshow(rgb, extent=[-dim//2, dim//2, -dim//2, dim//2])
    plt.gca().invert_yaxis()
    plt.title(f"L* = {l_}")
    plt.show()


def channels_lab(path, l):
    img = cv2.imread(path)
    dir, name = os.path.split(path)
    name = name.split(".")[0]
    lab = color.rgb2lab(img)
    gs = lab[:,:,0] / 100 * 255
    cv2.imwrite(f"{dir}/{name}.png", img)
    cv2.imwrite(f"{dir}/{name}_gs.png", gs)
    lab[:,:,0] = l
    rgb = color.lab2rgb(lab) * 255
    cv2.imwrite(f"{dir}/{name}_color.png", rgb)

def plot_lab(l_, padding=3):
    from colorful import cielab
    bins = cielab.LABBins()
    shape = bins.ab_to_q.shape
    bin_size = 10
    a = np.zeros(shape, dtype=float)
    b = np.zeros(shape, dtype=float)
    l = np.ones_like(a, dtype=float) * l_

    for x in range(shape[0]):
        for y in range(shape[1]):
            index = bins.ab_to_q[y, x]
            if index < 0:
                l[y, x] = 100
                continue
            ab = bins.q_to_ab[index]
            a[y, x] = ab[0]
            b[y, x] = ab[1]

    a += 5
    b += 5
    lab = np.stack((l, a, b), 2)
    plot = np.zeros((shape[0] + padding * 2, shape[1] + padding * 2, 3))
    plot[:,:, 0] = 100
    plot[padding:shape[0]+padding, padding:shape[1]+padding, :] = lab
    plot = np.repeat(np.repeat(plot, bin_size, 0), bin_size, 1)
    plot = color.lab2rgb(plot)
    plt.title(f"L* = {l_}")
    plt.imshow(plot, extent=[-plot.shape[0]//2, plot.shape[0]//2, -plot.shape[1]//2, plot.shape[1]//2])
    plt.show()
    plt.close()

# channels_lab("thesis/graphics/img/lab_channels/san_g.png", 75)
# create_plot(20)
plot_lab(20)
plot_lab(40)
plot_lab(60)
plot_lab(80)
