from skimage import color
import matplotlib.pyplot as plt
import importlib


def display_lab(img):
    img = color.lab2rgb(img)
    plt.imshow(img)
    plt.show()
    plt.close()


def import_attr(val):
    pos = val.rfind(".")
    module_str = val[:pos]
    val = val[pos + 1:]
    module = importlib.import_module(module_str)
    return getattr(module, val)


def factory(desc):
    # Dynamically load class and create instance
    # For a class to be loaded, the descriptor has to have 2 fields:
    #   class: class from module to be loaded (eg. module.path.Class)
    #   attr: attributes for creating the class
    Class = import_attr(desc['class'])
    return Class(**desc['args'])