import util
import os
import torchvision


#dataset = util.ImageDataset("/hdd/imagenet/train")
#print(len(dataset))

dataset = torchvision.datasets.ImageFolder("/hdd/imagenet/train")

print(len(dataset))
