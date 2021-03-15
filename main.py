import util
import os
import torchvision

#dataset = util.ImageDataset("/hdd/imagenet/train")
#print(len(dataset))

dataset = torchvision.datasets.ImageNet("/hdd/imagenet", "train")

print(dataset)
