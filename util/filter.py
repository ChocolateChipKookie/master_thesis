import torch
import torchvision.transforms
from module import ShortResize, rgb2lab

def is_grayscale(img, threshold_val=10, threshold_percentage=.9):
    def check_channel(channel):
        # Checks if threshold percentage of pixels are in range -threshold < channel < threshold
        total_elements = channel.shape[0] * channel.shape[1]
        threshold_elements = total_elements * threshold_percentage
        lo = -threshold_val <= channel
        hi = channel <= threshold_val
        in_range = torch.logical_and(hi, lo)
        non_zero = torch.count_nonzero(in_range)
        return non_zero > threshold_elements

    return check_channel(img[1]) and check_channel(img[2])


def is_monochrome(img, min_eccentricity=7.5):
    avg = img.mean(dim=[1, 2])
    centered = img[1:] - avg[1:, None, None]
    dist = torch.linalg.norm(centered, dim=0)
    mean = dist.mean().item()
    return mean < min_eccentricity


def filter_dataset(src_path, func, invert=False, log_period=None):
    transform = torchvision.transforms.Compose([
        ShortResize(256),
        torchvision.transforms.RandomCrop(256),
        rgb2lab(),
        torchvision.transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(src_path, transform=transform)

    valid = []
    invalid = []

    def get_log():
        def log_function(index):
            if index % log_period == 0:
                print(f'[{i} / {len(dataset)}] valid: {len(valid)}- invalid: {len(invalid)}')

        def log_dummy(index):
            pass

        if log_period:
            return log_function
        else:
            return log_dummy

    log = get_log()

    if invert:
        is_valid = lambda x: not func(x)
    else:
        is_valid = func

    for i, img in enumerate(dataset):
        img = img[0]
        log(i)
        if is_valid(img):
            valid.append(i)
        else:
            invalid.append(i)
    return valid, invalid
