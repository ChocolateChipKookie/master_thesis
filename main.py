
def train():
    from colorful.solver import Solver
    import torch

    config = {}
    config['device'] = "cuda:0"
    config['dtype'] = torch.float32
    config['lambda'] = 0.5
    config['lr'] = 3e-5
    config['weight_decay'] = 1e-3
    config['iterations'] = 200000
    config['batch_size'] = 40

    config['data_path'] = "./imagenet/train"
    config['dataloader_workers'] = 4

    config['validate_every'] = 1000
    config['val_data_path'] = "./imagenet/val"
    config['val_data_size'] = 25000
    config['snapshot_every'] = 1
    config['snapshot_dir'] = "/hdd/adi/colorful/snapshots"

    config['progress_every'] = 10
    config['progress_dir'] = './tmp/progress'

#    config['model_file'] = '/hdd/adi/colorful/snapshots/26_03(18:46:24)-116000_7330.pth'
#    config['start_iteration'] = 116001
    config['backup_dir'] = 'tmp'
    config['backup_every'] = 200

#    config['model_file'] = 'tmp/snapshots/23_03(05:06:05)-19000_9896.pth'
#    config['start_iteration'] = 19001

    solver = Solver(config)
    solver.train()


def colorize():
    import colorful.model
    import torch
    import torchvision
    import util
    import matplotlib.pyplot as plt
    from skimage import color

    model = colorful.model.Colorful()
    model.load_state_dict(torch.load('tmp/snapshots/23_03(05:06:05)-19000_9896.pth'))
    model.eval()

    # Dataset and data loader
    transform = torchvision.transforms.Compose([
        util.ShortResize(256),
        torchvision.transforms.RandomCrop(256),
        util.rgb2lab(),
        torchvision.transforms.ToTensor(),
    ])

    fix, (ax1, ax2, ax3) = plt.subplots(1, 3)

    dataset = torchvision.datasets.ImageFolder("./imagenet/val", transform=transform)
    image = dataset[3000][0]
    l_channel = image[:1].type(torch.float)

    image_normal = color.lab2rgb(image.permute(1, 2, 0))
    image_gs = torch.cat(3*[l_channel]).permute(1, 2, 0)
    image_gs /= 100

    image_colorized = model(l_channel.view(1, *l_channel.shape))
    image_colorized = color.lab2rgb(image_colorized)

    ax1.imshow(image_gs)
    ax2.imshow(image_normal)
    ax3.imshow(image_colorized)

    plt.show()


def restore():
    from colorful.solver import Solver
    import json
    import torch
    settings = json.loads(open("tmp/state.json", 'r').read())
    config = torch.load("tmp/config.pth")
    iter = settings['iter']

    config['lr'] = 3e-4
    config['model_file'] = 'tmp/backup_model.pth'
    config['start_iteration'] = iter + 1
    solver = Solver(config)
    solver.train()


def filter(src_path, out_file, threshold_val=10, threshold_percentage=0.9):
    import torchvision
    import torch
    import util
    transform = torchvision.transforms.Compose([
        util.ShortResize(256),
        torchvision.transforms.RandomCrop(256),
        util.rgb2lab(),
        torchvision.transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(src_path, transform=transform)

    def is_grayscale(img):
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

    valid = []
    invalid = []
    for i, img in enumerate(dataset):
        img = img[0]
        if not is_grayscale(img):
            valid.append(i)
        else:
            util.display_lab(img.permute(1, 2, 0))
            print(f"Not valid {i}")
            invalid.append(i)

    print(f'Total valid {len(valid)}\n Not valid: {len(invalid)}')
    with open(out_file, 'w') as file:
        for i in valid:
            file.write(f'{i}\n')


if __name__ == '__main__':
    filter("./imagenet/val", './masks/val.txt')
