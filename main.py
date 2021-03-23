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
    config['batch_size'] = 20

    config['data_path'] = "./imagenet/val"
    config['dataloader_workers'] = 4

    config['validate_every'] = 1000
    config['val_data_path'] = "./imagenet/val"
    config['val_data_size'] = 25000
    config['snapshot_every'] = 1
    config['snapshot_dir'] = "./tmp/snapshots"

    config['progress_every'] = 10
    config['progress_dir'] = './tmp/progress'

    config['model_file'] = 'tmp/snapshots/23_03(05:06:05)-19000_9896.pth'
    config['start_iteration'] = 19001

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


if __name__ == '__main__':
    train()