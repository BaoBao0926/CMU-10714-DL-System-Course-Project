from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_20_cifar10_loader():

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root="./cifar-data",
        train=False,
        download=True,
        transform=transform
    )

    # 取前 20 张
    idxs = list(range(20))
    subset = Subset(dataset, idxs)

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
    )

    return loader
