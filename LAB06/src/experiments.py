import time
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader


def run_experiment():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    workers_list = [0, 1, 2, 4, 8]

    for workers in workers_list:

        loader = DataLoader(
            dataset,
            batch_size=128,
            num_workers=workers
        )

        start = time.perf_counter()

        for images, labels in loader:
            pass

        end = time.perf_counter()

        print(
            f"Workers={workers} "
            f"Time={end-start:.2f}s"
        )


if __name__ == "__main__":
    run_experiment()