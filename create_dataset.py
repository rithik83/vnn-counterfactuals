import os
import torch
import torchvision
import torchvision.transforms as T

# Use a directory in Colab to store datasets
DATASET_DIR = "/content/data"

def build_dataset(is_train=False):

    transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
    
    dataset = torchvision.datasets.MNIST(
        root=DATASET_DIR, train=is_train, download=True, transform=transform
    )

    return dataset