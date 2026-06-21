import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=32, data_dir='./data/chest_xray'):
    """
    Loads the Chest X-Ray Pneumonia dataset.
    Images are resized to 64x64 grayscale to keep training fast on M4.
    """

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # convert to grayscale
        transforms.Resize((64, 64)),                  # resize all images to 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))          # normalize to [-1, 1]
    ])

    train_dataset = datasets.ImageFolder(
        root=f'{data_dir}/train',
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=f'{data_dir}/val',
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        root=f'{data_dir}/test',
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")  # ['NORMAL', 'PNEUMONIA']

    return train_loader, val_loader, test_loader