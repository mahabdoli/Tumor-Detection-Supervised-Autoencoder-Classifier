from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os


TRAIN_DIR = '/content/drive/MyDrive/med project/med/input/train'
VALID_DIR = '/content/drive/MyDrive/med project/med/input/valid'
IMAGE_SIZE = 32 # Image size of resize when applying transforms.
BATCH_SIZE = 32
NUM_WORKERS = 2 # Number of parallel processes for data preparation.

# Training transforms
def get_train_transform(IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return train_transform

# Validation transforms
def get_valid_transform(IMAGE_SIZE):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return valid_transform

def get_datasets():
    """
    Function to prepare the Datasets.

    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        TRAIN_DIR, 
        transform=(get_train_transform(IMAGE_SIZE))
    )
    dataset_valid = datasets.ImageFolder(
        VALID_DIR, 
        transform=(get_valid_transform(IMAGE_SIZE))
    )
    return dataset_train, dataset_valid, dataset_train.classes

def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    class_weights = []
    for root, subdir, files in os.walk(TRAIN_DIR):
        if len(files) > 0:
            class_weights.append(1/len(files))
            
    sample_weights = [0] * len(dataset_train)
    
    for idx, (data, label) in enumerate(dataset_train):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    
    sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        sampler=sampler, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader 