"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import torch
import torchvision
import os

BATCH_SIZE=32
NUM_WORKERS=os.cpu_count()

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       train_transform: torchvision.transforms.Compose,
                       test_transform: torchvision.transforms.Compose,
                       batch_size: int = BATCH_SIZE,
                       num_workers: int = NUM_WORKERS):
  '''This Function helps in loading our dataset and finally converting them into Data Loaders for our model to train

  Args:
    train_dir: str - Train data Directory Path.
    test_dir: str - Test data Directory Path.
    train_transform: torchvision.transforms.Compose - torchvision transforms to perform on training data.
    train_transform: torchvision.transforms.Compose - torchvision transforms to perform on testing data.
    batch_size: int - Number of samples per batch in each of the DataLoaders.
    num_workers: int - An integer for number of workers per DataLoader.

  Returns: (train_dataloader, test_dataloader, class_names)
    train_dataloader: torch.utils.data.DataLoader - Train data converted to iterable batches.
    test_dataloader: torch.utils.data.DataLoader - Test data converted to iterable batches.
    class_names: list - Class Names found while loading Data

  Example:
  train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir="path/to/train_data/",
    test_dir="path/to/test_data/",
    train_transform=train_data_transform_pipeline,
    test_transform=test_data_transform_pipeline,
    batch_size=32,
    num_workers=16
  )

  '''

  # Converting our data into PyTorch ImageFolder Datasets
  train_data = torchvision.datasets.ImageFolder(
      root=train_dir,
      transform=train_transform
  )

  test_data = torchvision.datasets.ImageFolder(
      root=test_dir,
      transform=test_transform
  )

  print(f'Found {len(train_data)} images in train directory')
  print(f'Found {len(test_data)} images in test directory')

  class_names = train_data.classes

  # Converting our PyTorch ImageFolder Datasets into DataLoaders
  train_dataloader = torch.utils.data.DataLoader(
      dataset=train_data,
      shuffle=True,
      batch_size=batch_size,
      num_workers=num_workers
  )

  test_dataloader = torch.utils.data.DataLoader(
      dataset=test_data,
      shuffle=False,
      batch_size=batch_size,
      num_workers=num_workers
  )

  print('---------------')
  print(f'Train Data has {len(train_dataloader)} batches of size {train_dataloader.batch_size}')
  print(f'Test Data has {len(test_dataloader)} batches of size {test_dataloader.batch_size}')

  return train_dataloader, test_dataloader, class_names
