
from torchvision.transforms.autoaugment import TrivialAugmentWide
import os
import data_setup, model_builder, engine, utils

import torch
from torchvision import transforms

import argparse
# Make Arguments
parser = argparse.ArgumentParser(
    prog='Training TinyVGG model Script',
    description='Trains a model, based on different Hyperparameter settings'
)

parser.add_argument('--epochs', type=int, help='Number of times model should look over data')
parser.add_argument('--batch_size', type=int, help='Batch Size for DataLoaders')
parser.add_argument('--hidden_units', type=int, help='Hidden Units for model')
parser.add_argument('--lr', type=float, help='Learning rate for model')
parser.add_argument('--train_path', type=str, help='Train Data path')
parser.add_argument('--test_path', type=str, help='Test Data Path')
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.lr
TRAIN_PATH = args.train_path
TEST_PATH = args.test_path

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting up Data Transformation Pipeline
train_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.65),
    transforms.TrivialAugmentWide(num_magnitude_bins=25),
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

# Set up our data into dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=TRAIN_PATH,
    test_dir=TEST_PATH,
    train_transform=train_transforms,
    test_transform=test_transforms,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count()
)

# Building our model
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)

# Training our model
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device
)

# Saving our model
utils.save_model(model=model, path='modular_scripts_model.pt')
