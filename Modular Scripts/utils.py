"""
Contains functions for Saving and Loading a Model
"""

import torch

def save_model(model: torch.nn.Module,
               path: str):
  '''Saves model's state dict to the path mentioned

  Args:
    model: torch.nn.Module - PyTorch model to save.
    path: str - File path for saving model.
  '''

  if not ".pt" in path or ".pth" in path:
    print(f'Please mention the file extension: ".pt" or ".pth" in path')

  else:
    torch.save(obj=model.state_dict(), f=path)
  print(f'Model saved to {path}')



def load_model(constructed_model: torch.nn.Module, path: str):
  '''Loades a model's state dict into a existing model of the same architecture

  Args:
    constructed_model: torch.nn.Module - Pre-built model of the same architecture of saved model.
    path: str - Saved model's path.
  '''

  if not ".pt" in path or ".pth" in path:
    print(f'Please mention the file extension: ".pt" or ".pth" in path')
    return None
  else:
    constructed_model.load_state_dict(torch.load(f=path))
    return constructed_model
