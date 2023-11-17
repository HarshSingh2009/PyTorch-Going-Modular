"""
contains functions to load a model and then make predictions on it
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import model_builder
import ast

import argparse
parser = argparse.ArgumentParser(
    prog='Predicting using  TinyVGG model in Script',
    description='Predicts on Image using TinyVGG model'
)

parser.add_argument('--model_saved_path', type=str, help='Path at which PyTorch model is saved')
parser.add_argument('--img_path', type=str, help='Path of Image')
parser.add_argument('--img_shape', type=int, help='shape of Image needed by PyTorch model')
parser.add_argument('--class_names', type=str, help='Class names inside a string with 1 space between each')
args = parser.parse_args()

model_path = args.model_saved_path
img_path = args.img_path
img_size = args.img_shape
class_names = ast.literal_eval(args.class_names)

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=16,
    output_shape=len(class_names)
).to(device)

loaded_model.load_state_dict(torch.load(model_path))

# set up image resizing transformation
custom_image_transform = transforms.Compose([
  transforms.Resize(size=(img_size, img_size), antialias=True)
])

img = torchvision.io.read_image(img_path).type(torch.float) / 255.
input_tensor = custom_image_transform(img).unsqueeze(dim=0).to(device)

loaded_model.eval()
with torch.inference_mode():
  y_logits = loaded_model(input_tensor)
y_probs = torch.softmax(y_logits, dim=1)
y_preds = torch.argmax(y_probs, dim=1)

plt.imshow(input_tensor.cpu().squeeze().permute([1, 2, 0]))
plt.title(f'Class: {class_names[y_preds.item()]} \nProbs: {torch.max(y_probs).item():.3f}%')
plt.axis('off')
plt.show()
