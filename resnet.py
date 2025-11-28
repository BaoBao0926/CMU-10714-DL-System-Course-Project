import torch
from torchvision import models
from PIL import Image

# 1. Load model + correct preprocessing
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()   # <-- correct preprocess for inference

# 2. Load image
img = Image.open("test.jpg").convert("RGB")

# 3. Preprocess image
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # shape: (1,3,224,224)

# 4. Inference
with torch.no_grad():
    output = model(input_batch)  # logits

# 5. Convert logits to probabilities
prob = torch.nn.functional.softmax(output[0], dim=0)

print(prob)