
# Model with weight: Inference

## 1. ViT for Image Classification
[ViT](https://github.com/lukemelas/PyTorch-Pretrained-ViT/tree/master#) evaluated on ImageNet1k

```bash
from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=True)
```

## 2.ResNet18 for Image Classification
[ResNet]( )

```bash
import torch
from torchvision import models
from PIL import Image

# 1. Load model + correct preprocessing
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()   # <-- correct preprocess for inference

# 2. Load image
img = Image.open("image.jpg").convert("RGB")

# 3. Preprocess image
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # shape: (1,3,224,224)

# 4. Inference
with torch.no_grad():
    output = model(input_batch)  # logits

# 5. Convert logits to probabilities
prob = torch.nn.functional.softmax(output[0], dim=0)

print(prob)
```
## 3. UNet for image segmentation

[UNet](https://github.com/milesial/Pytorch-UNet) on Carvana dataset

you should refer this repo.

# Model without weigh

## 1. Vision Transfomer Variants

You could follow this [website](https://github.com/lucidrains/vit-pytorch). You should select 3 models (better including 3D ViT)

## 2.NLP variants

You could follow this [website](https://github.com/graykode/nlp-tutorial) to achieve some basic algorithm (world2vec, textCNN, BERT)





