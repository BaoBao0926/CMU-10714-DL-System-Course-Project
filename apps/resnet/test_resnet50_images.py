import torch
from torchvision import models
from PIL import Image
import os

# ----------------------------------------------------
# 1. 加载 torchvision 官方 ResNet50 + 预训练权重
# ----------------------------------------------------
print("Loading ResNet50 …")

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()   # ImageNet preprocess


# ----------------------------------------------------
# 2. 加载 ImageNet 类别标签（自动下载）
# ----------------------------------------------------
categories = weights.meta["categories"]


# ----------------------------------------------------
# 3. 遍历 ./images 文件夹中的所有图片
# ----------------------------------------------------
image_folder = "./data/images200"
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(("jpg", "jpeg", "png"))])

print(f"Found {len(image_files)} images.")

correct = 0

for idx, img_name in enumerate(image_files):
    img_path = os.path.join(image_folder, img_name)

    # ------------- Load image -------------
    img = Image.open(img_path).convert("RGB")

    # ----------- Preprocess (resize, crop, norm) -----------
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # shape: (1,3,224,224)

    # ----------- Forward pass -----------
    with torch.no_grad():
        output = model(input_batch)

    # ----------- Convert to probabilities -----------
    prob = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(prob, 5)

    print(f"\n===== Image {idx+1}: {img_name} =====")
    for i in range(5):
        print(f"{top5_prob[i].item()*100:.2f}%  --  {categories[top5_catid[i]]}")

print("\nDone!")
