import os
os.environ["NEEDLE_BACKEND"] = "hip"
import torch
from torchvision import models
from PIL import Image
import numpy as np
import needle as ndl
from needle import Tensor

# --- converter / loader ---
from torch2needle.torch2needle_converter import torch2needle_fx
from torch2needle.weight_converter import load_torch_weights_by_mapping
from operator_fusion.operator_fusion import OperatorFusion


# ------------------------------------------------------
# 1. Load torchvision ResNet101 + preprocess + label map
# ------------------------------------------------------
def load_torch_resnet101():
    weights = models.ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    return model, preprocess, categories


# ------------------------------------------------------
# 2. Convert ResNet101 to Needle
# ------------------------------------------------------
def convert_resnet101_to_needle(torch_model, device, dtype):
    needle_model, trace_log, mapping = torch2needle_fx(
        torch_model,
        device=device,
        dtype=dtype
    )
    load_torch_weights_by_mapping(
        mapping, verbose=True,
        device=device,
        dtype=dtype
    )
    needle_model.eval()
    fusion_engine = OperatorFusion()
    fused_model = fusion_engine.fuse_model(needle_model)
    fused_model.eval()
    return fused_model


# ------------------------------------------------------
# 3. PyTorch inference
# ------------------------------------------------------
def infer_torch(model, img, preprocess):
    x = preprocess(img)
    x = x.unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out[0], dim=0)
    return prob.cpu().numpy()


# ------------------------------------------------------
# 4. Needle inference
# ------------------------------------------------------
def infer_needle(model, img, preprocess, device, dtype):
    x = preprocess(img)
    x = x.unsqueeze(0).numpy()
    x_nd = Tensor(x, device=device, dtype=dtype)
    out = model(x_nd)
    prob = out.numpy()[0]
    prob = np.exp(prob) / np.sum(np.exp(prob))
    return prob


# ------------------------------------------------------
# 5. Evaluate accuracy on 20 images
# ------------------------------------------------------
def evaluate_on_images(torch_model, needle_model,image_dir, preprocess, categories, device, dtype):
    image_dir = image_dir
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(("jpg","jpeg","png"))])

    torch_correct = 0
    needle_correct = 0

    print(f"\nFound {len(files)} images.\n")

    for idx, name in enumerate(files, 1):
        img = Image.open(os.path.join(image_dir, name)).convert("RGB")

        torch_prob = infer_torch(torch_model, img, preprocess)
        torch_pred = torch_prob.argmax()

        needle_prob = infer_needle(needle_model, img, preprocess, device, dtype)
        needle_pred = needle_prob.argmax()

        label_t = categories[torch_pred]
        label_n = categories[needle_pred]

        print(f"\n===== Image {idx}: {name} =====")
        print(f"PyTorch: {label_t}")
        print(f"Needle : {label_n}")
        print(f"Agreement: {torch_pred == needle_pred}")

        if torch_pred == needle_pred:
            needle_correct += 1

        torch_correct += 1

    print("\n==============================")
    print(f"PyTorch accuracy: {torch_correct}/{len(files)}")
    print(f"Needle accuracy:  {needle_correct}/{len(files)}")
    print(f"Accuracy diff:    {torch_correct - needle_correct}")
    print("==============================\n")


# ------------------------------------------------------
# 6. Main
# ------------------------------------------------------
if __name__ == "__main__":
    print("\n===== Step 1: Load PyTorch ResNet101 =====")
    torch_model, preprocess, categories = load_torch_resnet101()

    print("\n===== Step 2: Convert to Needle =====")
    device = ndl.hip()
    dtype = "float32"

    needle_model = convert_resnet101_to_needle(
        torch_model, device, dtype
    )

    print("\n===== Step 3: Evaluate on 20 images =====")
    image_dir = "./data/images200"
    evaluate_on_images(
        torch_model, needle_model,image_dir,
        preprocess, categories,
        device, dtype
    )
