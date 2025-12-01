import os
os.environ["NEEDLE_BACKEND"] = "hip"
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from unet.unet_model import UNet
# 导入转换和融合工具
import needle as ndl
from torch2needle.torch2needle_converter import torch2needle_fx
from torch2needle.weight_converter import load_torch_weights_by_mapping
from operator_fusion.operator_fusion import OperatorFusion

# -----------------------------
# 1. 加载 UNet + 权重
# -----------------------------
def load_unet(weight_path):
    weight_path = weight_path

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"❌ 权重未找到: {weight_path}")

    net = UNet(n_channels=3, n_classes=2)
    state_dict = torch.load(weight_path, map_location="cpu")
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    print("✔ 模型加载成功")
    return net


# -----------------------------
# 2. 图像预处理（保持原分辨率）
# -----------------------------
def preprocess_keep_res(img):
    tf = transforms.Compose([transforms.ToTensor()])
    img_t = tf(img).unsqueeze(0)      # shape (1,3,H,W)
    return img_t

def convert_to_needle_with_fusion(torch_model,device,dtype):
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

# -----------------------------
# 3. 输出 mask（0/1 → 0/255）
# -----------------------------
def tensor_to_mask(mask_t, size):
    mask = mask_t.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).resize(size)
    return mask_img


# -----------------------------
# 4. 生成彩色透明 overlay
# -----------------------------
def create_overlay(original, mask_img, color=(0, 255, 0), alpha=0.4):
    overlay = original.copy().convert("RGBA")
    mask_colored = Image.new("RGBA", original.size, color + (0,))

    # mask 白色区域涂上颜色
    mask_pixels = mask_img.point(lambda p: 255 if p > 128 else 0)
    mask_pixels = mask_pixels.convert("L")

    mask_colored.putalpha(mask_pixels)

    # 原图 + 透明 mask
    blended = Image.alpha_composite(overlay, mask_colored)
    return blended.convert("RGB")


# -----------------------------
# 5. 并排对比原图 + mask
# -----------------------------
def side_by_side(original, mask_img):
    w, h = original.size
    canvas = Image.new("RGB", (2 * w, h), "white")
    canvas.paste(original, (0, 0))
    canvas.paste(mask_img.convert("RGB"), (w, 0))
    return canvas


# -----------------------------
# 6. 主流程：处理 img_23.jpg   ← 已按你要求改成 23
# -----------------------------
def process_img21():
    img_id = 21
    img_path = f"apps/Unet/images/img_{img_id}.jpg"
    output_dir = "apps/Unet/results"
    weight_path = "apps/Unet/unet_carvana_scale1.0_epoch2.pth"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {img_path}")

    # Load model
    net = load_unet(weight_path)

    # Read image
    img = Image.open(img_path).convert("RGB")
    img_t = preprocess_keep_res(img)

    # # convert to needle model
    # device = ndl.hip()
    # dtype = "float32"
    # net = convert_to_needle_with_fusion(net,device,dtype)
    # img_t = ndl.Tensor(img_t.numpy(), device=device, dtype=dtype)

    # Run inference
    with torch.no_grad():
        out = net(img_t)
        out = torch.softmax(torch.tensor(out.numpy()), dim=1)
        mask_t = out[:, 1, :, :]  # FG channel

    # Convert mask to PIL image
    mask_img = tensor_to_mask(mask_t, img.size)

    # Save pure mask
    mask_img.save(f"{output_dir}/img_{img_id}_mask.png")
    print("✔ 纯 mask saved.")

    # Create transparent overlay
    overlay_img = create_overlay(img, mask_img)
    overlay_img.save(f"{output_dir}/img_{img_id}_overlay.png")
    print("✔ 透明 overlay saved.")

    # Side-by-side compare
    compare_img = side_by_side(img, mask_img)
    compare_img.save(f"{output_dir}/img_{img_id}_compare.png")
    print("✔ 对比图 saved.")

    print("\n🎉 处理完成！输出位于 apps/Unet/results")


if __name__ == "__main__":
    process_img21()
