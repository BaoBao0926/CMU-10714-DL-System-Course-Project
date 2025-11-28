import os
import requests

save_dir = "images"
os.makedirs(save_dir, exist_ok=True)

url = "https://picsum.photos/id/237/512/512"   # ✔ 永远可用
save_path = os.path.join(save_dir, "img_15.jpg")

print("Downloading replacement for img_15.jpg ...")

try:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    print(f"✅ Saved replacement image -> {save_path}")
except Exception as e:
    print(f"❌ Error: {e}")
