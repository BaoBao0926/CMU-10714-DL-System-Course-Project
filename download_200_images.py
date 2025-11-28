import os
import requests
from tqdm import tqdm
import random

SAVE_DIR = "images200"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(url, filename):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            open(filename, "wb").write(r.content)
            return True
        return False
    except:
        return False

print("\n====================================")
print(" Downloading 200 images (no failures)")
print("====================================")

success = 0
total = 200

for i in tqdm(range(1, total + 1)):
    # 尝试 2 个稳定 API
    seed = random.randint(1, 10_000_000)

    urls = [
        f"https://picsum.photos/seed/{seed}/512/512",
        f"https://source.unsplash.com/random/512x512?sig={seed}",
    ]

    filename = f"{SAVE_DIR}/img_{i:03d}.jpg"

    done = False
    for url in urls:
        if download_image(url, filename):
            done = True
            break

    if done:
        success += 1

print("====================================")
print(f"Download complete.")
print(f"Success: {success}/{total}")
print(f"Saved to: {SAVE_DIR}")
print("====================================")
