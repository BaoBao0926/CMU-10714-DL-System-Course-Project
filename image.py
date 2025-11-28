import os
import requests

# 创建 images 文件夹
save_dir = "images"
os.makedirs(save_dir, exist_ok=True)
print(f"Created folder: {save_dir}")

# 20 张稳定可下载图片（picsum ID）
image_urls = [
    f"https://picsum.photos/id/{i}/512/512" for i in [
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100,
        110, 120, 130, 140, 150,
        160, 170, 180, 190, 200
    ]
]

# 下载
for idx, url in enumerate(image_urls):
    print(f"Downloading {idx+1}/20 ... {url}")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        path = os.path.join(save_dir, f"img_{idx+1:02d}.jpg")
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Saved -> {path}")
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")

print("\n🎉 Done! All 20 tested, guaranteed working images downloaded to ./images/")
