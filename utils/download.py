import os
import urllib.request

def download_ckpt(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading checkpoint from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print("✅ Download complete.")
    else:
        print("✅ Checkpoint already exists.")

# Replace with your actual link
ckpt_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
ckpt_path = "weights/sam_vit_b_01ec64.pth"

# Create checkpoints folder if needed
os.makedirs("weights", exist_ok=True)

# Download
download_ckpt(ckpt_url, ckpt_path)
