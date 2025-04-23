import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import re

# === Set paths ===
this_dir = os.path.dirname(__file__)
generation_path = os.path.abspath(os.path.join(this_dir))
sys.path.append(generation_path)

from models.generators import g_multivanilla

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = os.path.abspath(os.path.join(this_dir, "../../2025-04-18_03-58-38"))
input_image_path = os.path.abspath(os.path.join(this_dir, "../images/colusseum.png"))
output_path = os.path.abspath(os.path.join(this_dir, "output_singan_building.png"))

# === Load image ===
image = Image.open(input_image_path).convert("RGB")
transform = transforms.ToTensor()
# Generate fake reals with matching shape and feature width


# === Load state_dict to detect feature width ===
checkpoint_path = os.path.join(model_dir, "g_multivanilla.pt")
state_dict = torch.load(checkpoint_path, map_location=device)

# === Detect feature width dynamically from state_dict ===
def detect_feature_width(state_dict):
    for k, v in state_dict.items():
        if "curr.features.0.conv.weight" in k:
            return v.shape[0]  # e.g., 32 or 128
    return 32  # fallback

feature_width = detect_feature_width(state_dict)
print(f"ℹ️ Detected feature width: {feature_width}")

# === Initialize generator and load state dict ===
netG = g_multivanilla(min_features=feature_width, max_features=feature_width).to(device)
netG.load_state_dict(state_dict, strict=False)  # allow missing/extra keys
netG.eval()

# === Load amplitudes ===
# === Load amplitudes ===
amps_path = os.path.join(model_dir, "amps.pt")
amps_dict = torch.load(amps_path, map_location=device)

# ✅ Convert dict to a list sorted by scale index
amps = [amps_dict[f"s{i}"] for i in range(len(amps_dict))]
print("🔍 amps dictionary:")
for k, v in amps_dict.items():
    print(f"{k}: {v} (type: {type(v)}, shape: {getattr(v, 'shape', 'N/A')})")

def generate_noise_like_model(model, amps):
    noise = {}
    for key in amps:
        amp = amps[key]
        block = getattr(model.prev, key) if hasattr(model.prev, key) else model.curr  # fallback
        shape = block.features[0].conv.weight.shape  # shape: (out_channels, in_channels, kH, kW)
        out_channels = shape[0]
        H = W = 25  # 🔧 default small size, adjust if needed
        noise[key] = torch.randn((1, out_channels, H, W), device=device) * amp
    return noise

# === Generate output ===
# Keep amps as a dict
amps = amps_dict
reals = {f"s{i}": torch.randn(1, feature_width, 25, 25, device=device) for i in range(len(amps))}

# === Generate output ===
with torch.no_grad():
    noise = {k: torch.randn(1, feature_width, 25, 25, device=device) * amps[k].item() for k in amps}
    fake = netG(reals, noise, amps).clamp(0, 1).cpu()


# === Save result ===
output_image = transforms.ToPILImage()(fake.squeeze(0))
output_image.save(output_path)
print(f"✅ Stylized image saved to: {output_path}")
