import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import re
import gc

# Enable memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

# === Set paths ===
this_dir = os.path.dirname(__file__)
generation_path = os.path.abspath(os.path.join(this_dir))
sys.path.append(generation_path)

from models.generators import g_multivanilla

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check initial GPU memory
if torch.cuda.is_available():
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

model_dir = os.path.abspath(os.path.join(this_dir, "../../2025-04-18_03-58-38"))
input_image_path = os.path.abspath(os.path.join(this_dir, "../images/colusseum.png"))
output_path = os.path.abspath(os.path.join(this_dir, "output_singan_building.png"))

# === Load image ===
image = Image.open(input_image_path).convert("RGB")
transform = transforms.ToTensor()

# === Load state_dict to detect feature width with memory optimization ===
checkpoint_path = os.path.join(model_dir, "g_multivanilla.pt")
print(f"Loading model from: {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

# === Detect feature width dynamically from state_dict ===
def detect_feature_width(state_dict):
    for k, v in state_dict.items():
        if "curr.features.0.conv.weight" in k:
            return v.shape[0]  # e.g., 32 or 128
    return 32  # fallback

feature_width = detect_feature_width(state_dict)
print(f"ℹ️ Detected feature width: {feature_width}")

# === Initialize generator with memory optimization ===
print("Initializing generator...")
netG = g_multivanilla(min_features=feature_width, max_features=feature_width).to(device)

# Convert model to half precision to save memory
netG = netG.half()
# Convert state_dict to half precision
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
        state_dict[k] = v.half()

netG.load_state_dict(state_dict, strict=False)  # allow missing/extra keys
netG.eval()

# Free up memory
del state_dict
torch.cuda.empty_cache()
gc.collect()

# === Load amplitudes with memory optimization ===
print("Loading amplitudes...")
amps_path = os.path.join(model_dir, "amps.pt")
amps_dict = torch.load(amps_path, map_location=device, weights_only=True)

# Convert amplitudes to half precision
for k, v in amps_dict.items():
    if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
        amps_dict[k] = v.half()

print("🔍 amps dictionary:")
for k, v in amps_dict.items():
    print(f"{k}: {v} (type: {type(v)}, shape: {getattr(v, 'shape', 'N/A')})")

# === Generate output with reduced memory footprint ===
print("Generating image...")

# Process in chunks to reduce memory footprint
with torch.no_grad():  # Disable gradient tracking
    # Use smaller tensor dimensions (reduce from 25x25 if needed)
    tensor_size = 16  # Reduced from 25
    
    print(f"Creating reals with size {tensor_size}x{tensor_size}")
    reals = {f"s{i}": torch.randn(1, feature_width, tensor_size, tensor_size, 
                                device=device, dtype=torch.float16) 
            for i in range(len(amps_dict))}
    
    print(f"Creating noise tensors")
    noise = {}
    # Process noise generation in batches
    for k in amps_dict:
        noise[k] = torch.randn(1, feature_width, tensor_size, tensor_size, 
                             device=device, dtype=torch.float16) * amps_dict[k].half()
        print(f"Generated noise for {k}")
    
    # Check memory before forward pass
    if torch.cuda.is_available():
        print(f"Memory before generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    
    # Modified forward pass that's more memory efficient
    # This assumes the g_multivanilla implementation supports half precision
    fake = netG(reals, noise, amps_dict).clamp(0, 1).cpu()

# Free up GPU memory
del reals, noise, amps_dict, netG
torch.cuda.empty_cache()
gc.collect()

# === Save result ===
print("Saving output image...")
output_image = transforms.ToPILImage()(fake.squeeze(0))
output_image.save(output_path)
print(f"✅ Stylized image saved to: {output_path}")