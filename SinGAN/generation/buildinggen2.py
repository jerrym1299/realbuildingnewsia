import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import sys

# Set PyTorch memory management configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import your models (assuming they're in the proper path)
sys.path.append('/kaggle/working/realbuildingnewsia/SinGAN')
from generation.models.generators import GeneratorMultiScale

def load_model(checkpoint_path, amps_path, device):
    """Load model with memory-optimized settings"""
    print(f"Loading model from {checkpoint_path}")
    
    # Use weights_only=True for security and memory optimization
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Feature width detection
    feature_width = None
    for key, value in state_dict.items():
        if 'body.0.weight' in key:
            feature_width = value.size(0)
            break
    
    if feature_width:
        print(f"ℹ️ Detected feature width: {feature_width}")
    else:
        feature_width = 128  # Default if not found
        print(f"⚠️ Couldn't detect feature width, using default: {feature_width}")
    
    # Load amps dictionary
    print(f"Loading amplitude dictionary from {amps_path}")
    amps_dict = torch.load(amps_path, map_location=device, weights_only=True)
    
    # Print amps dictionary for debugging
    print("🔍 amps dictionary:")
    for key, value in amps_dict.items():
        print(f"{key}: {value.item() if isinstance(value, torch.Tensor) else value} "
              f"(type: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'})")
    
    return state_dict, amps_dict, feature_width

def generate_image(netG, reals, amps, device, batch_size=1, chunk_size=2, max_scale=12):
    """Generate image in chunks to avoid memory issues"""
    # Create noise in chunks and process separately
    noise_list = []
    
    # Determine the number of scales to use
    # Reduce max_scale if needed to save memory
    num_scales = min(len(reals), max_scale)
    print(f"Using {num_scales} scales out of {len(reals)} available")
    
    # Process in chunks to reduce memory usage
    with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
        # Initialize with base noise
        base_noise = torch.randn(1, 3, reals[-1].shape[2], reals[-1].shape[3], device=device)
        current_img = netG.init_forward(base_noise, reals, amps)
        
        # Free memory
        del base_noise
        torch.cuda.empty_cache()
        
        # Process each scale
        for scale_idx in range(1, num_scales):
            key = f's{scale_idx}'
            if key in amps:
                # Process current scale
                noise = torch.randn(1, 3, reals[num_scales-scale_idx-1].shape[2], 
                                  reals[num_scales-scale_idx-1].shape[3], device=device)
                
                # Scale the noise tensor to use less memory if needed
                noise_amp = amps[key].view(-1, 1, 1, 1)
                
                # Process and update current image
                current_img = netG.forward_single_scale(current_img, noise, noise_amp, scale_idx, num_scales)
                
                # Free memory after each scale
                del noise
                torch.cuda.empty_cache()
                
                print(f"Processed scale {scale_idx}/{num_scales-1}")
    
    return current_img.clamp(0, 1).cpu()

def main():
    # Paths
    checkpoint_path = "/kaggle/working/realbuildingnewsia/SinGAN/checkpoints/building1/netG.pth"
    amps_path = "/kaggle/working/realbuildingnewsia/SinGAN/checkpoints/building1/amps.pth"
    
    # Monitor initial memory usage
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    
    # Load model with optimized settings
    state_dict, amps_dict, feature_width = load_model(checkpoint_path, amps_path, device)
    
    # Create generator with reduced parameters if possible
    netG = GeneratorMultiScale(feature_width).to(device)
    
    # Convert to half precision to save memory
    netG = netG.half()
    
    # Load state dict
    try:
        netG.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Prepare reals (assuming this part of your code remains the same)
    # For example:
    # This is a placeholder - replace with your actual code for loading reals
    reals = []
    for scale in range(12):  # Assuming 12 scales
        # Create dummy data for demonstration
        # Replace with your actual image data loading
        h = 32 * (2 ** scale)
        w = 32 * (2 ** scale)
        reals.append(torch.zeros(1, 3, h, w, device=device).half())  # Use half precision
    
    # Print memory usage after model loading
    if torch.cuda.is_available():
        print(f"Memory after model loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    
    # Clear unnecessary variables to free memory
    del state_dict
    torch.cuda.empty_cache()
    
    # Start with model in eval mode to save memory
    netG.eval()
    with torch.no_grad():  # Disable gradient computation
        fake = generate_image(netG, reals, amps_dict, device)
    
    # Save the generated image
    fake_img = np.transpose(fake[0].numpy(), (1, 2, 0)) * 255
    fake_img = fake_img.astype(np.uint8)
    
    # Convert to PIL image and save
    pil_img = Image.fromarray(fake_img)
    pil_img.save("generated_building.png")
    print("Generated image saved as 'generated_building.png'")

if __name__ == "__main__":
    main()