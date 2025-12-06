import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from torchvision.utils import save_image
import os
from tqdm import tqdm

# Settings
model_num = 10
prefix = ""

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = f"./{prefix}finetuned_model_semantic-{model_num}"  # Point this to your saved model folder
base_model_id = "Manojb/stable-diffusion-2-1-base"

# Inference settings
num_samples = 100
image_size = 256
num_inference_steps = 5
guidance_scale = 1.0

# Output
output_dir = f"./{prefix}-test_results-{model_num}"
os.makedirs(output_dir, exist_ok=True)

# Load Models
print(f"Loading base components from {base_model_id}...")
vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae").to(device)
scheduler = PNDMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

print(f"Loading fine-tuned UNet from {model_path}...")
unet = UNet2DConditionModel.from_pretrained(model_path).to(device)

vae.half()
unet.half()

# Inference Loop
print(f"Generating {num_samples} images...")

# Create random noise
latents = torch.randn(
    (num_samples, unet.config.in_channels, image_size // 8, image_size // 8),
    device=device,
    dtype=torch.float16
)

seq_length = 77
encoder_hidden_states = torch.zeros(
    (num_samples, seq_length, 1024), 
    device=device, 
    dtype=torch.float16
)

scheduler.set_timesteps(num_inference_steps)
latents = latents * scheduler.init_noise_sigma

# Denoising loop
for t in tqdm(scheduler.timesteps):
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # Predict the noise residual
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=encoder_hidden_states
        ).sample

    latents = scheduler.step(noise_pred, t, latents).prev_sample

# Decode and Save
print("Decoding images...")
with torch.no_grad():
    latents = latents / 0.18215
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1)

for i in range(num_samples):
    save_path = os.path.join(output_dir, f"generated_sample_{i}.png")
    save_image(image[i], save_path)
    print(f"Saved: {save_path}")

print("\nTest complete.")
