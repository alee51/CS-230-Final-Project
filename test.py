import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from torchvision.utils import save_image
import os
from tqdm import tqdm

model_num = 10 # epochs in training
prefix = "" # prefix to find saved model

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = f"./{prefix}finetuned_model_semantic-{model_num}"
base_model_id = "Manojb/stable-diffusion-2-1-base"

num_samples = 100
image_size = 256
num_inference_steps = 5

output_dir = f"./{prefix}test_results-{model_num}"
os.makedirs(output_dir, exist_ok=True)


print(f"Loading base components from {base_model_id}...")
vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
scheduler = PNDMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

print(f"Loading fine-tuned UNet from {model_path}...")
unet = UNet2DConditionModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

print(f"Generating {num_samples} images...")
latents = torch.randn( # Start with random latents
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

for t in tqdm(scheduler.timesteps):
    latent_model_input = scheduler.scale_model_input(latents, timestep=t)

    with torch.no_grad():
        noise_pred = unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=encoder_hidden_states
        ).sample

    latents = scheduler.step(noise_pred, t, latents).prev_sample


print("Decoding images...")
with torch.no_grad():
    latents = latents / 0.18215 # Scale latents back to valid VAE range
    image = vae.decode(latents).sample

# Post-process: [-1, 1] -> [0, 1]
image = ((image + 1) / 2).clamp(0, 1)

for i in range(num_samples):
    save_path = os.path.join(output_dir, f"generated_sample_{i}.png")
    save_image(image[i], save_path)
    print(f"Saved: {save_path}")

print("\nTest complete.")
