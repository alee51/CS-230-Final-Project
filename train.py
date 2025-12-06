import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from torchvision.utils import save_image
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler
from tqdm import tqdm
import os
from pathlib import Path
import bitsandbytes as bnb 

# Settings
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 256
latent_channels = 4

# Training settings
batch_size = 3  # Any more and CUDA runs out of memory
num_epochs = 1
learning_rate = 1e-4
save_every = 500
gradient_accumulation_steps = 8 

# Semantic Loss Settings
lambda_sem = 0.1

train_image_dir = "../imagenet256x256/" 
output_dir = f"./lr4-lambda10-finetuned_model_semantic-{num_epochs}"
checkpoint_dir = f"./lr4-lambda10-checkpoints_semantic-{num_epochs}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Dataset
class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_paths = list(Path(image_dir).glob("**/*.jpg"))
        
        from random import shuffle
        shuffle(self.image_paths)
        self.image_paths = self.image_paths[:1000]  # Limit to 1k images for faster testing
        
        import shutil
        for i, path in enumerate(self.image_paths):
            print(path)
            shutil.copyfile(str(path), str(f'../imagenet-sample/{i:03}.jpg'))
            
        exit(0)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_tensor = self.transform(image) * 2 - 1 
        return image_tensor

# Load models
print("Loading SD models...")
vae = AutoencoderKL.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="unet").to(device)
noise_scheduler = DDPMScheduler.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="scheduler")

print("Loading ResNet50...")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
resnet.fc = torch.nn.Identity()
resnet.eval()
resnet.requires_grad_(False) # Freeze ResNet completely

vae.requires_grad_(False) 
unet.enable_gradient_checkpointing()
unet.enable_xformers_memory_efficient_attention()

resnet_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resnet_resize = transforms.Resize((224, 224)) # ResNet native resolution

def process_for_resnet(images_tensor):
    """
    Input: Images in range [-1, 1] (SD format)
    Output: Images in range [0, 1] -> Normalized -> Resized for ResNet
    """
    img = (images_tensor + 1) / 2
    img = resnet_resize(img)
    img = resnet_normalization(img)
    return img

# Prepare training
dataset = ImageDataset(train_image_dir, image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

seq_length = 77
encoder_hidden_states_template = torch.zeros((1, seq_length, 1024), device=device)

# Training loop
print(f"Starting training with Semantic Loss (Weight: {lambda_sem})...")

global_step = 0
unet.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for step, images in enumerate(progress_bar):
        images = images.to(device)
        
        with torch.no_grad():
            clean_images_processed = process_for_resnet(images)
            target_semantic_embedding = resnet(clean_images_processed).detach()

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        encoder_hidden_states = encoder_hidden_states_template.expand(images.shape[0], -1, -1)
        
        with torch.cuda.amp.autocast():
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            
            mse_loss = F.mse_loss(noise_pred, noise)
            
            alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps]
            beta_prod_t = 1 - alpha_prod_t
            
            alpha_prod_t = alpha_prod_t.flatten().view(-1, 1, 1, 1)
            beta_prod_t = beta_prod_t.flatten().view(-1, 1, 1, 1)
            
            pred_original_latents = (noisy_latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            
            pred_pixels = vae.decode(pred_original_latents / 0.18215).sample
            
            pred_pixels_processed = process_for_resnet(pred_pixels)
            
            pred_semantic_embedding = resnet(pred_pixels_processed)
            
            sem_loss = F.cosine_embedding_loss(
                pred_semantic_embedding, 
                target_semantic_embedding, 
                target=torch.ones(pred_semantic_embedding.shape[0], device=device)
            )
            
            total_loss = mse_loss + (lambda_sem * sem_loss)
            total_loss = total_loss / gradient_accumulation_steps
        
        scaler.scale(total_loss).backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        
        # Logging
        current_loss = total_loss.item() * gradient_accumulation_steps
        epoch_loss += current_loss
        progress_bar.set_postfix({
            "mse": f"{mse_loss.item():.4f}", 
            "sem": f"{sem_loss.item():.4f}",
            "total": f"{current_loss:.4f}"
        })
        
        # Save checkpoint
        if global_step > 0 and global_step % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
            torch.save({
                'step': global_step,
                'unet_state_dict': unet.state_dict(),
            }, checkpoint_path)
    
    print(f"Epoch {epoch+1} average loss: {epoch_loss / len(dataloader):.4f}")

# Save and Test
print("Saving final model...")
unet.save_pretrained(output_dir)

del optimizer
torch.cuda.empty_cache()
