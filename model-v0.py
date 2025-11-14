import os, random
from glob import glob
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

from diffusers import UNet2DModel, DDPMScheduler

import torchvision.transforms as T
from PIL import Image

# -----------------------------------------------------------------

# Original configs:
# class Config:
#     data_root = "imagenet256x256"
#     image_size = 256                # [may test on 64x64 instead of 256x256 first]
#     train_split = 0.9               # 90% train, 10% val
#     batch_size = 8                  # adjust for GPU; 8–16 recommended
#     lr = 1e-4
#     weight_decay = 0.0
#     max_steps = 20000               # increase to 100k+ for full training
#     num_train_timesteps = 1000
#     lambda_sem = 0.1                # SLCD loss weight
#     sem_encoder_name = "resnet50"   # can swap for DINOv3 later
#     log_interval = 50

# Small test configs:
class Config:
    data_root = "imagenet256x256"
    image_size = 256                # [may test on 64x64 instead of 256x256 first]
    train_split = 0.9               # 90% train, 10% val
    batch_size = 8                  # adjust for GPU; 8–16 recommended
    lr = 1e-4
    weight_decay = 0.0
    max_steps = 20000               # increase to 100k+ for full training
    num_train_timesteps = 100
    lambda_sem = 0.1                # SLCD loss weight, hyperparameter
    sem_encoder_name = "resnet50"   # can swap for DINOv3 later
    log_interval = 1

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# -----------------------------------------------------------------

def build_split_lists(root_dir, train_ratio=0.9, seed=42):
    root = Path(root_dir)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    train_items = []
    val_items = []

    random.seed(seed)

    for cls in classes:
        files = sorted(glob(str(root / cls / "*.jpg"))) + \
                sorted(glob(str(root / cls / "*.png"))) + \
                sorted(glob(str(root / cls / "*.jpeg")))

        random.shuffle(files)
        n_train = int(len(files) * train_ratio)

        cls_idx = class_to_idx[cls]
        train_items.extend([(f, cls_idx) for f in files[:n_train]])
        val_items.extend([(f, cls_idx) for f in files[n_train:]])

    return train_items, val_items, class_to_idx

train_items, val_items, class_to_idx = build_split_lists(cfg.data_root, cfg.train_split)

print('train, val:')
print(len(train_items), len(val_items))

# -----------------------------------------------------------------

transform = T.Compose([
    T.Resize((cfg.image_size, cfg.image_size)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

class ImageNetLocal(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

train_dataset = ImageNetLocal(train_items, transform)
val_dataset   = ImageNetLocal(val_items,   transform)

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

# ----------------------------------------------------------------

import timm

sem_encoder = timm.create_model(cfg.sem_encoder_name, pretrained=True, num_classes=0)
sem_encoder.eval()
sem_encoder.to(device)

for p in sem_encoder.parameters():
    p.requires_grad = False

# ----------------------------------------------------------------

# Original model hyperparameters:
# model = UNet2DModel(
#     sample_size=cfg.image_size,   # 256
#     in_channels=3,
#     out_channels=3,
#     block_out_channels=(128, 256, 512, 512),  # deeper for 256x256
#     layers_per_block=2,
#     attention_head_dim=64
# ).to(device)

# small test hyperparameters:
model = UNet2DModel(
    sample_size=cfg.image_size,   # 256
    in_channels=3,
    out_channels=3,
    block_out_channels=(128, 64, 64),  # deeper for 256x256
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    layers_per_block=4,
    attention_head_dim=8
).to(device)

scheduler = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps)

optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

# ----------------------------------------------------------------

def predict_x0_from_eps(x_t, t, eps, scheduler):
    alpha_prod_t = scheduler.alphas_cumprod[t].to(x_t.device)
    sqrt_alpha = alpha_prod_t.sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = (1 - alpha_prod_t).sqrt().view(-1, 1, 1, 1)
    x0 = (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha
    return x0.clamp(-1, 1)

# ----------------------------------------------------------------

global_step = 0
loss_history = []

print('starting...')

model.train()
print('trained')
start_time = time.time()

while global_step < cfg.max_steps:
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    bs = imgs.size(0)

    # Timesteps
    t = torch.randint(0, cfg.num_train_timesteps, (bs,), device=device)

    print('step 1')

    # q(x_t | x_0)
    noise = torch.randn_like(imgs)
    alpha = scheduler.alphas_cumprod[t].view(bs, 1, 1, 1)
    x_t = imgs * alpha.sqrt() + noise * (1 - alpha).sqrt()

    eps_pred = model(x_t, t).sample
    loss_diff = F.mse_loss(eps_pred, noise)

    print('step 2')

    # ------- Semantic Loss -------
    # sem_ts = [0, 200, 500, 800]
    sem_ts = [0, 20, 50, 80]
    sem_losses = []

    with torch.no_grad():
        real_img = (imgs + 1) / 2
        feat_real = sem_encoder(real_img)
        if feat_real.ndim > 2:
            feat_real = torch.adaptive_avg_pool2d(feat_real, 1).flatten(1)

    print('step 3')

    for ts in sem_ts:
        print(f'for loop {ts}')
        ts_tensor = torch.full((bs,), ts, device=device)

        alpha_s = scheduler.alphas_cumprod[ts].to(device)
        noise_s = torch.randn_like(imgs)
        x_ts = imgs * alpha_s.sqrt() + noise_s * (1-alpha_s).sqrt()

        eps_s = model(x_ts, ts_tensor).sample
        x0_hat = predict_x0_from_eps(x_ts, ts_tensor, eps_s, scheduler)

        feat_hat = sem_encoder((x0_hat + 1) / 2)
        if feat_hat.ndim > 2:
            feat_hat = torch.adaptive_avg_pool2d(feat_hat, 1).flatten(1)

        cos = F.cosine_similarity(feat_real, feat_hat, dim=1)
        sem_losses.append((1 - cos).mean())

    print('step 4')

    loss_sem = torch.stack(sem_losses).mean()
    loss = loss_diff + cfg.lambda_sem * loss_sem

    print('step 5')

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('step 6')

    if global_step % cfg.log_interval == 0:
        elapsed = time.time() - start_time
        print(f"[{global_step}] loss={loss.item():.4f} diff={loss_diff.item():.4f} sem={loss_sem.item():.4f} time={elapsed:.1f}s")
        loss_history.append(loss.item())
        start_time = time.time()

    global_step += 1
    print(f'just did step {global_step}')
