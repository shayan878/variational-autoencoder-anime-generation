import torch
import torch.optim as optim
from tqdm import tqdm
from .model import vae_loss

def train_vae(model, dataloader, device, num_epochs=5, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_recon_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images in progress_bar:
            images = images.to(device)
            
            recon_images, mu, logvar = model(images)
            recon_loss = vae_loss(recon_images, images, mu, logvar)

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            progress_bar.set_postfix({'loss': recon_loss.item()})

        avg_loss = total_recon_loss / len(dataloader)
        recon_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Avg Reconstruction Loss: {avg_loss:.4f}")

    return recon_losses
