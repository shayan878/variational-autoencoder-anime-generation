import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import AnimeDataset
from src.model import VAE
from src.train import train_vae

def setup_kaggle_env():
    # Make sure you set your Kaggle credentials before downloading
    # os.environ['KAGGLE_USERNAME'] = "your_username"
    # os.environ['KAGGLE_KEY'] = "your_key"
    pass

def download_datasets():
    print("Checking datasets...")
    if not os.path.exists('./animefacedataset'):
        print("Downloading animefacedataset...")
        os.system('kaggle datasets download -d splcher/animefacedataset')
        os.system('unzip -q animefacedataset.zip -d animefacedataset')
    else:
        print("Anime dataset already exists.")

def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Reconstruction Loss for VAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.png")
    print("Loss plot saved as 'loss_plot.png'")

def main():
    # 1. Setup Environment and Data
    setup_kaggle_env()
    download_datasets()
    
    # 2. Configurations
    dataset_anime_dir = './animefacedataset/images/'
    batch_size = 64
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Data Loading
    print("Loading dataset...")
    dataset_anime = AnimeDataset(dataset_anime_dir)
    dataloader_anime = DataLoader(dataset_anime, batch_size=batch_size, shuffle=True)

    # 4. Model Initialization
    print("Initializing VAE model...")
    model_vae = VAE().to(device)

    # 5. Training
    print("Starting training...")
    losses = train_vae(model_vae, dataloader_anime, device, num_epochs=num_epochs)

    # 6. Visualization
    plot_losses(losses)

if __name__ == "__main__":
    main()
