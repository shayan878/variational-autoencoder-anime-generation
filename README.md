# VAE Anime Face Generation

A PyTorch implementation of a Variational Autoencoder (VAE) trained on the Anime Face Dataset to encode and reconstruct anime faces.

## Project Structure
* `src/dataset.py`: PyTorch Dataset class (`AnimeDataset`) for loading and transforming images.
* `src/model.py`: VAE network architecture (Encoder, Decoder) and the custom `vae_loss` function.
* `src/train.py`: Training loop utilizing `tqdm` for progress tracking.
* `main.py`: The entry point that handles Kaggle dataset downloads, data loading, training, and plotting loss curves.

## Setup & Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Kaggle Credentials**:
The pipeline automatically downloads the dataset via the Kaggle API. You must configure your Kaggle credentials.
Either set the environment variables in your terminal:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```
Or place your `kaggle.json` file in `~/.kaggle/kaggle.json`.

## Usage

Execute the main pipeline:
```bash
python main.py
```

**What it does:**
1. Downloads and unzips the `animefacedataset`.
2. Initializes the PyTorch DataLoader and custom VAE architecture.
3. Trains the VAE for 5 epochs on your GPU (if available) or CPU.
4. Generates a line plot of the reconstruction loss over time (`loss_plot.png`).
