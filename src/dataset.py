import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AnimeDataset(Dataset):
    def __init__(self, dataset_anime_dir, transform=None):
        self.dataset_anime_dir = dataset_anime_dir
        # Only load valid image files
        self.images = [f for f in os.listdir(dataset_anime_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Default VAE transformations
        self.default_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = os.path.join(self.dataset_anime_dir, self.images[i])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        else:
            img = self.default_transform(img)
            
        return img
