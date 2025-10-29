import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class EyePacsLoader(Dataset):
    """
    Dataset loader for EyePACS (Eye Picture Archive Communication System).
    """
    
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Initialize the EyePacsLoader dataset.
        
        Args:
            root_dir (str): Root directory containing the images
            csv_file (str): Path to the CSV file with labels
            transform (callable, optional): Optional transform to be applied on images
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = pd.read_csv(csv_file)
        
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, label) pair
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name and label from dataframe
        img_name = self.data_frame.iloc[idx, 0]
        label = self.data_frame.iloc[idx, 1]
        
        # Construct full image path (assuming .jpeg extension)
        img_path = os.path.join(self.root_dir, f"{img_name}.jpeg")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label
