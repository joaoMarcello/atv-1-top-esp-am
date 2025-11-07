import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class EyePacsLoader(Dataset):
    """
    Dataset loader for EyePACS (Eye Picture Archive Communication System).
    """
    
    def __init__(self, root_dir, csv_file, transform=None, label_column='level'):
        """
        Initialize the EyePacsLoader dataset.
        
        Args:
            root_dir (str): Root directory containing the images
            csv_file (str): Path to the CSV file with labels
            transform (callable, optional): Optional transform to be applied on images
            label_column (str): Name of the column containing the labels (default: 'level')
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.label_column = label_column
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
            idx = idx.item()  # Converter tensor para int
        
        # Get image name and label from dataframe using column names
        row = self.data_frame.iloc[idx]
        img_name = row.iloc[0]  # First column is always the image name
        label = int(row[self.label_column])  # Get label from specified column and convert to int
        
        # Construct full image path (assuming .jpeg extension)
        img_path = os.path.join(self.root_dir, f"{img_name}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Label já é int, converter diretamente para tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
