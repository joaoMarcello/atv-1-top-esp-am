import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Sampler
from PIL import Image


class SubsetWeightedRandomSampler(Sampler):
    """
    WeightedRandomSampler que funciona com subsets de índices.
    
    Diferente do WeightedRandomSampler padrão que assume índices 0 a len(dataset)-1,
    este sampler trabalha com um subset específico de índices globais.
    
    Args:
        indices (array-like): Índices globais do subset (ex: [0, 5, 10, 1360, ...])
        labels (array-like): Labels correspondentes aos índices do subset
        num_samples (int): Número de amostras por época
        replacement (bool): Se permite repetição (True para oversampling)
    """
    
    def __init__(self, indices, labels, num_samples, replacement=True):
        self.indices = np.array(indices)
        self.labels = np.array(labels)
        self.num_samples = num_samples
        self.replacement = replacement
        
        # Calcular pesos baseados nas labels do subset
        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[self.labels]
        
        # Normalizar pesos
        self.weights = sample_weights / sample_weights.sum()
    
    def __iter__(self):
        # Amostrar POSIÇÕES no subset (0 a len(indices)-1)
        sampled_positions = np.random.choice(
            len(self.indices),
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights
        )
        
        # Converter posições para índices globais
        sampled_indices = self.indices[sampled_positions]
        
        # Retornar como iterador
        return iter(sampled_indices.tolist())
    
    def __len__(self):
        return self.num_samples


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
    
    def get_class_weights(self):
        """
        Calcula pesos para balanceamento de classes usando WeightedRandomSampler.
        
        Returns:
            torch.Tensor: Pesos para cada amostra do dataset
        """
        # Obter todas as labels
        labels = self.data_frame[self.label_column].values
        
        # Contar frequência de cada classe
        class_counts = np.bincount(labels)
        
        # Calcular peso inverso da frequência
        # Classes raras recebem pesos maiores
        class_weights = 1.0 / class_counts
        
        # Atribuir peso a cada amostra baseado na sua classe
        sample_weights = class_weights[labels]
        
        return torch.DoubleTensor(sample_weights)
    
    def create_balanced_sampler(self, indices=None, oversampling_factor=1.0):
        """
        Cria um sampler balanceado para batches com proporção de classes equilibrada.
        
        Com oversampling_factor > 1.0:
        - Garante que a maioria das imagens sejam vistas pelo menos uma vez
        - Classes raras aparecem mais vezes (balanceamento desejável)
        - Época demora mais tempo (proporcional ao fator)
        
        Args:
            indices (array-like, optional): Subset de índices a usar (para K-Fold).
                                            Se None, usa todo o dataset.
            oversampling_factor (float): Fator de sobreamostragem (padrão 1.0 = sem oversampling)
                                         1.5 = +50% amostras, 2.0 = +100% amostras
        
        Returns:
            Sampler: SubsetWeightedRandomSampler (se indices fornecido) ou WeightedRandomSampler (dataset completo)
        """
        if indices is not None:
            # ✅ MODO SUBSET: Usar SubsetWeightedRandomSampler customizado
            # Converter para numpy array se necessário
            if not isinstance(indices, np.ndarray):
                indices = np.array(indices)
            
            # Obter labels do subset
            subset_labels = self.data_frame.iloc[indices][self.label_column].values
            
            # Número de amostras por época (com oversampling)
            num_samples = int(len(indices) * oversampling_factor)
            
            # Criar sampler customizado que retorna índices GLOBAIS
            sampler = SubsetWeightedRandomSampler(
                indices=indices,
                labels=subset_labels,
                num_samples=num_samples,
                replacement=True
            )
            
            return sampler
        
        else:
            # ✅ MODO DATASET COMPLETO: Usar WeightedRandomSampler padrão
            # Obter labels do dataset completo
            labels = self.data_frame[self.label_column].values
            
            # Contar frequência de cada classe
            class_counts = np.bincount(labels)
            
            # Calcular peso inverso da frequência
            class_weights = 1.0 / class_counts
            
            # Atribuir peso a cada amostra baseado na sua classe
            sample_weights = class_weights[labels]
            
            # Calcular número de amostras por época
            num_samples = int(len(labels) * oversampling_factor)
            
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=num_samples,
                replacement=True  # Permite repetição (sobreamostragem de classes raras)
            )
            
            return sampler
