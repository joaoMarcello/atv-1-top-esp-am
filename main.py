import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import KFold
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model import AnyNet, CoralLoss
from dataset import EyePacsLoader


# Configurações globais
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5
N_EPOCHS = 50
K_FOLDS = 3
N_TRIALS = 50
RANDOM_SEED = 42
DATA_DIR = 'data/train'
CSV_FILE = 'data/trainLabels.csv'

# Setar seeds para reprodutibilidade
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


class EarlyStopping:
    """Early stopping para parar o treinamento quando a validação não melhorar"""
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): Quantas épocas esperar após a última melhoria
            verbose (bool): Se True, imprime mensagem para cada melhoria
            delta (float): Mínima mudança para considerar como melhoria
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss, model=None):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss
            self.counter = 0


def get_transforms(image_size=224):
    """Define as transformações de data augmentation"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device, head_type):
    """Treina o modelo por uma época"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calcular acurácia
        if head_type == "coral_head":
            predictions = model.head.predict(outputs)
        else:
            _, predictions = torch.max(outputs, 1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()
        
        # Atualizar barra de progresso
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, head_type):
    """Valida o modelo"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calcular acurácia
            if head_type == "coral_head":
                predictions = model.head.predict(outputs)
            else:
                _, predictions = torch.max(outputs, 1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            
            # Atualizar barra de progresso
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_fold(model, train_loader, val_loader, criterion, optimizer, device, 
               n_epochs, head_type, patience=7, verbose=False):
    """Treina um fold com early stopping"""
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)
    best_val_loss = np.Inf
    
    for epoch in range(n_epochs):
        if verbose:
            print(f'\nEpoch {epoch+1}/{n_epochs}')
        
        # Treinar
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, head_type)
        
        # Validar
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, 
                                          device, head_type)
        
        if verbose:
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        early_stopping(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if early_stopping.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_loss


def objective(trial):
    """Função objetivo para otimização com Optuna"""
    
    # Sugerir hiperparâmetros
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    stem_channels = trial.suggest_categorical('stem_channels', [16, 32, 64])
    block_type = trial.suggest_categorical('block_type', ['residual', 'se_attention', 'self_attention'])
    head_type = trial.suggest_categorical('head_type', ['normal_head', 'coral_head'])
    
    # Configurar transforms
    train_transform, val_transform = get_transforms()
    
    # Criar dataset completo
    full_dataset = EyePacsLoader(
        root_dir=DATA_DIR,
        csv_file=CSV_FILE,
        transform=train_transform
    )
    
    # Configurar K-Fold Cross Validation
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_losses = []
    
    # Iterar sobre os folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f'\nTrial {trial.number} | Fold {fold + 1}/{K_FOLDS}')
        print(f'Hyperparameters: lr={lr:.6f}, batch_size={batch_size}, '
              f'stem_channels={stem_channels}, block_type={block_type}, head_type={head_type}')
        
        # Criar samplers para train e validation
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        # Criar dataloaders
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # Para validação, usar transform sem augmentation
        val_dataset = EyePacsLoader(
            root_dir=DATA_DIR,
            csv_file=CSV_FILE,
            transform=val_transform
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # Criar modelo
        model = AnyNet(
            num_classes=NUM_CLASSES,
            stem_channels=stem_channels,
            stage_channels=[64, 128, 256, 512],
            stage_depths=[2, 2, 3, 2],
            groups=8,
            width_per_group=4,
            block_type=block_type,
            head_type=head_type,
            stem_kernel_size=3
        ).to(DEVICE)
        
        # Escolher loss apropriada baseada no head_type
        if head_type == "coral_head":
            criterion = CoralLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Criar otimizador
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        
        # Treinar fold
        fold_loss = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            n_epochs=N_EPOCHS,
            head_type=head_type,
            patience=7,
            verbose=False
        )
        
        fold_losses.append(fold_loss)
        
        # Limpar memória
        del model
        del optimizer
        del criterion
        torch.cuda.empty_cache()
    
    # Retornar média das losses dos folds
    avg_loss = np.mean(fold_losses)
    print(f'\nTrial {trial.number} | Average Loss: {avg_loss:.4f}')
    
    return avg_loss


def train_final_model(best_params, save_path='best_model.pth'):
    """Treina o modelo final com os melhores hiperparâmetros"""
    print("\n" + "="*80)
    print("TREINANDO MODELO FINAL COM MELHORES HIPERPARÂMETROS")
    print("="*80)
    print(f"\nMelhores hiperparâmetros encontrados:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Configurar transforms
    train_transform, val_transform = get_transforms()
    
    # Criar datasets
    train_dataset = EyePacsLoader(
        root_dir=DATA_DIR,
        csv_file=CSV_FILE,
        transform=train_transform
    )
    
    # Dividir em train/val (80/20)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Criar samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataset = EyePacsLoader(
        root_dir=DATA_DIR,
        csv_file=CSV_FILE,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=best_params['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Criar modelo
    model = AnyNet(
        num_classes=NUM_CLASSES,
        stem_channels=best_params['stem_channels'],
        stage_channels=[64, 128, 256, 512],
        stage_depths=[2, 2, 3, 2],
        groups=8,
        width_per_group=4,
        block_type=best_params['block_type'],
        head_type=best_params['head_type'],
        stem_kernel_size=3
    ).to(DEVICE)
    
    # Escolher loss apropriada
    if best_params['head_type'] == "coral_head":
        criterion = CoralLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Criar otimizador
    optimizer = optim.RMSprop(model.parameters(), lr=best_params['lr'])
    
    # Treinar modelo
    early_stopping = EarlyStopping(patience=10, verbose=True)
    best_val_loss = np.Inf
    
    print(f"\nIniciando treinamento do modelo final por {N_EPOCHS} épocas...")
    
    for epoch in range(N_EPOCHS):
        print(f'\n{"="*80}')
        print(f'Epoch {epoch+1}/{N_EPOCHS}')
        print(f'{"="*80}')
        
        # Treinar
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, 
            DEVICE, best_params['head_type']
        )
        
        # Validar
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, 
            DEVICE, best_params['head_type']
        )
        
        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'hyperparameters': best_params
            }, save_path)
            print(f'✓ Modelo salvo em {save_path}')
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*80}")
    print(f"TREINAMENTO FINALIZADO")
    print(f"{'='*80}")
    print(f"Melhor Val Loss: {best_val_loss:.4f}")
    print(f"Modelo salvo em: {save_path}")
    
    return model


def main():
    """Função principal"""
    print("="*80)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA")
    print("="*80)
    print(f"\nConfiguração:")
    print(f"  Device: {DEVICE}")
    print(f"  Número de classes: {NUM_CLASSES}")
    print(f"  Épocas por trial: {N_EPOCHS}")
    print(f"  K-Folds: {K_FOLDS}")
    print(f"  Número de trials: {N_TRIALS}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  CSV file: {CSV_FILE}")
    
    # Verificar se os arquivos existem
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {DATA_DIR}")
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {CSV_FILE}")
    
    # Criar study do Optuna
    print("\n" + "="*80)
    print("INICIANDO OTIMIZAÇÃO")
    print("="*80)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=RANDOM_SEED),
        study_name='anynet_optimization'
    )
    
    # Otimizar
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # Resultados
    print("\n" + "="*80)
    print("RESULTADOS DA OTIMIZAÇÃO")
    print("="*80)
    print(f"\nMelhor trial: {study.best_trial.number}")
    print(f"Melhor loss: {study.best_trial.value:.4f}")
    print(f"\nMelhores hiperparâmetros:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Salvar resultados do study
    study_results_path = 'optuna_study_results.txt'
    with open(study_results_path, 'w') as f:
        f.write("OTIMIZAÇÃO DE HIPERPARÂMETROS - RESULTADOS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Melhor trial: {study.best_trial.number}\n")
        f.write(f"Melhor loss: {study.best_trial.value:.4f}\n\n")
        f.write("Melhores hiperparâmetros:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "="*80 + "\n\n")
        f.write("Top 5 Trials:\n")
        for i, trial in enumerate(study.best_trials[:5], 1):
            f.write(f"\n{i}. Trial {trial.number} - Loss: {trial.value:.4f}\n")
            for key, value in trial.params.items():
                f.write(f"   {key}: {value}\n")
    
    print(f"\nResultados salvos em: {study_results_path}")
    
    # Treinar modelo final com melhores hiperparâmetros
    final_model = train_final_model(
        best_params=study.best_trial.params,
        save_path='best_anynet_model.pth'
    )
    
    print("\n" + "="*80)
    print("PROCESSO COMPLETO!")
    print("="*80)


if __name__ == '__main__':
    main()
