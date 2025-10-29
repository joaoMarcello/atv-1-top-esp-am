import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
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
NUM_WORKERS = 2
DATA_DIR = 'C:/Users/Public/Documents/DATASETS/diabetic-retinopathy-detection/train'
CSV_FILE = 'data/trainLabels.csv'
BEST_MODEL_SAVE_DIR = 'best_model_data'

# Setar seeds para reprodutibilidade
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Criar diretório para salvar melhor modelo
os.makedirs(BEST_MODEL_SAVE_DIR, exist_ok=True)


class EarlyStopping:
    """Early stopping para parar o treinamento quando o F1-score não melhorar"""
    
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
        self.best_f1 = 0.0
        self.delta = delta
    
    def __call__(self, f1_score, model=None):
        score = f1_score
        
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f'F1-score improved ({self.best_f1:.6f} --> {f1_score:.6f})')
            self.best_f1 = f1_score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print(f'F1-score improved ({self.best_f1:.6f} --> {f1_score:.6f})')
            self.best_f1 = f1_score
            self.counter = 0


def calculate_metrics(y_true, y_pred, num_classes=5):
    """
    Calcula métricas de avaliação: Sensibilidade, Especificidade, F1-score e Kappa
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        num_classes: Número de classes
    
    Returns:
        dict com as métricas calculadas
    """
    # Converter para numpy se for tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Calcular sensibilidade e especificidade por classe
    sensitivities = []
    specificities = []
    
    for i in range(num_classes):
        # True Positives, False Negatives, False Positives, True Negatives
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        # Sensibilidade (Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sensitivities.append(sensitivity)
        
        # Especificidade
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    # Média das sensibilidades e especificidades (macro average)
    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    
    # F1-score (macro average)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    return {
        'sensitivity': avg_sensitivity,
        'specificity': avg_specificity,
        'f1_score': f1,
        'kappa': kappa,
        'sensitivities_per_class': sensitivities,
        'specificities_per_class': specificities
    }


def get_transforms(image_size=224):
    """Define as transformações de data augmentation"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-180, 180)),
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


def train_epoch(model, dataloader, criterion, optimizer, device, head_type, num_classes=5):
    """Treina o modelo por uma época"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
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
        
        # Calcular predições
        if head_type == "coral_head":
            predictions = model.head.predict(outputs)
        else:
            _, predictions = torch.max(outputs, 1)
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        running_loss += loss.item()
        
        # Atualizar barra de progresso
        acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        pbar.set_postfix({'loss': loss.item(), 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    
    # Calcular métricas
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes)
    metrics['loss'] = epoch_loss
    metrics['accuracy'] = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    
    return metrics


def validate_epoch(model, dataloader, criterion, device, head_type, num_classes=5):
    """Valida o modelo"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calcular predições
            if head_type == "coral_head":
                predictions = model.head.predict(outputs)
            else:
                _, predictions = torch.max(outputs, 1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item()
            
            # Atualizar barra de progresso
            acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
            pbar.set_postfix({'loss': loss.item(), 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    
    # Calcular métricas
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes)
    metrics['loss'] = epoch_loss
    metrics['accuracy'] = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    
    return metrics


def train_fold(model, train_loader, val_loader, criterion, optimizer, device, 
               n_epochs, head_type, num_classes=5, patience=7, verbose=False):
    """Treina um fold com early stopping baseado em F1-score"""
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)
    best_val_f1 = 0.0
    best_train_metrics = None
    best_val_metrics = None
    best_epoch = 0
    
    for epoch in range(n_epochs):
        if verbose:
            print(f'\nEpoch {epoch+1}/{n_epochs}')
        
        # Treinar
        train_metrics = train_epoch(model, train_loader, criterion, 
                                    optimizer, device, head_type, num_classes)
        
        # Validar
        val_metrics = validate_epoch(model, val_loader, criterion, 
                                     device, head_type, num_classes)
        
        if verbose:
            print(f'Train - Loss: {train_metrics["loss"]:.4f} | Acc: {train_metrics["accuracy"]:.2f}% | '
                  f'F1: {train_metrics["f1_score"]:.4f} | Kappa: {train_metrics["kappa"]:.4f}')
            print(f'Val   - Loss: {val_metrics["loss"]:.4f} | Acc: {val_metrics["accuracy"]:.2f}% | '
                  f'F1: {val_metrics["f1_score"]:.4f} | Kappa: {val_metrics["kappa"]:.4f}')
            print(f'Val   - Sen: {val_metrics["sensitivity"]:.4f} | Spec: {val_metrics["specificity"]:.4f}')
        
        # Early stopping baseado em F1-score
        early_stopping(val_metrics['f1_score'])
        
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_train_metrics = train_metrics.copy()
            best_val_metrics = val_metrics.copy()
            best_epoch = epoch + 1
        
        if early_stopping.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_f1, best_train_metrics, best_val_metrics, best_epoch


def objective(trial, best_f1_tracker):
    """Função objetivo para otimização com Optuna"""
    
    # Sugerir hiperparâmetros
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    stem_channels = trial.suggest_categorical('stem_channels', [16, 32])
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
    fold_results = []  # Armazenar resultados detalhados de cada fold
    
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
            num_workers=NUM_WORKERS,
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
            num_workers=NUM_WORKERS,
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
        fold_f1, fold_train_metrics, fold_val_metrics, fold_best_epoch = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            n_epochs=N_EPOCHS,
            head_type=head_type,
            num_classes=NUM_CLASSES,
            patience=10,  # Maior paciência para F1-score (mais volátil que loss)
            verbose=False
        )
        
        fold_losses.append(fold_f1)
        
        # Armazenar resultados do fold
        fold_results.append({
            'fold': fold + 1,
            'best_epoch': fold_best_epoch,
            'train_metrics': fold_train_metrics,
            'val_metrics': fold_val_metrics
        })
        
        # Limpar memória
        del model
        del optimizer
        del criterion
        torch.cuda.empty_cache()
    
    # Retornar média dos F1-scores dos folds (queremos maximizar)
    avg_f1 = np.mean(fold_losses)
    print(f'\nTrial {trial.number} | Average F1-score: {avg_f1:.4f}')
    
    # Se este é o melhor trial até agora, salvar o modelo e configurações
    if avg_f1 > best_f1_tracker['best_f1']:
        best_f1_tracker['best_f1'] = avg_f1
        best_f1_tracker['best_trial'] = trial.number
        best_f1_tracker['best_params'] = {
            'lr': lr,
            'batch_size': batch_size,
            'stem_channels': stem_channels,
            'block_type': block_type,
            'head_type': head_type
        }
        best_f1_tracker['fold_results'] = fold_results
        
        # Salvar configuração e resultados
        import json
        config_path = os.path.join(BEST_MODEL_SAVE_DIR, 'best_model_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'trial_number': trial.number,
                'avg_f1_score': float(avg_f1),
                'hyperparameters': best_f1_tracker['best_params'],
                'fold_results': [
                    {
                        'fold': fr['fold'],
                        'best_epoch': fr['best_epoch'],
                        'train_metrics': {k: float(v) if not isinstance(v, list) else v 
                                         for k, v in fr['train_metrics'].items()},
                        'val_metrics': {k: float(v) if not isinstance(v, list) else v 
                                       for k, v in fr['val_metrics'].items()}
                    }
                    for fr in fold_results
                ],
                'model_architecture': {
                    'num_classes': NUM_CLASSES,
                    'stem_channels': stem_channels,
                    'stage_channels': [64, 128, 256, 512],
                    'stage_depths': [2, 2, 3, 2],
                    'groups': 8,
                    'width_per_group': 4,
                    'block_type': block_type,
                    'head_type': head_type,
                    'stem_kernel_size': 3
                }
            }, f, indent=2)
        
        print(f'\n>>> Novo melhor modelo encontrado! F1: {avg_f1:.4f}')
        print(f'   Configuração salva em: {config_path}')
    
    return avg_f1


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
        num_workers=NUM_WORKERS,
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
        num_workers=NUM_WORKERS,
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
    early_stopping = EarlyStopping(patience=15, verbose=True)  # Paciência maior no modelo final
    best_val_f1 = 0.0
    best_metrics = None
    
    print(f"\nIniciando treinamento do modelo final por {N_EPOCHS} épocas...")
    
    for epoch in range(N_EPOCHS):
        print(f'\n{"="*80}')
        print(f'Epoch {epoch+1}/{N_EPOCHS}')
        print(f'{"="*80}')
        
        # Treinar
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, 
            DEVICE, best_params['head_type'], NUM_CLASSES
        )
        
        # Validar
        val_metrics = validate_epoch(
            model, val_loader, criterion, 
            DEVICE, best_params['head_type'], NUM_CLASSES
        )
        
        print(f'\nTrain - Loss: {train_metrics["loss"]:.4f} | Acc: {train_metrics["accuracy"]:.2f}% | '
              f'F1: {train_metrics["f1_score"]:.4f} | Kappa: {train_metrics["kappa"]:.4f}')
        print(f'        Sen: {train_metrics["sensitivity"]:.4f} | Spec: {train_metrics["specificity"]:.4f}')
        print(f'\nVal   - Loss: {val_metrics["loss"]:.4f} | Acc: {val_metrics["accuracy"]:.2f}% | '
              f'F1: {val_metrics["f1_score"]:.4f} | Kappa: {val_metrics["kappa"]:.4f}')
        print(f'        Sen: {val_metrics["sensitivity"]:.4f} | Spec: {val_metrics["specificity"]:.4f}')
        
        # Salvar melhor modelo baseado em F1-score
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_metrics = val_metrics.copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'hyperparameters': best_params
            }, save_path)
            print(f'✓ Modelo salvo em {save_path} (F1: {best_val_f1:.4f})')
        
        # Early stopping baseado em F1-score
        early_stopping(val_metrics['f1_score'])
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*80}")
    print(f"TREINAMENTO FINALIZADO")
    print(f"{'='*80}")
    print(f"\nMelhores métricas de validação:")
    print(f"  F1-score: {best_metrics['f1_score']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.2f}%")
    print(f"  Kappa: {best_metrics['kappa']:.4f}")
    print(f"  Sensitivity: {best_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {best_metrics['specificity']:.4f}")
    print(f"  Loss: {best_metrics['loss']:.4f}")
    print(f"\nModelo salvo em: {save_path}")
    
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
    print(f"  Num workers: {NUM_WORKERS}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  CSV file: {CSV_FILE}")
    print(f"  Best model save directory: {BEST_MODEL_SAVE_DIR}")
    
    # Verificar se os arquivos existem
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {DATA_DIR}")
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {CSV_FILE}")
    
    # Tracker para melhor F1-score
    best_f1_tracker = {
        'best_f1': 0.0,
        'best_trial': None,
        'best_params': None,
        'fold_results': None
    }
    
    # Criar study do Optuna
    print("\n" + "="*80)
    print("INICIANDO OTIMIZAÇÃO")
    print("="*80)
    
    study = optuna.create_study(
        direction='maximize',  # Maximizar F1-score
        sampler=TPESampler(seed=RANDOM_SEED),
        study_name='anynet_optimization'
    )
    
    # Otimizar (passar best_f1_tracker para objective)
    study.optimize(lambda trial: objective(trial, best_f1_tracker), 
                   n_trials=N_TRIALS, show_progress_bar=True)
    
    # Resultados
    print("\n" + "="*80)
    print("RESULTADOS DA OTIMIZAÇÃO")
    print("="*80)
    print(f"\nMelhor trial: {study.best_trial.number}")
    print(f"Melhor F1-score: {study.best_trial.value:.4f}")
    print(f"\nMelhores hiperparâmetros:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Salvar resultados do study
    study_results_path = 'optuna_study_results.txt'
    with open(study_results_path, 'w') as f:
        f.write("OTIMIZAÇÃO DE HIPERPARÂMETROS - RESULTADOS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Melhor trial: {study.best_trial.number}\n")
        f.write(f"Melhor F1-score: {study.best_trial.value:.4f}\n\n")
        f.write("Melhores hiperparâmetros:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "="*80 + "\n\n")
        f.write("Top 5 Trials:\n")
        for i, trial in enumerate(study.best_trials[:5], 1):
            f.write(f"\n{i}. Trial {trial.number} - F1-score: {trial.value:.4f}\n")
            for key, value in trial.params.items():
                f.write(f"   {key}: {value}\n")
    
    print(f"\nResultados salvos em: {study_results_path}")
    
    # Treinar modelo final com melhores hiperparâmetros
    final_model_path = os.path.join(BEST_MODEL_SAVE_DIR, 'best_model.pth')
    final_model = train_final_model(
        best_params=study.best_trial.params,
        save_path=final_model_path
    )
    
    print("\n" + "="*80)
    print("PROCESSO COMPLETO!")
    print("="*80)


if __name__ == '__main__':
    main()
