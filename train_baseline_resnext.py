"""
Script para treinar baseline ResNeXt pré-treinado (ImageNet21k)
Usa validação cruzada K=3 mas treina apenas o primeiro fold
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Importar timm para modelo pré-treinado
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("ERRO: timm não está disponível. Instale com: pip install timm")
    exit(1)

from dataset import EyePacsLoader


def get_args():
    """Parse argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description='Treinamento de Baseline ResNeXt pré-treinado',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configurações de treinamento
    parser.add_argument('--n_epochs', type=int, default=40,
                        help='Número de épocas de treinamento')
    parser.add_argument('--k_folds', type=int, default=3,
                        help='Número de folds para validação cruzada')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=6,
                        help='Paciência para early stopping')
    parser.add_argument('--min_epochs', type=int, default=20,
                        help='Mínimo de épocas antes de early stopping')
    parser.add_argument('--random_seed', type=int, default=24,
                        help='Seed para reprodutibilidade')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Número de workers para DataLoader')
    
    # Caminhos de dados
    parser.add_argument('--data_dir', type=str,
                        default='C:/Users/Public/Documents/DATASETS/diabetic-retinopathy-detection/train_processed_224',
                        help='Diretório contendo as imagens de treinamento')
    parser.add_argument('--csv_file', type=str, default='data/train_labels_v2.csv',
                        help='Arquivo CSV com os labels')
    parser.add_argument('--save_dir', type=str, default='baseline_results',
                        help='Diretório para salvar resultados')
    
    # Configurações do modelo
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Número de classes para classificação')
    parser.add_argument('--model_name', type=str, default='resnext50_32x4d',
                        help='Nome do modelo timm (ex: resnext50_32x4d, resnext101_32x8d)')
    parser.add_argument('--pretrained', type=str, default='imagenet',
                        help='Dataset de pré-treino (imagenet ou None)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device para treinamento')
    
    args = parser.parse_args()
    
    # Configurar device
    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    # Criar diretório de salvamento
    os.makedirs(args.save_dir, exist_ok=True)
    
    return args


def get_class_weights(csv_path, device):
    """Calcula pesos de classe balanceados"""
    df = pd.read_csv(csv_path)
    class_counts = df['level'].value_counts().sort_index().values
    classes = np.arange(len(class_counts))
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=np.repeat(classes, class_counts)
    )
    
    return torch.FloatTensor(class_weights).to(device)


def calculate_metrics(y_true, y_pred, num_classes=5):
    """Calcula métricas de avaliação"""
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
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sensitivities.append(sensitivity)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    # Métricas globais
    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    accuracy = np.mean(y_true == y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Kappa por classe
    kappa_per_class = []
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        kappa_class = cohen_kappa_score(y_true_binary, y_pred_binary)
        kappa_per_class.append(kappa_class)
    
    return {
        'sensitivity': avg_sensitivity,
        'specificity': avg_specificity,
        'f1_score': f1,
        'kappa': kappa,
        'accuracy': accuracy,
        'sensitivities_per_class': sensitivities,
        'specificities_per_class': specificities,
        'f1_per_class': f1_per_class,
        'kappa_per_class': kappa_per_class,
        'confusion_matrix': cm.tolist()
    }


def get_transforms(image_size=224):
    """Define transformações de data augmentation"""
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


class EarlyStopping:
    """Early stopping para parar o treinamento quando a métrica não melhorar"""
    
    def __init__(self, patience=10, verbose=False, delta=0.001, min_epochs=15, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = None
        self.delta = delta
        self.min_epochs = min_epochs
        self.current_epoch = 0
        self.mode = mode
        
        if mode not in ['min', 'max']:
            raise ValueError("mode deve ser 'min' ou 'max'")
    
    def __call__(self, metric_value, model=None):
        self.current_epoch += 1
        score = metric_value
        
        if self.best_score is None:
            self.best_score = score
            self.best_metric = metric_value
            if self.verbose:
                print(f'  → Epoch {self.current_epoch}: Métrica inicial = {metric_value:.4f}')
        else:
            # Verificar se houve melhoria significativa
            if self.mode == 'max':
                improved = score > (self.best_score + self.delta)
            else:
                improved = score < (self.best_score - self.delta)
            
            if improved:
                self.best_score = score
                self.best_metric = metric_value
                self.counter = 0
                if self.verbose:
                    print(f'  → Epoch {self.current_epoch}: Métrica melhorou para {metric_value:.4f}')
            else:
                # Só incrementar contador se já passou do mínimo de épocas
                if self.current_epoch >= self.min_epochs:
                    self.counter += 1
                    if self.verbose:
                        print(f'  → Epoch {self.current_epoch}: Sem melhoria. Contador: {self.counter}/{self.patience}')
                    
                    if self.counter >= self.patience:
                        self.early_stop = True
                        if self.verbose:
                            print(f'  → Early stopping ativado na época {self.current_epoch}')
                else:
                    if self.verbose:
                        print(f'  → Epoch {self.current_epoch}: Sem melhoria, mas ainda no período mínimo ({self.current_epoch}/{self.min_epochs})')


def train_epoch(model, dataloader, criterion, optimizer, device, num_classes=5):
    """Treina o modelo por uma época"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(outputs, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        running_loss += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes)
    metrics['loss'] = epoch_loss
    
    return metrics


def validate_epoch(model, dataloader, criterion, device, num_classes=5):
    """Valida o modelo"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes)
    metrics['loss'] = epoch_loss
    
    return metrics


def train_fold(model, train_loader, val_loader, criterion, optimizer, 
               device, n_epochs, num_classes, patience, min_epochs):
    """Treina um fold com early stopping"""
    early_stopping = EarlyStopping(patience=patience, verbose=True, 
                                   min_epochs=min_epochs, delta=0.001, mode='max')
    
    best_val_f1 = 0.0
    best_train_metrics = None
    best_val_metrics = None
    best_epoch = 0
    best_model_state = None
    
    # Histórico de treinamento
    history = {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'train_kappa': [],
        'train_sensitivity': [],
        'train_specificity': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_kappa': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'best_epoch': 0
    }
    
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        print('-' * 80)
        
        # Treinar
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        
        # Validar
        val_metrics = validate_epoch(model, val_loader, criterion, device, num_classes)
        
        # Salvar no histórico
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_score'])
        history['train_kappa'].append(train_metrics['kappa'])
        history['train_sensitivity'].append(train_metrics['sensitivity'])
        history['train_specificity'].append(train_metrics['specificity'])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['val_kappa'].append(val_metrics['kappa'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        
        # Imprimir métricas
        print(f'\nTrain → Loss: {train_metrics["loss"]:.4f} | Acc: {train_metrics["accuracy"]*100:.2f}% | '
              f'F1: {train_metrics["f1_score"]:.4f} | Kappa: {train_metrics["kappa"]:.4f}')
        print(f'        Sen: {train_metrics["sensitivity"]:.4f} | Spec: {train_metrics["specificity"]:.4f}')
        
        print(f'\nVal   → Loss: {val_metrics["loss"]:.4f} | Acc: {val_metrics["accuracy"]*100:.2f}% | '
              f'F1: {val_metrics["f1_score"]:.4f} | Kappa: {val_metrics["kappa"]:.4f}')
        print(f'        Sen: {val_metrics["sensitivity"]:.4f} | Spec: {val_metrics["specificity"]:.4f}')
        
        # Early stopping
        early_stopping(val_metrics['f1_score'])
        
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            print(f'\n  ✓ Melhor modelo atualizado! F1 = {best_val_f1:.4f}')
        
        if early_stopping.early_stop:
            print(f'\n⚠ Early stopping ativado na época {epoch+1}')
            print(f'  Melhor época foi: {best_epoch} (F1 = {best_val_f1:.4f})')
            break
    
    history['best_epoch'] = best_epoch
    
    return best_val_f1, best_train_metrics, best_val_metrics, best_epoch, best_model_state, history


def main():
    """Função principal"""
    args = get_args()
    
    # Setar seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    print("=" * 80)
    print("TREINAMENTO DE BASELINE: RESNEXT PRÉ-TREINADO (IMAGENET)")
    print("=" * 80)
    print(f"\nConfiguração:")
    print(f"  Device: {args.device}")
    print(f"  Modelo: {args.model_name}")
    print(f"  Pré-treinamento: {args.pretrained}")
    print(f"  Número de classes: {args.num_classes}")
    print(f"  Épocas: {args.n_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  K-Folds: {args.k_folds} (treinar apenas Fold 1)")
    print(f"  Paciência: {args.patience}")
    print(f"  Min epochs: {args.min_epochs}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  CSV file: {args.csv_file}")
    print(f"  Save directory: {args.save_dir}")
    
    # Criar modelo
    print(f"\n{'='*80}")
    print("CRIANDO MODELO")
    print(f"{'='*80}")
    
    # Usar pré-treinamento se disponível
    pretrained = args.pretrained != 'None'
    
    model = timm.create_model(
        args.model_name,
        pretrained=pretrained,
        num_classes=args.num_classes
    )
    model = model.to(args.device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModelo criado: {args.model_name}")
    print(f"  Total de parâmetros: {total_params:,}")
    print(f"  Parâmetros treináveis: {trainable_params:,}")
    print(f"  Pré-treinado: {'Sim' if pretrained else 'Não'}")
    
    # Preparar dados
    print(f"\n{'='*80}")
    print("PREPARANDO DADOS")
    print(f"{'='*80}")
    
    train_transform, val_transform = get_transforms()
    
    full_dataset = EyePacsLoader(
        root_dir=args.data_dir,
        csv_file=args.csv_file,
        transform=train_transform
    )
    
    # Obter labels do CSV
    df_labels = pd.read_csv(args.csv_file)
    all_labels = df_labels['level'].values
    
    print(f"\nDataset carregado:")
    print(f"  Total de amostras: {len(full_dataset)}")
    print(f"  Distribuição de classes:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"    Classe {cls}: {count} amostras ({count/len(all_labels)*100:.2f}%)")
    
    # Calcular pesos de classe
    class_weights = get_class_weights(args.csv_file, args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"\nPesos de classe (balanceados):")
    for i, w in enumerate(class_weights.cpu().numpy()):
        print(f"  Classe {i}: {w:.4f}")
    
    # Configurar K-Fold (mas treinar apenas o primeiro)
    kfold = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
    
    print(f"\n{'='*80}")
    print(f"TREINANDO FOLD 1/{args.k_folds}")
    print(f"{'='*80}")
    
    # Pegar apenas o primeiro fold
    for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(np.zeros(len(all_labels)), all_labels)):
        if fold_idx > 0:
            break  # Treinar apenas o primeiro fold
        
        print(f"\nFold 1:")
        print(f"  Amostras de treino: {len(train_ids)}")
        print(f"  Amostras de validação: {len(val_ids)}")
        
        # Criar datasets do fold
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_dataset = EyePacsLoader(
            root_dir=args.data_dir,
            csv_file=args.csv_file,
            transform=val_transform
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Criar otimizador
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Treinar fold
        print(f"\nIniciando treinamento...")
        
        best_f1, best_train_metrics, best_val_metrics, best_epoch, best_model_state, history = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=args.device,
            n_epochs=args.n_epochs,
            num_classes=args.num_classes,
            patience=args.patience,
            min_epochs=args.min_epochs
        )
        
        print(f"\n{'='*80}")
        print("TREINAMENTO CONCLUÍDO")
        print(f"{'='*80}")
        print(f"\nMelhores resultados (Época {best_epoch}):")
        print(f"\n  VALIDAÇÃO:")
        print(f"    F1-score: {best_val_metrics['f1_score']:.4f}")
        print(f"    Accuracy: {best_val_metrics['accuracy']*100:.2f}%")
        print(f"    Kappa: {best_val_metrics['kappa']:.4f}")
        print(f"    Sensitivity: {best_val_metrics['sensitivity']:.4f}")
        print(f"    Specificity: {best_val_metrics['specificity']:.4f}")
        print(f"    Loss: {best_val_metrics['loss']:.4f}")
        
        print(f"\n  TREINO:")
        print(f"    F1-score: {best_train_metrics['f1_score']:.4f}")
        print(f"    Accuracy: {best_train_metrics['accuracy']*100:.2f}%")
        print(f"    Kappa: {best_train_metrics['kappa']:.4f}")
        print(f"    Sensitivity: {best_train_metrics['sensitivity']:.4f}")
        print(f"    Specificity: {best_train_metrics['specificity']:.4f}")
        print(f"    Loss: {best_train_metrics['loss']:.4f}")
        
        # Salvar modelo
        model_path = os.path.join(args.save_dir, 'baseline_model.pth')
        torch.save(best_model_state, model_path)
        print(f"\n✓ Modelo salvo: {model_path}")
        
        # Salvar histórico
        history_path = os.path.join(args.save_dir, 'baseline_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        print(f"✓ Histórico salvo: {history_path}")
        
        # Salvar configuração
        config = {
            'model_name': args.model_name,
            'pretrained': args.pretrained,
            'num_classes': args.num_classes,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'patience': args.patience,
            'min_epochs': args.min_epochs,
            'best_epoch': best_epoch,
            'best_f1': best_f1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_path = os.path.join(args.save_dir, 'baseline_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Configuração salva: {config_path}")
        
        # Salvar resultados em TXT
        results_path = os.path.join(args.save_dir, 'baseline_results.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RESULTADOS DO BASELINE: RESNEXT PRÉ-TREINADO\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Data/Hora: {config['timestamp']}\n")
            f.write(f"Modelo: {args.model_name}\n")
            f.write(f"Pré-treinamento: {args.pretrained}\n")
            f.write(f"Device: {args.device}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("HIPERPARÂMETROS\n")
            f.write("-"*80 + "\n\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"Weight decay: {args.weight_decay}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Número de épocas: {args.n_epochs}\n")
            f.write(f"Paciência (early stopping): {args.patience}\n")
            f.write(f"Mínimo de épocas: {args.min_epochs}\n")
            f.write(f"K-Folds: {args.k_folds} (treinado apenas Fold 1)\n\n")
            
            f.write("-"*80 + "\n")
            f.write("RESULTADOS DO TREINAMENTO\n")
            f.write("-"*80 + "\n\n")
            f.write(f"Melhor época: {best_epoch}\n")
            f.write(f"Épocas totais executadas: {len(history['epochs'])}\n\n")
            
            f.write("MÉTRICAS DE VALIDAÇÃO (MELHOR ÉPOCA):\n")
            f.write("┌─────────────────────┬──────────┐\n")
            f.write("│      Métrica        │  Valor   │\n")
            f.write("├─────────────────────┼──────────┤\n")
            f.write(f"│ F1-score (macro)    │  {best_val_metrics['f1_score']:6.4f}  │\n")
            f.write(f"│ Accuracy            │  {best_val_metrics['accuracy']*100:5.2f}%  │\n")
            f.write(f"│ Kappa               │  {best_val_metrics['kappa']:6.4f}  │\n")
            f.write(f"│ Sensitivity (macro) │  {best_val_metrics['sensitivity']:6.4f}  │\n")
            f.write(f"│ Specificity (macro) │  {best_val_metrics['specificity']:6.4f}  │\n")
            f.write(f"│ Loss                │  {best_val_metrics['loss']:6.4f}  │\n")
            f.write("└─────────────────────┴──────────┘\n\n")
            
            f.write("MÉTRICAS POR CLASSE (VALIDAÇÃO):\n")
            f.write("┌───────┬────────────┬──────────────┬──────────────┬─────────────┐\n")
            f.write("│ Classe│  F1-score  │ Sensitivity  │ Specificity  │    Kappa    │\n")
            f.write("├───────┼────────────┼──────────────┼──────────────┼─────────────┤\n")
            for i in range(args.num_classes):
                f.write(f"│   {i}   │   {best_val_metrics['f1_per_class'][i]:6.4f}   │")
                f.write(f"    {best_val_metrics['sensitivities_per_class'][i]:6.4f}    │")
                f.write(f"    {best_val_metrics['specificities_per_class'][i]:6.4f}    │")
                f.write(f"   {best_val_metrics['kappa_per_class'][i]:6.4f}    │\n")
            f.write("└───────┴────────────┴──────────────┴──────────────┴─────────────┘\n\n")
            
            f.write("\nMÉTRICAS DE TREINO (MELHOR ÉPOCA):\n")
            f.write("┌─────────────────────┬──────────┐\n")
            f.write("│      Métrica        │  Valor   │\n")
            f.write("├─────────────────────┼──────────┤\n")
            f.write(f"│ F1-score (macro)    │  {best_train_metrics['f1_score']:6.4f}  │\n")
            f.write(f"│ Accuracy            │  {best_train_metrics['accuracy']*100:5.2f}%  │\n")
            f.write(f"│ Kappa               │  {best_train_metrics['kappa']:6.4f}  │\n")
            f.write(f"│ Sensitivity (macro) │  {best_train_metrics['sensitivity']:6.4f}  │\n")
            f.write(f"│ Specificity (macro) │  {best_train_metrics['specificity']:6.4f}  │\n")
            f.write(f"│ Loss                │  {best_train_metrics['loss']:6.4f}  │\n")
            f.write("└─────────────────────┴──────────┘\n\n")
            
            f.write("-"*80 + "\n")
            f.write("MATRIZ DE CONFUSÃO (VALIDAÇÃO)\n")
            f.write("-"*80 + "\n\n")
            cm = np.array(best_val_metrics['confusion_matrix'])
            f.write("Predito →\n")
            f.write("Real ↓   ")
            for i in range(args.num_classes):
                f.write(f"  {i:5}")
            f.write("\n")
            for i in range(args.num_classes):
                f.write(f"   {i}     ")
                for j in range(args.num_classes):
                    f.write(f"{cm[i][j]:5}  ")
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Resultados salvos: {results_path}")
        
        print(f"\n{'='*80}")
        print("PROCESSO COMPLETO!")
        print(f"{'='*80}")
        print(f"\nArquivos salvos em: {args.save_dir}/")
        print(f"  - baseline_model.pth (modelo treinado)")
        print(f"  - baseline_history.pkl (histórico de treinamento)")
        print(f"  - baseline_config.json (configuração)")
        print(f"  - baseline_results.txt (resultados detalhados)")
        
        break  # Garantir que só treina o primeiro fold


if __name__ == '__main__':
    main()
