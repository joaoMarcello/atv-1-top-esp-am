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
import pickle
import argparse
warnings.filterwarnings('ignore')

from model import AnyNet, CoralLoss
from dataset import EyePacsLoader


def get_args():
    """
    Configura e processa argumentos de linha de comando
    
    Returns:
        argparse.Namespace: Objeto contendo todos os argumentos configurados
    """
    parser = argparse.ArgumentParser(
        description='Treinamento de AnyNet para Classificação de Retinopatia Diabética',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configurações de treinamento
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Número de épocas por trial')
    parser.add_argument('--k_folds', type=int, default=3,
                        help='Número de folds para validação cruzada')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Número de trials para otimização Optuna')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Seed para reprodutibilidade')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Número de workers para DataLoader')
    parser.add_argument('--save_study_every', type=int, default=2,
                        help='Salvar study do Optuna a cada N trials')
    
    # Caminhos de dados
    parser.add_argument('--data_dir', type=str,
                        default='C:/Users/Public/Documents/DATASETS/diabetic-retinopathy-detection/train',
                        help='Diretório contendo as imagens de treinamento')
    parser.add_argument('--csv_file', type=str, default='data/trainLabels.csv',
                        help='Arquivo CSV com os labels')
    parser.add_argument('--save_dir', type=str, default='best_model_data',
                        help='Diretório para salvar modelos e resultados')
    
    # Configurações do modelo
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Número de classes para classificação')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device para treinamento (auto detecta CUDA)')
    
    args = parser.parse_args()
    
    # Configurar device
    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    # Criar diretório de salvamento se não existir
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Configurar path do pickle do Optuna
    args.optuna_study_pkl = os.path.join(args.save_dir, 'optuna_study.pkl')
    
    return args




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
    
    # F1-score (macro average e por classe)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    # Acurácia geral (mantém escala 0-1 como as outras métricas)
    accuracy = np.mean(y_true == y_pred)
    
    # Cohen's Kappa (geral e por classe usando matriz de confusão)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Kappa por classe (calculado como kappa binário: classe i vs resto)
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
        'kappa_per_class': kappa_per_class
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
    
    # Calcular métricas (accuracy em escala 0-1)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes)
    metrics['loss'] = epoch_loss
    
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
    
    # Calcular métricas (accuracy em escala 0-1)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), num_classes)
    metrics['loss'] = epoch_loss
    
    return metrics


def train_fold(model, train_loader, val_loader, criterion, optimizer, device, 
               n_epochs, head_type, num_classes=5, patience=7, verbose=False):
    """Treina um fold com early stopping baseado em F1-score e retorna histórico completo"""
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)
    best_val_f1 = 0.0
    best_train_metrics = None
    best_val_metrics = None
    best_epoch = 0
    best_model_state = None
    
    # Histórico completo de todas as épocas
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
        if verbose:
            print(f'\nEpoch {epoch+1}/{n_epochs}')
        
        # Treinar
        train_metrics = train_epoch(model, train_loader, criterion, 
                                    optimizer, device, head_type, num_classes)
        
        # Validar
        val_metrics = validate_epoch(model, val_loader, criterion, 
                                     device, head_type, num_classes)
        
        # Salvar métricas no histórico
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
        
        if verbose:
            print(f'Train - Loss: {train_metrics["loss"]:.4f} | Acc: {train_metrics["accuracy"]*100:.2f}% | '
                  f'F1: {train_metrics["f1_score"]:.4f} | Kappa: {train_metrics["kappa"]:.4f}')
            print(f'Val   - Loss: {val_metrics["loss"]:.4f} | Acc: {val_metrics["accuracy"]*100:.2f}% | '
                  f'F1: {val_metrics["f1_score"]:.4f} | Kappa: {val_metrics["kappa"]:.4f}')
            print(f'Val   - Sen: {val_metrics["sensitivity"]:.4f} | Spec: {val_metrics["specificity"]:.4f}')
        
        # Early stopping baseado em F1-score
        early_stopping(val_metrics['f1_score'])
        
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_train_metrics = train_metrics.copy()
            best_val_metrics = val_metrics.copy()
            best_epoch = epoch + 1
            history['best_epoch'] = epoch + 1
            # Salvar estado do modelo (deep copy para evitar referências)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if early_stopping.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_f1, best_train_metrics, best_val_metrics, best_epoch, best_model_state, history


def objective(trial, best_f1_tracker, args):
    """Função objetivo para otimização com Optuna"""
    
    # Sugerir hiperparâmetros
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    stem_channels = trial.suggest_categorical('stem_channels', [16, 32])
    block_type = trial.suggest_categorical('block_type', ['residual', 'se_attention', 'self_attention'])
    head_type = trial.suggest_categorical('head_type', ['normal_head', 'coral_head'])
    
    # Otimizar profundidade da rede (stage_depths)
    depth_config = trial.suggest_categorical('depth_config', [
        'shallow',      # Redes rasas, mais rápidas
        'balanced',     # Configuração padrão
        'deep',         # Redes profundas
        'very_deep',    # Muito profundas
        'front_heavy',  # Mais blocos nos primeiros stages
        'back_heavy'    # Mais blocos nos últimos stages
    ])
    
    depth_configs = {
        'shallow':     [1, 2, 2, 1],   # Total: 6 blocos
        'balanced':    [2, 2, 3, 2],   # Total: 9 blocos (padrão atual)
        'deep':        [2, 3, 4, 3],   # Total: 12 blocos
        'very_deep':   [3, 4, 6, 3],   # Total: 16 blocos
        'front_heavy': [3, 3, 2, 1],   # Total: 9 blocos
        'back_heavy':  [1, 2, 3, 3]    # Total: 9 blocos
    }
    stage_depths = depth_configs[depth_config]
    
    # Configurar transforms
    train_transform, val_transform = get_transforms()
    
    # Criar dataset completo
    full_dataset = EyePacsLoader(
        root_dir=args.data_dir,
        csv_file=args.csv_file,
        transform=train_transform
    )
    
    # Configurar K-Fold Cross Validation
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
    fold_losses = []
    fold_results = []  # Armazenar resultados detalhados de cada fold
    fold_model_states = []  # Armazenar estados dos modelos de cada fold
    fold_histories = []  # Armazenar históricos de treinamento de cada fold
    
    # Iterar sobre os folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f'\nTrial {trial.number} | Fold {fold + 1}/{args.k_folds}')
        print(f'Hyperparameters: lr={lr:.6f}, batch_size={batch_size}, '
              f'stem_channels={stem_channels}, block_type={block_type}, head_type={head_type}')
        print(f'Architecture: depth_config={depth_config}, stage_depths={stage_depths}')
        
        # Criar samplers para train e validation
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        # Criar dataloaders
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Para validação, usar transform sem augmentation
        val_dataset = EyePacsLoader(
            root_dir=args.data_dir,
            csv_file=args.csv_file,
            transform=val_transform
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Criar modelo
        model = AnyNet(
            num_classes=args.num_classes,
            stem_channels=stem_channels,
            stage_channels=[64, 128, 256, 512],
            stage_depths=stage_depths,
            groups=8,
            width_per_group=4,
            block_type=block_type,
            head_type=head_type,
            stem_kernel_size=3
        ).to(args.device)
        
        # Escolher loss apropriada baseada no head_type
        if head_type == "coral_head":
            criterion = CoralLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Criar otimizador
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        
        # Treinar fold
        fold_f1, fold_train_metrics, fold_val_metrics, fold_best_epoch, fold_model_state, fold_history = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=args.device,
            n_epochs=args.n_epochs,
            head_type=head_type,
            num_classes=args.num_classes,
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
        
        # Armazenar estado do modelo (pesos)
        fold_model_states.append(fold_model_state)
        
        # Armazenar histórico de treinamento do fold
        fold_histories.append(fold_history)
        
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
            'head_type': head_type,
            'depth_config': depth_config,
            'stage_depths': stage_depths
        }
        best_f1_tracker['fold_results'] = fold_results
        
        # Salvar configuração e resultados
        import json
        config_path = os.path.join(args.save_dir, 'best_model_config.json')
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
                    'num_classes': args.num_classes,
                    'stem_channels': stem_channels,
                    'stage_channels': [64, 128, 256, 512],
                    'stage_depths': stage_depths,
                    'depth_config': depth_config,
                    'groups': 8,
                    'width_per_group': 4,
                    'block_type': block_type,
                    'head_type': head_type,
                    'stem_kernel_size': 3
                }
            }, f, indent=2)
        
        print(f'\n>>> Novo melhor modelo encontrado! F1: {avg_f1:.4f}')
        print(f'   Configuração salva em: {config_path}')
        
        # Salvar pesos dos modelos de cada fold (sobrescreve os anteriores)
        print(f'   Salvando pesos dos {args.k_folds} folds...')
        for fold_idx, (fold_state, fold_result) in enumerate(zip(fold_model_states, fold_results), start=1):
            fold_model_path = os.path.join(args.save_dir, f'best_model_{fold_idx}.pth')
            torch.save({
                'trial_number': trial.number,
                'fold': fold_idx,
                'model_state_dict': fold_state,
                'hyperparameters': best_f1_tracker['best_params'],
                'train_metrics': fold_result['train_metrics'],
                'val_metrics': fold_result['val_metrics'],
                'best_epoch': fold_result['best_epoch']
            }, fold_model_path)
            print(f'     >>> Fold {fold_idx} salvo: {fold_model_path} (F1: {fold_result["val_metrics"]["f1_score"]:.4f})')
        
        # Salvar históricos de treinamento de todos os folds
        histories_path = os.path.join(args.save_dir, 'best_model_histories.pkl')
        with open(histories_path, 'wb') as f:
            pickle.dump(fold_histories, f)
        print(f'   >>> Históricos de treinamento salvos: {histories_path}')
    
    return avg_f1


def train_final_model(best_params, args, save_path='best_model.pth'):
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
        root_dir=args.data_dir,
        csv_file=args.csv_file,
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
        batch_size=best_params['batch_size'],
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Criar modelo
    model = AnyNet(
        num_classes=args.num_classes,
        stem_channels=best_params['stem_channels'],
        stage_channels=[64, 128, 256, 512],
        stage_depths=best_params['stage_depths'],
        groups=8,
        width_per_group=4,
        block_type=best_params['block_type'],
        head_type=best_params['head_type'],
        stem_kernel_size=3
    ).to(args.device)
    
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
    best_epoch = 0
    
    # Inicializar histórico de treinamento
    final_history = {
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
    
    print(f"\nIniciando treinamento do modelo final por {args.n_epochs} épocas...")
    
    for epoch in range(args.n_epochs):
        print(f'\n{"="*80}')
        print(f'Epoch {epoch+1}/{args.n_epochs}')
        print(f'{"="*80}')
        
        # Treinar
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, 
            args.device, best_params['head_type'], args.num_classes
        )
        
        # Validar
        val_metrics = validate_epoch(
            model, val_loader, criterion, 
            args.device, best_params['head_type'], args.num_classes
        )
        
        # Adicionar métricas ao histórico
        final_history['epochs'].append(epoch + 1)
        final_history['train_loss'].append(train_metrics['loss'])
        final_history['train_accuracy'].append(train_metrics['accuracy'])
        final_history['train_f1'].append(train_metrics['f1_score'])
        final_history['train_kappa'].append(train_metrics['kappa'])
        final_history['train_sensitivity'].append(train_metrics['sensitivity'])
        final_history['train_specificity'].append(train_metrics['specificity'])
        final_history['val_loss'].append(val_metrics['loss'])
        final_history['val_accuracy'].append(val_metrics['accuracy'])
        final_history['val_f1'].append(val_metrics['f1_score'])
        final_history['val_kappa'].append(val_metrics['kappa'])
        final_history['val_sensitivity'].append(val_metrics['sensitivity'])
        final_history['val_specificity'].append(val_metrics['specificity'])
        
        print(f'\nTrain - Loss: {train_metrics["loss"]:.4f} | Acc: {train_metrics["accuracy"]*100:.2f}% | '
              f'F1: {train_metrics["f1_score"]:.4f} | Kappa: {train_metrics["kappa"]:.4f}')
        print(f'        Sen: {train_metrics["sensitivity"]:.4f} | Spec: {train_metrics["specificity"]:.4f}')
        print(f'\nVal   - Loss: {val_metrics["loss"]:.4f} | Acc: {val_metrics["accuracy"]*100:.2f}% | '
              f'F1: {val_metrics["f1_score"]:.4f} | Kappa: {val_metrics["kappa"]:.4f}')
        print(f'        Sen: {val_metrics["sensitivity"]:.4f} | Spec: {val_metrics["specificity"]:.4f}')
        
        # Salvar melhor modelo baseado em F1-score
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_metrics = val_metrics.copy()
            best_epoch = epoch + 1
            final_history['best_epoch'] = best_epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'hyperparameters': best_params
            }, save_path)
            print(f'>>> Modelo salvo em {save_path} (F1: {best_val_f1:.4f})')
        
        # Early stopping baseado em F1-score
        early_stopping(val_metrics['f1_score'])
        if early_stopping.early_stop:
            print(f"\n>>> Early stopping at epoch {epoch+1}")
            break
    
    # Salvar histórico de treinamento do modelo final
    final_history_path = os.path.join(args.save_dir, 'final_model_history.pkl')
    with open(final_history_path, 'wb') as f:
        pickle.dump(final_history, f)
    print(f'\n>>> Histórico do modelo final salvo: {final_history_path}')
    
    print(f"\n{'='*80}")
    print(f"TREINAMENTO FINALIZADO")
    print(f"{'='*80}")
    print(f"\nMelhores métricas de validação (época {best_epoch}):")
    print(f"  F1-score: {best_metrics['f1_score']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']*100:.2f}%")
    print(f"  Kappa: {best_metrics['kappa']:.4f}")
    print(f"  Sensitivity: {best_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {best_metrics['specificity']:.4f}")
    print(f"  Loss: {best_metrics['loss']:.4f}")
    print(f"\nModelo salvo em: {save_path}")
    
    return model


def save_study(study, filepath):
    """Salva o study do Optuna em pickle"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(study, f)
        print(f'>>> Study salvo em: {filepath}')
    except Exception as e:
        print(f'>>> Erro ao salvar study: {e}')


def load_study(filepath):
    """Carrega o study do Optuna de pickle"""
    try:
        with open(filepath, 'rb') as f:
            study = pickle.load(f)
        print(f'>>> Study carregado de: {filepath}')
        print(f'   Trials completados: {len(study.trials)}')
        if len(study.trials) > 0:
            print(f'   Melhor F1-score até agora: {study.best_value:.4f}')
        return study
    except FileNotFoundError:
        print(f'>>> Nenhum study anterior encontrado em {filepath}')
        return None
    except Exception as e:
        print(f'>>> Erro ao carregar study: {e}')
        return None


def main():
    """Função principal"""
    # Processar argumentos de linha de comando
    args = get_args()
    
    # Setar seeds para reprodutibilidade
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    
    print("="*80)
    print("OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA")
    print("="*80)
    print(f"\nConfiguração:")
    print(f"  Device: {args.device}")
    print(f"  Número de classes: {args.num_classes}")
    print(f"  Épocas por trial: {args.n_epochs}")
    print(f"  K-Folds: {args.k_folds}")
    print(f"  Número de trials: {args.n_trials}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Salvar study a cada: {args.save_study_every} trials")
    print(f"  Data directory: {args.data_dir}")
    print(f"  CSV file: {args.csv_file}")
    print(f"  Best model save directory: {args.save_dir}")
    print(f"  Study pickle file: {args.optuna_study_pkl}")
    
    # Verificar se os arquivos existem
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {args.data_dir}")
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {args.csv_file}")
    
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
    
    # Tentar carregar study anterior
    study = load_study(args.optuna_study_pkl)
    
    if study is None:
        # Criar novo study
        print("Criando novo study...")
        study = optuna.create_study(
            direction='maximize',  # Maximizar F1-score
            sampler=TPESampler(seed=args.random_seed),
            study_name='anynet_optimization'
        )
    else:
        print(f"Continuando otimização do trial {len(study.trials)}")
    
    # Callback para salvar study periodicamente
    def save_study_callback(study, trial):
        if trial.number % args.save_study_every == 0 and trial.number > 0:
            save_study(study, args.optuna_study_pkl)
    
    # Calcular quantos trials faltam
    trials_completed = len(study.trials)
    trials_remaining = max(0, args.n_trials - trials_completed)
    
    if trials_remaining > 0:
        print(f"\nExecutando {trials_remaining} trials restantes...")
        # Otimizar (passar best_f1_tracker e args para objective)
        study.optimize(
            lambda trial: objective(trial, best_f1_tracker, args), 
            n_trials=trials_remaining, 
            show_progress_bar=True,
            callbacks=[save_study_callback]
        )
        
        # Salvar study final
        save_study(study, args.optuna_study_pkl)
    else:
        print(f"\n>>> Otimização já completou {args.n_trials} trials!")
    
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
    
    # Salvar study final (garantir que está salvo)
    print("\nSalvando study final...")
    save_study(study, args.optuna_study_pkl)
    
    # Treinar modelo final com melhores hiperparâmetros
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    final_model = train_final_model(
        best_params=study.best_trial.params,
        args=args,
        save_path=final_model_path
    )
    
    print("\n" + "="*80)
    print("PROCESSO COMPLETO!")
    print("="*80)
    print(f"\n>>> Arquivos salvos:")
    print(f"  - Study Optuna: {args.optuna_study_pkl}")
    print(f"  - Melhor modelo: {final_model_path}")
    print(f"  - Configuração: {os.path.join(args.save_dir, 'best_model_config.json')}")
    print(f"  - Resultados: {study_results_path}")


if __name__ == '__main__':
    main()
