"""
Script de Inferência Customizado para AnyNet
Permite especificar qualquer arquivo de pesos (.pth) e carrega a configuração do Optuna Study

Uso:
    python inference_custom.py --study_path results/optuna_study.pkl --weights_path results/best_model_1.pth --test_csv data/test.csv --test_dir path/to/images
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import pickle
import json
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, classification_report
import optuna

from model import AnyNet
from dataset import EyePacsLoader


def get_args():
    """Parse argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description='Inferência customizada com modelo AnyNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths do modelo treinado
    parser.add_argument('--study_path', type=str, required=True,
                        help='Caminho para o arquivo pickle do Optuna study')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Caminho para o arquivo de pesos (.pth)')
    parser.add_argument('--trial_number', type=int, default=None,
                        help='Número do trial (opcional: 1º tenta extrair do checkpoint, 2º usa best_trial do Optuna)')
    
    # Paths do dataset de teste
    parser.add_argument('--test_csv', type=str, required=True,
                        help='CSV com labels do conjunto de teste')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Diretório com imagens do conjunto de teste')
    parser.add_argument('--label_column', type=str, default='level',
                        help='Nome da coluna no CSV que contém as labels')
    
    # Configurações de inferência
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size para inferência')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Número de workers para DataLoader')
    
    # Normalização
    parser.add_argument('--mean', type=float, nargs=3, default=None,
                        help='Média RGB para normalização. Se None, usa ImageNet')
    parser.add_argument('--std', type=float, nargs=3, default=None,
                        help='Desvio padrão RGB para normalização. Se None, usa ImageNet')
    
    # Output
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Path para salvar predições em CSV (default: auto-gerado)')
    parser.add_argument('--save_metrics', type=str, default=None,
                        help='Path para salvar métricas de teste (default: auto-gerado)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device para inferência')
    
    args = parser.parse_args()
    
    # Configurar device
    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    return args


def load_optuna_study(study_path):
    """Carrega o study do Optuna"""
    try:
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        print(f"✅ Optuna study carregado: {study_path}")
        print(f"   Total de trials: {len(study.trials)}")
        return study
    except FileNotFoundError:
        raise FileNotFoundError(f"Study não encontrado: {study_path}")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar study: {e}")


def load_checkpoint(weights_path, device):
    """Carrega checkpoint com informações do modelo"""
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        print(f"✅ Checkpoint carregado: {weights_path}")
        return checkpoint
    except FileNotFoundError:
        raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar checkpoint: {e}")


def get_trial_from_study(study, trial_number):
    """Busca um trial específico no study"""
    for trial in study.trials:
        if trial.number == trial_number:
            return trial
    raise ValueError(f"Trial {trial_number} não encontrado no study")


def extract_model_config_from_trial(trial, checkpoint):
    """
    Extrai configuração do modelo a partir do trial do Optuna
    
    Args:
        trial: Trial do Optuna
        checkpoint: Checkpoint carregado com métricas
        
    Returns:
        config: Dict com configuração do modelo
    """
    # Reconstruir stage_depths a partir de depth_config
    depth_configs = {
        'shallow':     [1, 2, 2, 1],
        'balanced':    [2, 2, 3, 2],
        'custom':      [3, 4, 5, 3],
        'deep':        [2, 3, 4, 3],
        'very_deep':   [3, 4, 6, 3],
        'front_heavy': [3, 3, 2, 1],
        'back_heavy':  [1, 2, 3, 3]
    }
    
    params = trial.params
    depth_config = params.get('depth_config', 'balanced')
    stage_depths = depth_configs.get(depth_config, [2, 2, 3, 2])
    
    # Extrair número de classes do checkpoint
    # Tentar diferentes fontes
    num_classes = None
    if 'val_metrics' in checkpoint and 'confusion_matrix' in checkpoint['val_metrics']:
        cm = checkpoint['val_metrics']['confusion_matrix']
        if isinstance(cm, (list, np.ndarray)):
            num_classes = len(cm)
    
    # Fallback: tentar extrair do model_state_dict
    if num_classes is None and 'model_state_dict' in checkpoint:
        # Procurar pela camada final (fc ou logits)
        for key in checkpoint['model_state_dict'].keys():
            if 'head.fc.weight' in key or 'head.logits.weight' in key:
                num_classes = checkpoint['model_state_dict'][key].shape[0]
                break
    
    # Fallback final
    if num_classes is None:
        print("⚠️ Não foi possível detectar num_classes automaticamente. Usando padrão: 5")
        num_classes = 5
    
    config = {
        'trial_number': trial.number,
        'hyperparameters': params,
        'model_architecture': {
            'num_classes': num_classes,
            'stem_channels': params.get('stem_channels', 32),
            'stage_channels': [256, 512, 1024, 2048],  # Valores usados no treinamento
            'stage_depths': stage_depths,
            'depth_config': depth_config,
            'groups': 32,
            'width_per_group': 4,
            'block_type': params.get('block_type', 'residual'),
            'head_type': params.get('head_type', 'normal_head'),
            'head_dropout': params.get('head_dropout', 0.0),
            'stem_kernel_size': params.get('stem_kernel_size', 3)
        },
        'checkpoint_info': {
            'fold': checkpoint.get('fold', 'N/A'),
            'best_epoch': checkpoint.get('best_epoch', 'N/A')
        }
    }
    
    return config


def create_model_from_config(config, device):
    """
    Cria modelo AnyNet baseado na configuração
    
    Args:
        config: Dict com configuração do modelo
        device: Device para carregar o modelo
        
    Returns:
        model: Modelo AnyNet inicializado
    """
    arch = config['model_architecture']
    
    print("="*80)
    print("CONFIGURAÇÃO DO MODELO")
    print("="*80)
    print(f"Trial: {config['trial_number']}")
    print(f"Fold: {config['checkpoint_info']['fold']}")
    print(f"Melhor época: {config['checkpoint_info']['best_epoch']}")
    print(f"\nHiperparâmetros:")
    for key, value in config['hyperparameters'].items():
        print(f"  {key}: {value}")
    print(f"\nArquitetura:")
    for key, value in arch.items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")
    
    print(f"Criando modelo...")
    print(f"  - Num classes: {arch['num_classes']}")
    print(f"  - Blocos: {arch['stage_depths']} (total: {sum(arch['stage_depths'])})")
    print(f"  - Block type: {arch['block_type']}")
    print(f"  - Head type: {arch['head_type']}")
    print(f"  - Head dropout: {arch['head_dropout']:.3f}")
    print(f"  - Stage channels: {arch['stage_channels']}")
    
    model = AnyNet(
        num_classes=arch['num_classes'],
        stem_channels=arch['stem_channels'],
        stage_channels=arch['stage_channels'],
        stage_depths=arch['stage_depths'],
        groups=arch['groups'],
        width_per_group=arch['width_per_group'],
        block_type=arch['block_type'],
        se_reduction=16,
        stem_kernel_size=arch['stem_kernel_size'],
        head_type=arch['head_type'],
        head_dropout=arch['head_dropout'],
        init_weights=False
    )
    
    model = model.to(device)
    print(f"✅ Modelo criado com sucesso!\n")
    
    return model


def load_model_weights(model, checkpoint):
    """Carrega pesos no modelo"""
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ Pesos carregados no modelo!")
        
        # Exibir métricas de validação se disponíveis
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
            print(f"\nMétricas de Validação (treinamento):")
            if 'f1_score' in val_metrics:
                print(f"  Val F1: {val_metrics['f1_score']:.4f}")
            if 'accuracy' in val_metrics:
                print(f"  Val Accuracy: {val_metrics['accuracy']*100:.2f}%")
            if 'kappa' in val_metrics:
                print(f"  Val Kappa: {val_metrics['kappa']:.4f}")
        print()
        
        return model
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar pesos no modelo: {e}")


def get_test_transform(mean=None, std=None):
    """Define transformações para teste (sem augmentation)"""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transform


def run_inference(model, dataloader, device, head_type):
    """Executa inferência no conjunto de teste"""
    model.eval()
    all_preds = []
    all_targets = []
    
    print("Executando inferência...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Inferência'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calcular predições
            if head_type == "coral_head":
                predictions = model.head.predict(outputs)
            else:  # normal_head
                _, predictions = torch.max(outputs, 1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_targets)


def calculate_test_metrics(y_true, y_pred, num_classes):
    """Calcula métricas detalhadas de teste"""
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Métricas por classe
    sensitivities = []
    specificities = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    # Métricas globais
    accuracy = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Kappa por classe (one-vs-rest)
    kappa_per_class = []
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        kappa_class = cohen_kappa_score(y_true_binary, y_pred_binary)
        kappa_per_class.append(kappa_class)
    
    # IoU por classe
    iou_per_class = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        iou_per_class.append(iou)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'kappa': kappa,
        'confusion_matrix': cm,
        'sensitivity': np.mean(sensitivities),
        'specificity': np.mean(specificities),
        'sensitivities_per_class': sensitivities,
        'specificities_per_class': specificities,
        'f1_per_class': f1_per_class,
        'kappa_per_class': kappa_per_class,
        'iou_per_class': iou_per_class,
        'iou': np.mean(iou_per_class)
    }


def save_predictions_csv(predictions, test_csv, label_column, output_path):
    """Salva predições em CSV"""
    df = pd.read_csv(test_csv)
    df['predicted'] = predictions
    
    if label_column in df.columns:
        df['correct'] = (df[label_column] == df['predicted']).astype(int)
    
    df.to_csv(output_path, index=False)
    print(f"✅ Predições salvas em: {output_path}")


def save_metrics_report(metrics, config, checkpoint, output_path, num_classes):
    """Salva relatório detalhado de métricas"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO DE INFERÊNCIA - CONJUNTO DE TESTE\n")
        f.write("="*80 + "\n\n")
        
        # Informações do modelo
        f.write("INFORMAÇÕES DO MODELO:\n")
        f.write("-"*80 + "\n")
        f.write(f"Trial: {config['trial_number']}\n")
        f.write(f"Fold: {config['checkpoint_info']['fold']}\n")
        f.write(f"Melhor época (validação): {config['checkpoint_info']['best_epoch']}\n")
        
        # Métricas de validação do treinamento
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
            if 'f1_score' in val_metrics:
                f.write(f"Val F1 (treinamento): {val_metrics['f1_score']:.4f}\n")
            if 'accuracy' in val_metrics:
                f.write(f"Val Accuracy (treinamento): {val_metrics['accuracy']*100:.2f}%\n")
            if 'kappa' in val_metrics:
                f.write(f"Val Kappa (treinamento): {val_metrics['kappa']:.4f}\n")
        f.write("\n")
        
        # Hiperparâmetros
        f.write("HIPERPARÂMETROS:\n")
        f.write("-"*80 + "\n")
        for key, value in config['hyperparameters'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Métricas de teste
        f.write("MÉTRICAS NO CONJUNTO DE TESTE:\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"F1-score (macro): {metrics['f1_score']:.4f}\n")
        f.write(f"Cohen's Kappa: {metrics['kappa']:.4f}\n")
        f.write(f"IoU (macro): {metrics['iou']:.4f}\n")
        f.write(f"Sensitivity (macro): {metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity (macro): {metrics['specificity']:.4f}\n\n")
        
        # Métricas por classe
        f.write("MÉTRICAS POR CLASSE:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Classe':<10} {'F1':<10} {'Kappa':<10} {'IoU':<10} {'Sens':<10} {'Spec':<10}\n")
        f.write("-"*80 + "\n")
        for i in range(num_classes):
            f.write(f"{i:<10} "
                   f"{metrics['f1_per_class'][i]:<10.4f} "
                   f"{metrics['kappa_per_class'][i]:<10.4f} "
                   f"{metrics['iou_per_class'][i]:<10.4f} "
                   f"{metrics['sensitivities_per_class'][i]:<10.4f} "
                   f"{metrics['specificities_per_class'][i]:<10.4f}\n")
        f.write("\n")
        
        # Matriz de confusão
        f.write("MATRIZ DE CONFUSÃO:\n")
        f.write("-"*80 + "\n")
        cm = metrics['confusion_matrix']
        
        # Header
        f.write("Real\\Pred  ")
        for i in range(num_classes):
            f.write(f"{i:<8}")
        f.write("\n")
        f.write("-"*80 + "\n")
        
        # Matriz
        for i in range(num_classes):
            f.write(f"{i:<10} ")
            for j in range(num_classes):
                f.write(f"{cm[i,j]:<8}")
            f.write("\n")
        f.write("\n")
        
        # Totais por classe
        f.write("DISTRIBUIÇÃO DE CLASSES:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Classe':<10} {'Total':<10} {'%':<10}\n")
        f.write("-"*80 + "\n")
        total = cm.sum()
        for i in range(num_classes):
            count = cm[i, :].sum()
            pct = 100 * count / total
            f.write(f"{i:<10} {count:<10} {pct:<10.2f}%\n")
        f.write(f"{'Total':<10} {total:<10} 100.00%\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✅ Relatório de métricas salvo em: {output_path}")


def main():
    """Função principal"""
    args = get_args()
    
    print("\n" + "="*80)
    print("INFERÊNCIA CUSTOMIZADA COM MODELO ANYNET")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Study: {args.study_path}")
    print(f"Weights: {args.weights_path}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Test dir: {args.test_dir}")
    print("="*80 + "\n")
    
    # Verificar se arquivos existem
    if not os.path.exists(args.study_path):
        raise FileNotFoundError(f"Study não encontrado: {args.study_path}")
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Pesos não encontrados: {args.weights_path}")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"CSV de teste não encontrado: {args.test_csv}")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Diretório de teste não encontrado: {args.test_dir}")
    
    # 1. Carregar study do Optuna
    study = load_optuna_study(args.study_path)
    
    # 2. Carregar checkpoint
    checkpoint = load_checkpoint(args.weights_path, args.device)
    
    # 3. Determinar número do trial
    if args.trial_number is not None:
        trial_number = args.trial_number
        print(f"Usando trial especificado: #{trial_number}")
    elif 'trial_number' in checkpoint:
        trial_number = checkpoint['trial_number']
        print(f"Trial extraído do checkpoint: #{trial_number}")
    else:
        # Usar best_trial do Optuna se não foi especificado
        trial_number = study.best_trial.number
        print(f"⭐ Usando melhor trial do Optuna: #{trial_number}")
        print(f"   Melhor F1-score: {study.best_value:.4f}")
    
    print()
    
    # 4. Buscar trial no study
    trial = get_trial_from_study(study, trial_number)
    
    # 5. Extrair configuração do modelo
    config = extract_model_config_from_trial(trial, checkpoint)
    num_classes = config['model_architecture']['num_classes']
    head_type = config['model_architecture']['head_type']
    
    # 6. Criar modelo
    model = create_model_from_config(config, args.device)
    
    # 7. Carregar pesos
    model = load_model_weights(model, checkpoint)
    
    # 8. Preparar dataset de teste
    print("Preparando dataset de teste...")
    test_transform = get_test_transform(mean=args.mean, std=args.std)
    
    test_dataset = EyePacsLoader(
        root_dir=args.test_dir,
        csv_file=args.test_csv,
        transform=test_transform,
        label_column=args.label_column
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset de teste: {len(test_dataset)} imagens\n")
    
    # 9. Executar inferência
    predictions, targets = run_inference(model, test_loader, args.device, head_type)
    
    # 10. Calcular métricas
    print("\nCalculando métricas...")
    metrics = calculate_test_metrics(targets, predictions, num_classes)
    
    # 11. Exibir resultados
    print("\n" + "="*80)
    print("RESULTADOS NO CONJUNTO DE TESTE")
    print("="*80)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"F1-score (macro): {metrics['f1_score']:.4f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
    print(f"IoU (macro): {metrics['iou']:.4f}")
    print(f"Sensitivity (macro): {metrics['sensitivity']:.4f}")
    print(f"Specificity (macro): {metrics['specificity']:.4f}")
    print("="*80 + "\n")
    
    # Métricas por classe
    print("Métricas por Classe:")
    print("-"*80)
    print(f"{'Classe':<10} {'F1':<10} {'Kappa':<10} {'IoU':<10} {'Sens':<10} {'Spec':<10}")
    print("-"*80)
    for i in range(num_classes):
        print(f"{i:<10} "
              f"{metrics['f1_per_class'][i]:<10.4f} "
              f"{metrics['kappa_per_class'][i]:<10.4f} "
              f"{metrics['iou_per_class'][i]:<10.4f} "
              f"{metrics['sensitivities_per_class'][i]:<10.4f} "
              f"{metrics['specificities_per_class'][i]:<10.4f}")
    print()
    
    # Matriz de confusão
    print("Matriz de Confusão:")
    print("-"*80)
    print(metrics['confusion_matrix'])
    print()
    
    # 12. Gerar nomes de arquivo automáticos se não fornecidos
    if args.save_predictions is None:
        weights_basename = os.path.basename(args.weights_path).replace('.pth', '')
        args.save_predictions = f"predictions_{weights_basename}.csv"
    
    if args.save_metrics is None:
        weights_basename = os.path.basename(args.weights_path).replace('.pth', '')
        args.save_metrics = f"metrics_{weights_basename}.txt"
    
    # 13. Salvar predições
    save_predictions_csv(predictions, args.test_csv, args.label_column, args.save_predictions)
    
    # 14. Salvar relatório de métricas
    save_metrics_report(metrics, config, checkpoint, args.save_metrics, num_classes)
    
    print("\n" + "="*80)
    print("INFERÊNCIA CONCLUÍDA!")
    print("="*80)
    print(f"Predições: {args.save_predictions}")
    print(f"Métricas: {args.save_metrics}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
