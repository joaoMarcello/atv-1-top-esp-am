"""
Script de Inferência para AnyNet
Carrega o melhor modelo do Optuna e faz predições no conjunto de teste

Uso:
    python inference.py --test_csv data/test_labels.csv --test_dir path/to/test/images
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

from model import AnyNet
from dataset import EyePacsLoader


def get_args():
    """Parse argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description='Inferência com melhor modelo AnyNet do Optuna',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths do modelo treinado
    parser.add_argument('--model_dir', type=str, default='best_model_data',
                        help='Diretório com os arquivos do modelo (study.pkl, config.json, pesos.pth)')
    parser.add_argument('--fold', type=int, default=1,
                        help='Qual fold usar para inferência (1, 2 ou 3)')
    
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
    parser.add_argument('--save_predictions', type=str, default='predictions.csv',
                        help='Path para salvar predições em CSV')
    parser.add_argument('--save_metrics', type=str, default='test_metrics.txt',
                        help='Path para salvar métricas de teste')
    
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


def load_model_config(model_dir):
    """
    Carrega configuração do melhor modelo
    
    Args:
        model_dir: Diretório com arquivos do modelo
        
    Returns:
        config: Dict com configuração do modelo
    """
    config_path = os.path.join(model_dir, 'best_model_config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config não encontrado: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("="*80)
    print("CONFIGURAÇÃO DO MODELO")
    print("="*80)
    print(f"Trial: {config['trial_number']}")
    print(f"F1-score médio (validação): {config['avg_f1_score']:.4f}")
    print(f"\nHiperparâmetros:")
    for key, value in config['hyperparameters'].items():
        print(f"  {key}: {value}")
    print(f"\nArquitetura:")
    for key, value in config['model_architecture'].items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")
    
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
        head_dropout=arch.get('head_dropout', 0.0),  # Compatibilidade com modelos antigos
        init_weights=False  # Não inicializar, vamos carregar pesos
    )
    
    model = model.to(device)
    
    print(f"Modelo criado: {arch['head_type']}")
    print(f"  - Blocos: {arch['stage_depths']} (total: {sum(arch['stage_depths'])})")
    print(f"  - Block type: {arch['block_type']}")
    print(f"  - Head dropout: {arch.get('head_dropout', 0.0):.3f}")
    print(f"  - Stem kernel: {arch['stem_kernel_size']}")
    print()
    
    return model


def load_model_weights(model, model_dir, fold, device):
    """
    Carrega pesos do modelo de um fold específico
    
    Args:
        model: Modelo AnyNet
        model_dir: Diretório com arquivos do modelo
        fold: Número do fold (1, 2 ou 3)
        device: Device
        
    Returns:
        model: Modelo com pesos carregados
        checkpoint: Dict com informações do checkpoint
    """
    weights_path = os.path.join(model_dir, f'best_model_{fold}.pth')
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
    
    print(f"Carregando pesos do Fold {fold}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Carregar state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Pesos carregados com sucesso!")
    print(f"  Trial: {checkpoint['trial_number']}")
    print(f"  Fold: {checkpoint['fold']}")
    print(f"  Melhor época: {checkpoint['best_epoch']}")
    print(f"  Val F1: {checkpoint['val_metrics']['f1_score']:.4f}")
    print(f"  Val Accuracy: {checkpoint['val_metrics']['accuracy']*100:.2f}%")
    print(f"  Val Kappa: {checkpoint['val_metrics']['kappa']:.4f}")
    print()
    
    return model, checkpoint


def get_test_transform(mean=None, std=None):
    """
    Define transformações para teste (sem augmentation)
    
    Args:
        mean: Média RGB para normalização
        std: Desvio padrão RGB para normalização
        
    Returns:
        transform: Composição de transformações
    """
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
    """
    Executa inferência no conjunto de teste
    
    Args:
        model: Modelo AnyNet em modo eval
        dataloader: DataLoader do conjunto de teste
        device: Device
        head_type: Tipo de head ('coral_head' ou 'normal_head')
        
    Returns:
        predictions: Array numpy com predições
        targets: Array numpy com labels verdadeiros
    """
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
    """
    Calcula métricas detalhadas de teste
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        num_classes: Número de classes
        
    Returns:
        metrics: Dict com todas as métricas
    """
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
    """
    Salva predições em CSV junto com informações originais
    
    Args:
        predictions: Array numpy com predições
        test_csv: Path do CSV de teste original
        label_column: Nome da coluna de labels
        output_path: Path para salvar CSV com predições
    """
    # Carregar CSV original
    df = pd.read_csv(test_csv)
    
    # Adicionar predições
    df['predicted'] = predictions
    
    # Adicionar coluna de acerto/erro
    if label_column in df.columns:
        df['correct'] = (df[label_column] == df['predicted']).astype(int)
    
    # Salvar
    df.to_csv(output_path, index=False)
    print(f"Predições salvas em: {output_path}")


def save_metrics_report(metrics, checkpoint, output_path, num_classes):
    """
    Salva relatório detalhado de métricas
    
    Args:
        metrics: Dict com métricas calculadas
        checkpoint: Dict com informações do checkpoint
        output_path: Path para salvar relatório
        num_classes: Número de classes
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO DE INFERÊNCIA - CONJUNTO DE TESTE\n")
        f.write("="*80 + "\n\n")
        
        # Informações do modelo
        f.write("INFORMAÇÕES DO MODELO:\n")
        f.write("-"*80 + "\n")
        f.write(f"Trial: {checkpoint['trial_number']}\n")
        f.write(f"Fold: {checkpoint['fold']}\n")
        f.write(f"Melhor época (validação): {checkpoint['best_epoch']}\n")
        f.write(f"Val F1 (treinamento): {checkpoint['val_metrics']['f1_score']:.4f}\n")
        f.write(f"Val Accuracy (treinamento): {checkpoint['val_metrics']['accuracy']*100:.2f}%\n")
        f.write(f"Val Kappa (treinamento): {checkpoint['val_metrics']['kappa']:.4f}\n\n")
        
        # Hiperparâmetros
        f.write("HIPERPARÂMETROS:\n")
        f.write("-"*80 + "\n")
        for key, value in checkpoint['hyperparameters'].items():
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
    
    print(f"Relatório de métricas salvo em: {output_path}")


def main():
    """Função principal"""
    args = get_args()
    
    print("\n" + "="*80)
    print("INFERÊNCIA COM MODELO ANYNET")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Modelo: {args.model_dir}")
    print(f"Fold: {args.fold}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Test dir: {args.test_dir}")
    print("="*80 + "\n")
    
    # Verificar se arquivos existem
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"CSV de teste não encontrado: {args.test_csv}")
    if not os.path.exists(args.test_dir):
        raise FileNotFoundError(f"Diretório de teste não encontrado: {args.test_dir}")
    
    # 1. Carregar configuração do modelo
    config = load_model_config(args.model_dir)
    num_classes = config['model_architecture']['num_classes']
    head_type = config['model_architecture']['head_type']
    
    # 2. Criar modelo
    model = create_model_from_config(config, args.device)
    
    # 3. Carregar pesos do fold especificado
    model, checkpoint = load_model_weights(model, args.model_dir, args.fold, args.device)
    
    # 4. Preparar dataset de teste
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
    
    print(f"Dataset de teste: {len(test_dataset)} imagens")
    print()
    
    # 5. Executar inferência
    predictions, targets = run_inference(model, test_loader, args.device, head_type)
    
    # 6. Calcular métricas
    print("\nCalculando métricas...")
    metrics = calculate_test_metrics(targets, predictions, num_classes)
    
    # 7. Exibir resultados
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
    
    # 8. Salvar predições
    save_predictions_csv(predictions, args.test_csv, args.label_column, args.save_predictions)
    
    # 9. Salvar relatório de métricas
    save_metrics_report(metrics, checkpoint, args.save_metrics, num_classes)
    
    print("\n" + "="*80)
    print("INFERÊNCIA CONCLUÍDA!")
    print("="*80)
    print(f"Predições: {args.save_predictions}")
    print(f"Métricas: {args.save_metrics}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
