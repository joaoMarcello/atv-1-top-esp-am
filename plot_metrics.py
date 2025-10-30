"""
Script para plotar históricos de treinamento dos modelos
Gera gráficos individuais para cada fold e comparações entre folds
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_fold_history(fold_history, fold_num, save_dir='plots/folds'):
    """
    Plota o histórico de treinamento de um único fold
    
    Args:
        fold_history: Dicionário com histórico do fold
        fold_num: Número do fold (1-indexed)
        save_dir: Diretório para salvar os gráficos
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = fold_history['epochs']
    best_epoch = fold_history.get('best_epoch', epochs[-1])
    
    # Criar figura com 3 linhas e 2 colunas
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Fold {fold_num} - Training History (Best Epoch: {best_epoch})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, fold_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, fold_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [a*100 for a in fold_history['train_accuracy']], 'b-', 
            label='Train Acc', linewidth=2)
    ax.plot(epochs, [a*100 for a in fold_history['val_accuracy']], 'r-', 
            label='Val Acc', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F1-Score
    ax = axes[1, 0]
    ax.plot(epochs, fold_history['train_f1'], 'b-', label='Train F1', linewidth=2)
    ax.plot(epochs, fold_history['val_f1'], 'r-', label='Val F1', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Kappa
    ax = axes[1, 1]
    ax.plot(epochs, fold_history['train_kappa'], 'b-', label='Train Kappa', linewidth=2)
    ax.plot(epochs, fold_history['val_kappa'], 'r-', label='Val Kappa', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Kappa')
    ax.set_title('Cohen\'s Kappa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Sensitivity
    ax = axes[2, 0]
    ax.plot(epochs, fold_history['train_sensitivity'], 'b-', 
            label='Train Sensitivity', linewidth=2)
    ax.plot(epochs, fold_history['val_sensitivity'], 'r-', 
            label='Val Sensitivity', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Sensitivity (Recall)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Specificity
    ax = axes[2, 1]
    ax.plot(epochs, fold_history['train_specificity'], 'b-', 
            label='Train Specificity', linewidth=2)
    ax.plot(epochs, fold_history['val_specificity'], 'r-', 
            label='Val Specificity', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Specificity')
    ax.set_title('Specificity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    save_path = os.path.join(save_dir, f'fold_{fold_num}_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'>>> Gráfico do Fold {fold_num} salvo: {save_path}')
    plt.close()


def plot_folds_comparison(fold_histories, save_dir='plots'):
    """
    Plota comparação de todos os folds em um único gráfico
    
    Args:
        fold_histories: Lista de dicionários com históricos dos folds
        save_dir: Diretório para salvar o gráfico
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_folds = len(fold_histories)
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
    
    # Criar figura com 3 linhas e 2 colunas
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Comparison of All {n_folds} Folds - Validation Metrics', 
                 fontsize=16, fontweight='bold')
    
    for fold_idx, fold_history in enumerate(fold_histories):
        fold_num = fold_idx + 1
        color = colors[fold_idx]
        epochs = fold_history['epochs']
        best_epoch = fold_history.get('best_epoch', epochs[-1])
        
        # 1. Loss
        ax = axes[0, 0]
        ax.plot(epochs, fold_history['val_loss'], color=color, 
                label=f'Fold {fold_num}', linewidth=2, alpha=0.7)
        ax.scatter(best_epoch, fold_history['val_loss'][best_epoch-1], 
                  color=color, s=100, marker='*', zorder=5)
        
        # 2. Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, [a*100 for a in fold_history['val_accuracy']], 
                color=color, label=f'Fold {fold_num}', linewidth=2, alpha=0.7)
        ax.scatter(best_epoch, fold_history['val_accuracy'][best_epoch-1]*100, 
                  color=color, s=100, marker='*', zorder=5)
        
        # 3. F1-Score
        ax = axes[1, 0]
        ax.plot(epochs, fold_history['val_f1'], color=color, 
                label=f'Fold {fold_num}', linewidth=2, alpha=0.7)
        ax.scatter(best_epoch, fold_history['val_f1'][best_epoch-1], 
                  color=color, s=100, marker='*', zorder=5)
        
        # 4. Kappa
        ax = axes[1, 1]
        ax.plot(epochs, fold_history['val_kappa'], color=color, 
                label=f'Fold {fold_num}', linewidth=2, alpha=0.7)
        ax.scatter(best_epoch, fold_history['val_kappa'][best_epoch-1], 
                  color=color, s=100, marker='*', zorder=5)
        
        # 5. Sensitivity
        ax = axes[2, 0]
        ax.plot(epochs, fold_history['val_sensitivity'], color=color, 
                label=f'Fold {fold_num}', linewidth=2, alpha=0.7)
        ax.scatter(best_epoch, fold_history['val_sensitivity'][best_epoch-1], 
                  color=color, s=100, marker='*', zorder=5)
        
        # 6. Specificity
        ax = axes[2, 1]
        ax.plot(epochs, fold_history['val_specificity'], color=color, 
                label=f'Fold {fold_num}', linewidth=2, alpha=0.7)
        ax.scatter(best_epoch, fold_history['val_specificity'][best_epoch-1], 
                  color=color, s=100, marker='*', zorder=5)
    
    # Configurar todos os subplots
    titles = ['Validation Loss', 'Validation Accuracy (%)', 'Validation F1-Score',
              'Validation Kappa', 'Validation Sensitivity', 'Validation Specificity']
    ylabels = ['Loss', 'Accuracy (%)', 'F1-Score', 'Kappa', 'Sensitivity', 'Specificity']
    
    for idx, ax in enumerate(axes.flat):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabels[idx])
        ax.set_title(titles[idx])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    save_path = os.path.join(save_dir, 'folds_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'>>> Gráfico de comparação dos folds salvo: {save_path}')
    plt.close()


def plot_final_model_history(final_history, save_dir='plots'):
    """
    Plota o histórico de treinamento do modelo final
    
    Args:
        final_history: Dicionário com histórico do modelo final
        save_dir: Diretório para salvar o gráfico
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = final_history['epochs']
    best_epoch = final_history.get('best_epoch', epochs[-1])
    
    # Criar figura com 3 linhas e 2 colunas
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Final Model Training History (Best Epoch: {best_epoch})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, final_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, final_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [a*100 for a in final_history['train_accuracy']], 'b-', 
            label='Train Acc', linewidth=2)
    ax.plot(epochs, [a*100 for a in final_history['val_accuracy']], 'r-', 
            label='Val Acc', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F1-Score
    ax = axes[1, 0]
    ax.plot(epochs, final_history['train_f1'], 'b-', label='Train F1', linewidth=2)
    ax.plot(epochs, final_history['val_f1'], 'r-', label='Val F1', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Kappa
    ax = axes[1, 1]
    ax.plot(epochs, final_history['train_kappa'], 'b-', label='Train Kappa', linewidth=2)
    ax.plot(epochs, final_history['val_kappa'], 'r-', label='Val Kappa', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Kappa')
    ax.set_title('Cohen\'s Kappa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Sensitivity
    ax = axes[2, 0]
    ax.plot(epochs, final_history['train_sensitivity'], 'b-', 
            label='Train Sensitivity', linewidth=2)
    ax.plot(epochs, final_history['val_sensitivity'], 'r-', 
            label='Val Sensitivity', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Sensitivity (Recall)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Specificity
    ax = axes[2, 1]
    ax.plot(epochs, final_history['train_specificity'], 'b-', 
            label='Train Specificity', linewidth=2)
    ax.plot(epochs, final_history['val_specificity'], 'r-', 
            label='Val Specificity', linewidth=2)
    ax.axvline(x=best_epoch, color='g', linestyle='--', label='Best Epoch', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Specificity')
    ax.set_title('Specificity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    save_path = os.path.join(save_dir, 'final_model_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'>>> Gráfico do modelo final salvo: {save_path}')
    plt.close()


def load_and_plot_all(model_dir='best_model_data', plots_dir='plots'):
    """
    Carrega todos os históricos e gera todos os gráficos
    
    Args:
        model_dir: Diretório onde estão salvos os históricos
        plots_dir: Diretório onde serão salvos os gráficos
    """
    print("\n" + "="*80)
    print("PLOTANDO HISTÓRICOS DE TREINAMENTO")
    print("="*80)
    
    # 1. Plotar históricos dos folds
    fold_histories_path = os.path.join(model_dir, 'best_model_histories.pkl')
    if os.path.exists(fold_histories_path):
        print(f"\n>>> Carregando históricos dos folds: {fold_histories_path}")
        with open(fold_histories_path, 'rb') as f:
            fold_histories = pickle.load(f)
        
        print(f">>> Encontrados {len(fold_histories)} folds")
        
        # Plotar cada fold individualmente
        print(f"\n>>> Gerando gráficos individuais dos folds...")
        for fold_idx, fold_history in enumerate(fold_histories):
            plot_fold_history(fold_history, fold_idx + 1, 
                            save_dir=os.path.join(plots_dir, 'folds'))
        
        # Plotar comparação entre folds
        print(f"\n>>> Gerando gráfico de comparação dos folds...")
        plot_folds_comparison(fold_histories, save_dir=plots_dir)
    else:
        print(f"\n>>> Arquivo de históricos dos folds não encontrado: {fold_histories_path}")
    
    # 2. Plotar histórico do modelo final
    final_history_path = os.path.join(model_dir, 'final_model_history.pkl')
    if os.path.exists(final_history_path):
        print(f"\n>>> Carregando histórico do modelo final: {final_history_path}")
        with open(final_history_path, 'rb') as f:
            final_history = pickle.load(f)
        
        print(f">>> Gerando gráfico do modelo final...")
        plot_final_model_history(final_history, save_dir=plots_dir)
    else:
        print(f"\n>>> Arquivo de histórico do modelo final não encontrado: {final_history_path}")
    
    print("\n" + "="*80)
    print("PLOTAGEM CONCLUÍDA")
    print("="*80)
    print(f"\nTodos os gráficos foram salvos em: {plots_dir}/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plotar históricos de treinamento')
    parser.add_argument('--model_dir', type=str, default='best_model_data',
                       help='Diretório com os históricos salvos')
    parser.add_argument('--plots_dir', type=str, default='plots',
                       help='Diretório para salvar os gráficos')
    
    args = parser.parse_args()
    
    load_and_plot_all(args.model_dir, args.plots_dir)
