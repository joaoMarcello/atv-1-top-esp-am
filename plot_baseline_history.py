"""
Script para plotar gráficos do treinamento baseline
Lê o histórico salvo e gera gráficos de métricas
"""

import os
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def parse_args():
    """Parse argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description='Plota gráficos do treinamento baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--history_path',
        type=str,
        default='baseline_results/baseline_history.pkl',
        help='Caminho para o arquivo pickle com histórico'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='baseline_results',
        help='Diretório para salvar os gráficos'
    )
    
    parser.add_argument(
        '--output_name',
        type=str,
        default='baseline_training_history.png',
        help='Nome do arquivo de saída'
    )
    
    return parser.parse_args()


def plot_baseline_history(history, output_path):
    """
    Gera gráfico com 6 subplots mostrando evolução das métricas
    
    Args:
        history: Dicionário com histórico de treinamento
        output_path: Caminho para salvar o gráfico
    """
    epochs = history['epochs']
    best_epoch = history.get('best_epoch', epochs[-1])
    
    # Criar figura com 6 subplots (3x2)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Baseline ResNeXt - Histórico de Treinamento', 
                fontsize=16, fontweight='bold')
    
    # 1. Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.axvline(x=best_epoch, color='g', linestyle='--', 
               label=f'Best Epoch ({best_epoch})', linewidth=1.5)
    ax1.set_xlabel('Época', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss por Época', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. F1-Score
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    ax2.plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', 
               label=f'Best Epoch ({best_epoch})', linewidth=1.5)
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel('F1-Score', fontsize=11)
    ax2.set_title('F1-Score por Época', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Accuracy
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    ax3.plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    ax3.axvline(x=best_epoch, color='g', linestyle='--', 
               label=f'Best Epoch ({best_epoch})', linewidth=1.5)
    ax3.set_xlabel('Época', fontsize=11)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Accuracy por Época', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Kappa
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['train_kappa'], 'b-', label='Train Kappa', linewidth=2)
    ax4.plot(epochs, history['val_kappa'], 'r-', label='Val Kappa', linewidth=2)
    ax4.axvline(x=best_epoch, color='g', linestyle='--', 
               label=f'Best Epoch ({best_epoch})', linewidth=1.5)
    ax4.set_xlabel('Época', fontsize=11)
    ax4.set_ylabel('Kappa', fontsize=11)
    ax4.set_title('Kappa por Época', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Sensitivity
    ax5 = axes[2, 0]
    ax5.plot(epochs, history['train_sensitivity'], 'b-', label='Train Sensitivity', linewidth=2)
    ax5.plot(epochs, history['val_sensitivity'], 'r-', label='Val Sensitivity', linewidth=2)
    ax5.axvline(x=best_epoch, color='g', linestyle='--', 
               label=f'Best Epoch ({best_epoch})', linewidth=1.5)
    ax5.set_xlabel('Época', fontsize=11)
    ax5.set_ylabel('Sensitivity', fontsize=11)
    ax5.set_title('Sensitivity por Época', fontsize=12, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # 6. Specificity
    ax6 = axes[2, 1]
    ax6.plot(epochs, history['train_specificity'], 'b-', label='Train Specificity', linewidth=2)
    ax6.plot(epochs, history['val_specificity'], 'r-', label='Val Specificity', linewidth=2)
    ax6.axvline(x=best_epoch, color='g', linestyle='--', 
               label=f'Best Epoch ({best_epoch})', linewidth=1.5)
    ax6.set_xlabel('Época', fontsize=11)
    ax6.set_ylabel('Specificity', fontsize=11)
    ax6.set_title('Specificity por Época', fontsize=12, fontweight='bold')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    # Ajustar layout e salvar
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Gráfico salvo: {output_path}")


def main():
    """Função principal"""
    args = parse_args()
    
    print("="*80)
    print("GERADOR DE GRÁFICOS DO BASELINE")
    print("="*80)
    print(f"\nConfiguração:")
    print(f"  History path: {args.history_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output name: {args.output_name}")
    
    # Verificar se arquivo existe
    if not os.path.exists(args.history_path):
        print(f"\n❌ ERRO: Arquivo não encontrado: {args.history_path}")
        print("   Execute o treinamento primeiro com train_baseline_resnext.py")
        return
    
    # Carregar histórico
    print(f"\n📂 Carregando histórico...")
    with open(args.history_path, 'rb') as f:
        history = pickle.load(f)
    
    print(f"✓ Histórico carregado:")
    print(f"  - Épocas treinadas: {len(history['epochs'])}")
    print(f"  - Melhor época: {history.get('best_epoch', 'N/A')}")
    
    # Criar diretório de saída se necessário
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Gerar gráfico
    print(f"\n📊 Gerando gráficos...")
    output_path = os.path.join(args.output_dir, args.output_name)
    plot_baseline_history(history, output_path)
    
    print(f"\n{'='*80}")
    print("PROCESSO COMPLETO!")
    print(f"{'='*80}")
    print(f"\nGráfico salvo em: {output_path}")


if __name__ == '__main__':
    main()
