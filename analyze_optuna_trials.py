"""
Script para analisar os melhores trials do Optuna
Extrai métricas globais e por classe dos top 5 trials
"""

import argparse
import pickle
import optuna
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
import os


def parse_args():
    """Parse argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description='Analisa os melhores trials do Optuna e salva métricas detalhadas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--study_path',
        type=str,
        required=True,
        help='Caminho para o arquivo pickle do study do Optuna (ex: best_model_data/optuna_study.pkl)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='top_trials_analysis.txt',
        help='Arquivo de saída para salvar a análise'
    )
    
    parser.add_argument(
        '--top_n',
        type=int,
        default=5,
        help='Número de melhores trials para analisar'
    )
    
    parser.add_argument(
        '--include_pruned',
        action='store_true',
        help='Se ativado, inclui trials pruned na análise (baseado no F1 do Fold 1)'
    )
    
    parser.add_argument(
        '--generate_plots',
        action='store_true',
        help='Se ativado, gera gráficos de métricas por época para cada trial'
    )
    
    parser.add_argument(
        '--plots_dir',
        type=str,
        default='plots',
        help='Diretório para salvar os gráficos'
    )
    
    return parser.parse_args()


def load_optuna_study(study_path):
    """Carrega o study do Optuna"""
    try:
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        print(f"✅ Study carregado com sucesso: {study_path}")
        print(f"   Total de trials: {len(study.trials)}")
        return study
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo não encontrado: {study_path}")
        return None
    except Exception as e:
        print(f"❌ Erro ao carregar study: {e}")
        return None


def get_trial_metrics_summary(trial):
    """Extrai resumo das métricas de um trial"""
    summary = {
        'trial_number': trial.number,
        'state': trial.state.name,
        'value': None,
        'params': trial.params,
        'fold_metrics': []
    }
    
    # Verificar se é trial completado
    if trial.state == optuna.trial.TrialState.COMPLETE:
        summary['value'] = trial.value
        # Aqui não temos acesso direto aos fold metrics, apenas ao valor final
        # mas podemos buscar nos user_attrs se foram salvos
        if 'fold_results' in trial.user_attrs:
            summary['fold_metrics'] = trial.user_attrs['fold_results']
    
    # Para trials pruned, tentar pegar métricas do primeiro fold
    elif trial.state == optuna.trial.TrialState.PRUNED:
        if 'first_fold_val_metrics' in trial.user_attrs:
            summary['value'] = trial.user_attrs['first_fold_val_metrics'].get('f1_score', 0.0)
            summary['pruned_reason'] = trial.user_attrs.get('pruned_reason', 'unknown')
            fold_metric = {
                'fold': 1,
                'val_metrics': trial.user_attrs.get('first_fold_val_metrics', {}),
                'train_metrics': trial.user_attrs.get('first_fold_train_metrics', {}),
                'best_epoch': trial.user_attrs.get('first_fold_best_epoch', 'N/A')
            }
            # Incluir histórico se disponível
            if 'first_fold_history' in trial.user_attrs:
                fold_metric['history'] = trial.user_attrs['first_fold_history']
            summary['fold_metrics'] = [fold_metric]
    
    return summary


def format_metrics_per_class(metrics, indent='    '):
    """Formata métricas por classe em tabela ASCII"""
    lines = []
    
    # Verificar se há métricas por classe
    has_metrics = any(key in metrics for key in ['f1_per_class', 'sensitivities_per_class', 
                                                   'specificities_per_class', 'kappa_per_class'])
    
    if not has_metrics:
        return ""
    
    # Determinar número de classes
    num_classes = 5  # Padrão
    if 'f1_per_class' in metrics:
        num_classes = len(metrics['f1_per_class'])
    elif 'sensitivities_per_class' in metrics:
        num_classes = len(metrics['sensitivities_per_class'])
    
    # Cabeçalho da tabela
    lines.append(f"{indent}┌───────┬────────────┬──────────────┬────────────────┬────────────┐")
    lines.append(f"{indent}│ Classe│  F1-Score  │ Sensibilidade│ Especificidade │   Kappa    │")
    lines.append(f"{indent}├───────┼────────────┼──────────────┼────────────────┼────────────┤")
    
    # Dados por classe
    for i in range(num_classes):
        f1 = metrics.get('f1_per_class', [0.0]*num_classes)[i]
        sens = metrics.get('sensitivities_per_class', [0.0]*num_classes)[i]
        spec = metrics.get('specificities_per_class', [0.0]*num_classes)[i]
        kappa = metrics.get('kappa_per_class', [0.0]*num_classes)[i]
        
        lines.append(f"{indent}│   {i}   │   {f1:6.4f}   │    {sens:6.4f}    │     {spec:6.4f}     │  {kappa:6.4f}  │")
    
    # Rodapé da tabela
    lines.append(f"{indent}└───────┴────────────┴──────────────┴────────────────┴────────────┘")
    
    return '\n'.join(lines)


def format_global_metrics(metrics, indent='    '):
    """Formata métricas globais em tabela ASCII"""
    lines = []
    
    # Tabela de métricas globais
    lines.append(f"{indent}┌─────────────────────┬──────────┐")
    lines.append(f"{indent}│      Métrica        │  Valor   │")
    lines.append(f"{indent}├─────────────────────┼──────────┤")
    
    if 'f1_score' in metrics:
        lines.append(f"{indent}│ F1-score (macro)    │  {metrics['f1_score']:6.4f}  │")
    if 'accuracy' in metrics:
        lines.append(f"{indent}│ Accuracy            │  {metrics['accuracy']*100:5.2f}%  │")
    if 'kappa' in metrics:
        lines.append(f"{indent}│ Kappa               │  {metrics['kappa']:6.4f}  │")
    if 'sensitivity' in metrics:
        lines.append(f"{indent}│ Sensitivity (macro) │  {metrics['sensitivity']:6.4f}  │")
    if 'specificity' in metrics:
        lines.append(f"{indent}│ Specificity (macro) │  {metrics['specificity']:6.4f}  │")
    if 'loss' in metrics:
        lines.append(f"{indent}│ Loss                │  {metrics['loss']:6.4f}  │")
    
    lines.append(f"{indent}└─────────────────────┴──────────┘")
    
    return '\n'.join(lines)


def generate_trial_plots(trial_summary, rank, output_dir):
    """
    Gera gráficos de métricas por época para um trial
    
    Args:
        trial_summary: Dicionário com informações do trial
        rank: Posição do trial no ranking
        output_dir: Diretório onde salvar os gráficos
    """
    # Verificar se há dados de histórico
    if not trial_summary['fold_metrics']:
        print(f"   ⚠️ Trial {trial_summary['trial_number']}: Sem dados de histórico para plotar")
        return None
    
    # Iterar pelos folds (geralmente será apenas 1 para trials pruned)
    for fold_idx, fold_data in enumerate(trial_summary['fold_metrics'], 1):
        if not isinstance(fold_data, dict) or 'history' not in fold_data:
            print(f"   ⚠️ Trial {trial_summary['trial_number']} Fold {fold_idx}: Sem histórico disponível")
            continue
        
        history = fold_data['history']
        
        # Verificar se há dados suficientes
        required_keys = ['epochs', 'train_loss', 'val_loss', 'train_f1', 'val_f1', 
                        'train_sensitivity', 'val_sensitivity', 'train_specificity', 'val_specificity']
        
        if not all(key in history for key in required_keys):
            print(f"   ⚠️ Trial {trial_summary['trial_number']} Fold {fold_idx}: Dados incompletos")
            continue
        
        epochs = history['epochs']
        if not epochs or len(epochs) == 0:
            print(f"   ⚠️ Trial {trial_summary['trial_number']} Fold {fold_idx}: Nenhuma época registrada")
            continue
        
        # Criar figura com 4 subplots (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Trial {trial_summary["trial_number"]} (Rank #{rank}) - Fold {fold_idx}\n'
                    f'Head: {trial_summary["params"].get("head_type", "N/A")} | '
                    f'Block: {trial_summary["params"].get("block_type", "N/A")} | '
                    f'Best Epoch: {fold_data.get("best_epoch", "N/A")}', 
                    fontsize=14, fontweight='bold')
        
        # 1. Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        if 'best_epoch' in fold_data and fold_data['best_epoch'] != 'N/A':
            ax1.axvline(x=fold_data['best_epoch'], color='g', linestyle='--', 
                       label=f'Best Epoch ({fold_data["best_epoch"]})', linewidth=1.5)
        ax1.set_xlabel('Época', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Loss por Época', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. F1-Score
        ax2 = axes[0, 1]
        ax2.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
        ax2.plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
        if 'best_epoch' in fold_data and fold_data['best_epoch'] != 'N/A':
            ax2.axvline(x=fold_data['best_epoch'], color='g', linestyle='--', 
                       label=f'Best Epoch ({fold_data["best_epoch"]})', linewidth=1.5)
        ax2.set_xlabel('Época', fontsize=11)
        ax2.set_ylabel('F1-Score', fontsize=11)
        ax2.set_title('F1-Score por Época', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # 3. Sensitivity
        ax3 = axes[1, 0]
        ax3.plot(epochs, history['train_sensitivity'], 'b-', label='Train Sensitivity', linewidth=2)
        ax3.plot(epochs, history['val_sensitivity'], 'r-', label='Val Sensitivity', linewidth=2)
        if 'best_epoch' in fold_data and fold_data['best_epoch'] != 'N/A':
            ax3.axvline(x=fold_data['best_epoch'], color='g', linestyle='--', 
                       label=f'Best Epoch ({fold_data["best_epoch"]})', linewidth=1.5)
        ax3.set_xlabel('Época', fontsize=11)
        ax3.set_ylabel('Sensitivity', fontsize=11)
        ax3.set_title('Sensitivity por Época', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # 4. Specificity
        ax4 = axes[1, 1]
        ax4.plot(epochs, history['train_specificity'], 'b-', label='Train Specificity', linewidth=2)
        ax4.plot(epochs, history['val_specificity'], 'r-', label='Val Specificity', linewidth=2)
        if 'best_epoch' in fold_data and fold_data['best_epoch'] != 'N/A':
            ax4.axvline(x=fold_data['best_epoch'], color='g', linestyle='--', 
                       label=f'Best Epoch ({fold_data["best_epoch"]})', linewidth=1.5)
        ax4.set_xlabel('Época', fontsize=11)
        ax4.set_ylabel('Specificity', fontsize=11)
        ax4.set_title('Specificity por Época', fontsize=12, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar figura (rank primeiro para ordenação alfabética)
        filename = f"rank_{rank:02d}_trial_{trial_summary['trial_number']:03d}_fold_{fold_idx}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   ✅ Gráfico salvo: {filename}")


def analyze_and_save(study, output_path, top_n=5, include_pruned=False, generate_plots=False, plots_dir='plots'):
    """Analisa os melhores trials e salva em arquivo"""
    
    # Separar trials por estado
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"\n📊 Análise do Study:")
    print(f"   Trials completados: {len(completed_trials)}")
    print(f"   Trials pruned: {len(pruned_trials)}")
    
    # Selecionar trials para análise
    if include_pruned and len(completed_trials) == 0:
        print(f"\n⚠️ Nenhum trial completado! Analisando trials pruned...")
        # Ordenar trials pruned por F1 do Fold 1
        trials_to_analyze = []
        for trial in pruned_trials:
            if 'first_fold_val_metrics' in trial.user_attrs:
                f1 = trial.user_attrs['first_fold_val_metrics'].get('f1_score', 0.0)
                trials_to_analyze.append((trial, f1))
        
        trials_to_analyze.sort(key=lambda x: x[1], reverse=True)
        top_trials = [t[0] for t in trials_to_analyze[:top_n]]
        analysis_type = "PRUNED"
    else:
        # Pegar os melhores trials completados (sempre ordenar manualmente para garantir top_n)
        completed_with_values = [(t, t.value) for t in completed_trials if t.value is not None]
        completed_with_values.sort(key=lambda x: x[1], reverse=True)
        top_trials = [t[0] for t in completed_with_values[:top_n]]
        analysis_type = "COMPLETADOS"
    
    print(f"   Analisando top {len(top_trials)} trials {analysis_type}...")
    
    # Criar arquivo de saída
    with open(output_path, 'w', encoding='utf-8') as f:
        # Cabeçalho
        f.write("="*80 + "\n")
        f.write(f"ANÁLISE DOS TOP {len(top_trials)} TRIALS DO OPTUNA\n")
        f.write("="*80 + "\n\n")

        f.write(f"Study: {Path(study.study_name).name if hasattr(study, 'study_name') else 'N/A'}\n")
        f.write(f"Total de trials: {len(study.trials)}\n")
        f.write(f"Trials completados: {len(completed_trials)}\n")
        f.write(f"Trials pruned: {len(pruned_trials)}\n")
        f.write(f"Tipo de análise: {analysis_type}\n")
        f.write(f"\n{'='*80}\n\n")
        
        # Analisar cada trial
        for rank, trial in enumerate(top_trials, 1):
            summary = get_trial_metrics_summary(trial)
            
            f.write(f"\n{'='*80}\n")
            f.write(f"RANK #{rank} - TRIAL {summary['trial_number']}\n")
            f.write(f"{'='*80}\n\n")
            
            # Status do trial
            f.write(f"Status: {summary['state']}\n")
            if summary['value'] is not None:
                if summary['state'] == 'COMPLETE':
                    f.write(f"F1-score médio (todos os folds): {summary['value']:.4f}\n")
                else:
                    f.write(f"F1-score (Fold 1 apenas): {summary['value']:.4f}\n")
                    if 'pruned_reason' in summary:
                        f.write(f"Razão do pruning: {summary['pruned_reason']}\n")
            
            # Hiperparâmetros
            f.write(f"\n{'-'*80}\n")
            f.write("HIPERPARÂMETROS\n")
            f.write(f"{'-'*80}\n\n")
            for key, value in summary['params'].items():
                f.write(f"  {key}: {value}\n")
            
            # Métricas por fold
            if summary['fold_metrics']:
                f.write(f"\n{'-'*80}\n")
                f.write("MÉTRICAS POR FOLD\n")
                f.write(f"{'-'*80}\n")
                
                for fold_data in summary['fold_metrics']:
                    if isinstance(fold_data, dict):
                        fold_num = fold_data.get('fold', 1)
                        f.write(f"\n  📁 FOLD {fold_num}\n")
                        f.write(f"  {'='*96}\n")
                        
                        # Melhor época e última época
                        if 'best_epoch' in fold_data:
                            f.write(f"  Melhor época: {fold_data['best_epoch']}\n")
                        
                        # Tentar obter última época do histórico
                        if 'history' in fold_data and 'epochs' in fold_data['history']:
                            last_epoch = fold_data['history']['epochs'][-1] if fold_data['history']['epochs'] else 'N/A'
                            f.write(f"  Última época executada: {last_epoch}\n")
                        
                        f.write("\n")
                        
                        # Métricas de validação
                        if 'val_metrics' in fold_data:
                            val_metrics = fold_data['val_metrics']
                            f.write(f"  🔹 MÉTRICAS DE VALIDAÇÃO (GLOBAIS):\n")
                            f.write(format_global_metrics(val_metrics, indent='    '))
                            f.write("\n\n")
                            
                            f.write(f"  🔹 MÉTRICAS DE VALIDAÇÃO (POR CLASSE):\n")
                            per_class = format_metrics_per_class(val_metrics, indent='    ')
                            if per_class:
                                f.write(per_class)
                            else:
                                f.write("    Métricas por classe não disponíveis\n")
                            f.write("\n\n")
                        
                        # Métricas de treino
                        if 'train_metrics' in fold_data:
                            train_metrics = fold_data['train_metrics']
                            f.write(f"  🔹 MÉTRICAS DE TREINO (GLOBAIS):\n")
                            f.write(format_global_metrics(train_metrics, indent='    '))
                            f.write("\n\n")
                            
                            f.write(f"  🔹 MÉTRICAS DE TREINO (POR CLASSE):\n")
                            per_class = format_metrics_per_class(train_metrics, indent='    ')
                            if per_class:
                                f.write(per_class)
                            else:
                                f.write("    Métricas por classe não disponíveis\n")
                            f.write("\n")
            else:
                f.write(f"\n{'-'*80}\n")
                f.write("⚠️ Métricas detalhadas não disponíveis para este trial\n")
                f.write(f"{'-'*80}\n")
            
            f.write("\n" + "="*80 + "\n")

        # Resumo comparativo
        f.write(f"\n\n{'='*80}\n")
        f.write("RESUMO COMPARATIVO DOS TOP TRIALS\n")
        f.write(f"{'='*80}\n\n")

        # Tabela ASCII para resumo
        f.write("┌──────┬─────────┬────────────┬──────────────┬─────────────────┬──────────────────┐\n")
        f.write("│ Rank │  Trial  │  F1-Score  │    Status    │    Head Type    │    Block Type    │\n")
        f.write("├──────┼─────────┼────────────┼──────────────┼─────────────────┼──────────────────┤\n")
        
        for rank, trial in enumerate(top_trials, 1):
            summary = get_trial_metrics_summary(trial)
            f1_str = f"{summary['value']:.4f}" if summary['value'] is not None else "  N/A   "
            head_type = summary['params'].get('head_type', 'N/A')
            block_type = summary['params'].get('block_type', 'N/A')
            
            # Truncar strings longas
            head_type_str = head_type[:15].ljust(15)
            block_type_str = block_type[:16].ljust(16)
            
            f.write(f"│  #{rank}  │  {summary['trial_number']:5}  │  {f1_str}  │ {summary['state']:12} │ {head_type_str}│ {block_type_str}│\n")
        
        f.write("└──────┴─────────┴────────────┴──────────────┴─────────────────┴──────────────────┘\n")

        f.write("\n" + "="*80 + "\n")
        f.write("FIM DA ANÁLISE\n")
        f.write("="*80 + "\n")

    print(f"\n✅ Análise salva em: {output_path}")
    print(f"   Total de trials analisados: {len(top_trials)}")
    
    # Gerar gráficos se solicitado
    if generate_plots:
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            print(f"\n📁 Diretório de plots criado: {plots_dir}")
        
        print(f"\n📊 Gerando gráficos de treinamento...")
        generated_plots = 0
        
        for rank, trial in enumerate(top_trials, 1):
            summary = get_trial_metrics_summary(trial)
            print(f"\n  🔄 Processando Trial {summary['trial_number']} (Rank #{rank})...")
            
            result = generate_trial_plots(summary, rank, plots_dir)
            if result:
                generated_plots += 1
        
        print(f"\n✅ Total de gráficos gerados: {generated_plots}")
    
    return top_trials


def main():
    """Função principal"""
    args = parse_args()
    
    print("="*80)
    print("ANALISADOR DE TRIALS DO OPTUNA")
    print("="*80)
    print(f"\nConfiguração:")
    print(f"  Study path: {args.study_path}")
    print(f"  Output file: {args.output}")
    print(f"  Top N trials: {args.top_n}")
    print(f"  Incluir pruned: {'Sim' if args.include_pruned else 'Não'}")
    print(f"  Gerar gráficos: {'Sim' if args.generate_plots else 'Não'}")
    if args.generate_plots:
        print(f"  Diretório dos gráficos: {args.plots_dir}")
    
    # Carregar study
    study = load_optuna_study(args.study_path)
    if study is None:
        print("\n❌ Não foi possível carregar o study. Abortando...")
        return
    
    # Analisar e salvar
    analyze_and_save(
        study, 
        args.output, 
        args.top_n, 
        args.include_pruned,
        generate_plots=args.generate_plots,
        plots_dir=args.plots_dir
    )
    
    print("\n" + "="*80)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*80)


if __name__ == '__main__':
    main()
