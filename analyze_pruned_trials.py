"""
Script para analisar trials pruned do Optuna
Permite extrair e visualizar todas as informações salvas dos trials que foram pruned
"""
import pickle
import optuna
import pandas as pd
import json
from pathlib import Path


def load_study(study_path='best_model_data/optuna_study.pkl'):
    """Carrega o study do Optuna"""
    try:
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        print(f"✓ Study carregado: {study_path}")
        return study
    except FileNotFoundError:
        print(f"✗ Arquivo não encontrado: {study_path}")
        return None
    except Exception as e:
        print(f"✗ Erro ao carregar study: {e}")
        return None


def analyze_pruned_trials(study, save_csv=True, save_json=True):
    """Analisa todos os trials pruned e extrai suas informações"""
    
    if study is None:
        print("Nenhum study para analisar")
        return None
    
    # Filtrar trials pruned
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print(f"\n{'='*80}")
    print(f"ANÁLISE DE TRIALS PRUNED")
    print(f"{'='*80}\n")
    print(f"Total de trials: {len(study.trials)}")
    print(f"  - Completados: {len(completed_trials)}")
    print(f"  - Pruned: {len(pruned_trials)}")
    if len(study.trials) > 0:
        print(f"  - Taxa de pruning: {100*len(pruned_trials)/len(study.trials):.1f}%\n")
    
    if not pruned_trials:
        print("Nenhum trial foi pruned.")
        return None
    
    # Coletar dados dos trials pruned
    pruned_data = []
    
    for trial in pruned_trials:
        data = {
            'trial_number': trial.number,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start,
            'datetime_complete': trial.datetime_complete,
            'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None,
        }
        
        # Adicionar hiperparâmetros
        for key, value in trial.params.items():
            data[f'param_{key}'] = value
        
        # Adicionar atributos do usuário (métricas salvas)
        data['pruned_reason'] = trial.user_attrs.get('pruned_reason', 'N/A')
        data['pruned_threshold'] = trial.user_attrs.get('pruned_threshold', None)
        data['folds_completed'] = trial.user_attrs.get('folds_completed', 0)
        data['first_fold_best_epoch'] = trial.user_attrs.get('first_fold_best_epoch', None)
        
        # Métricas de validação do fold 1
        if 'first_fold_val_metrics' in trial.user_attrs:
            val_metrics = trial.user_attrs['first_fold_val_metrics']
            data['val_f1_score'] = val_metrics.get('f1_score', None)
            data['val_accuracy'] = val_metrics.get('accuracy', None)
            data['val_kappa'] = val_metrics.get('kappa', None)
            data['val_sensitivity'] = val_metrics.get('sensitivity', None)
            data['val_specificity'] = val_metrics.get('specificity', None)
            data['val_loss'] = val_metrics.get('loss', None)
        
        # Métricas de treino do fold 1
        if 'first_fold_train_metrics' in trial.user_attrs:
            train_metrics = trial.user_attrs['first_fold_train_metrics']
            data['train_f1_score'] = train_metrics.get('f1_score', None)
            data['train_accuracy'] = train_metrics.get('accuracy', None)
            data['train_kappa'] = train_metrics.get('kappa', None)
            data['train_sensitivity'] = train_metrics.get('sensitivity', None)
            data['train_specificity'] = train_metrics.get('specificity', None)
            data['train_loss'] = train_metrics.get('loss', None)
        
        pruned_data.append(data)
    
    # Criar DataFrame
    df = pd.DataFrame(pruned_data)
    
    # Exibir resumo
    print(f"\n{'='*80}")
    print("RESUMO DOS TRIALS PRUNED")
    print(f"{'='*80}\n")
    
    if 'val_f1_score' in df.columns:
        print("Estatísticas do F1-score de validação (Fold 1):")
        print(f"  Média: {df['val_f1_score'].mean():.4f}")
        print(f"  Mediana: {df['val_f1_score'].median():.4f}")
        print(f"  Min: {df['val_f1_score'].min():.4f}")
        print(f"  Max: {df['val_f1_score'].max():.4f}")
        print(f"  Desvio padrão: {df['val_f1_score'].std():.4f}\n")
    
    if 'duration' in df.columns and df['duration'].notna().any():
        print("Tempo economizado com pruning:")
        avg_duration = df['duration'].mean()
        total_saved = avg_duration * len(pruned_trials) * 2  # Economia de ~2 folds
        print(f"  Tempo médio por trial pruned: {avg_duration:.1f}s ({avg_duration/60:.1f} min)")
        print(f"  Tempo total economizado (estimativa): {total_saved:.1f}s ({total_saved/60:.1f} min)\n")
    
    # Salvar dados
    if save_csv:
        csv_path = 'pruned_trials_analysis.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Dados salvos em CSV: {csv_path}")
    
    if save_json:
        # Criar versão mais completa com históricos
        detailed_data = []
        for trial in pruned_trials:
            trial_data = {
                'trial_number': trial.number,
                'state': trial.state.name,
                'params': trial.params,
                'user_attrs': dict(trial.user_attrs),
                'datetime_start': str(trial.datetime_start),
                'datetime_complete': str(trial.datetime_complete),
            }
            detailed_data.append(trial_data)
        
        json_path = 'pruned_trials_detailed.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Dados detalhados salvos em JSON: {json_path}")
    
    print(f"\n{'='*80}")
    print("ANÁLISE POR HIPERPARÂMETRO")
    print(f"{'='*80}\n")
    
    # Analisar quais hiperparâmetros são mais comuns em trials pruned
    param_columns = [col for col in df.columns if col.startswith('param_')]
    
    for col in param_columns:
        param_name = col.replace('param_', '')
        print(f"\n{param_name}:")
        value_counts = df[col].value_counts()
        for value, count in value_counts.items():
            percentage = 100 * count / len(df)
            print(f"  {value}: {count} trials ({percentage:.1f}%)")
    
    return df


def main():
    """Função principal"""
    print("\n" + "="*80)
    print("ANÁLISE DE TRIALS PRUNED - OPTUNA")
    print("="*80 + "\n")
    
    # Carregar study
    study = load_study('best_model_data/optuna_study.pkl')
    
    if study is None:
        return
    
    # Analisar trials pruned
    df = analyze_pruned_trials(study, save_csv=True, save_json=True)
    
    if df is not None:
        print(f"\n{'='*80}")
        print("ANÁLISE CONCLUÍDA!")
        print(f"{'='*80}\n")
        print("Arquivos gerados:")
        print("  - pruned_trials_analysis.csv (dados tabulares)")
        print("  - pruned_trials_detailed.json (dados completos com históricos)")
        print("\nVocê pode abrir o CSV no Excel/Pandas para análise mais detalhada.")


if __name__ == '__main__':
    main()
