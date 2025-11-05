@echo off
REM Script para analisar trials do Optuna e gerar gráficos
REM Usage: analyze_trials_with_plots.bat

echo ========================================
echo   ANALISADOR DE TRIALS DO OPTUNA
echo   com Geração de Gráficos
echo ========================================
echo.

REM Ativar ambiente Python (ajuste o caminho conforme necessário)
REM call conda activate seu_ambiente

REM Executar análise com geração de gráficos
python analyze_optuna_trials.py ^
    --study_path best_model_data/optuna_study.pkl ^
    --output top_trials_analysis_with_plots.txt ^
    --top_n 10 ^
    --include_pruned ^
    --generate_plots ^
    --plots_dir plots_analysis

echo.
echo ========================================
echo   Análise concluída!
echo ========================================
echo   Arquivo texto: top_trials_analysis_with_plots.txt
echo   Gráficos salvos em: plots_analysis/
echo ========================================

pause
