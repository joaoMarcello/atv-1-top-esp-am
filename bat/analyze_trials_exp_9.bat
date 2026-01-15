@echo off
REM ============================================================================
REM Análise de Trials do Optuna - Experimento 9
REM Analisa os melhores trials da otimização do experimento 9
REM ============================================================================

echo ============================================================================
echo ANALISANDO TRIALS DO EXPERIMENTO 9
echo ============================================================================
echo.

REM Ativar ambiente virtual (se existir)
REM Descomente a linha abaixo se estiver usando venv
REM call venv\Scripts\activate.bat

REM Navegar para o diretorio do projeto
cd /d "%~dp0.."

REM Parametros da analise
set STUDY_PATH=results_itapecuru_exp_9\optuna_study.pkl
set OUTPUT_FILE=results_itapecuru_exp_9\top_trials_analysis.txt
set TOP_N=10
set PLOTS_DIR=results_itapecuru_exp_9\trial_plots

echo Configuracoes:
echo   Study: %STUDY_PATH%
echo   Output: %OUTPUT_FILE%
echo   Top N trials: %TOP_N%
echo   Plots dir: %PLOTS_DIR%
echo.

REM Executar analise dos trials
python analyze_optuna_trials.py ^
    --study_path %STUDY_PATH% ^
    --output %OUTPUT_FILE% ^
    --top_n %TOP_N% ^
    --generate_plots ^
    --plots_dir %PLOTS_DIR%

echo.
echo ============================================================================
echo ANALISE CONCLUIDA
echo ============================================================================
echo.
echo Resultados salvos em:
echo   - Analise: %OUTPUT_FILE%
echo   - Graficos: %PLOTS_DIR%\
echo.
echo Para visualizar a analise:
echo   type %OUTPUT_FILE%
echo.

pause
