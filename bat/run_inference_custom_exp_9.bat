@echo off
REM ============================================================================
REM Script de Inferência Customizado - Experimento 9 (Itapecuru)
REM Permite especificar qualquer arquivo de pesos
REM ============================================================================

echo ================================================================================
echo INFERENCIA CUSTOMIZADA - EXPERIMENTO 9
echo ================================================================================
echo.

REM Configurações do experimento
set EXPERIMENT_DIR=results_itapecuru_exp_9
set STUDY_PATH=%EXPERIMENT_DIR%\optuna_study.pkl
set OUTPUT_DIR=%EXPERIMENT_DIR%\test_results_custom

REM Configurações do teste
set TEST_CSV=data\revised\color\itapecuru_test_color_rev.csv
set TEST_DIR=C:\Users\Public\Documents\DATASETS\Itapecuru\RETINOGRAFO_EYER_PROCESSED
set LABEL_COLUMN=quality

REM Configurações de inferência
set BATCH_SIZE=32
set NUM_WORKERS=2

REM Normalização customizada (Itapecuru)
set MEAN=0.387444 0.208384 0.132099
set STD=0.282517 0.211356 0.166142

REM ============================================================================
REM CONFIGURAÇÃO: Escolha o arquivo de pesos (.pth) para usar
REM ============================================================================
REM Opções disponíveis:
REM   - best_model_1.pth (Fold 1)
REM   - best_model_2.pth (Fold 2)
REM   - best_model_3.pth (Fold 3)

REM Defina aqui qual peso usar:
set WEIGHTS_PATH=%EXPERIMENT_DIR%\final_model.pth

REM Opcional: número do trial
REM Se não fornecido, ordem de prioridade:
REM   1. Tenta extrair do checkpoint (.pth)
REM   2. Usa o best_trial do Optuna study
REM set TRIAL_NUMBER=19

REM ============================================================================

echo Configuracoes:
echo   Study: %STUDY_PATH%
echo   Weights: %WEIGHTS_PATH%
echo   Test CSV: %TEST_CSV%
echo   Test dir: %TEST_DIR%
echo   Output: %OUTPUT_DIR%
echo.

REM Criar diretório de saída
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
    echo Diretorio de saida criado: %OUTPUT_DIR%
)

REM Executar inferência
echo.
echo Executando inferencia...
echo ================================================================================

python inference_custom.py ^
    --study_path "%STUDY_PATH%" ^
    --weights_path "%WEIGHTS_PATH%" ^
    --test_csv "%TEST_CSV%" ^
    --test_dir "%TEST_DIR%" ^
    --label_column "%LABEL_COLUMN%" ^
    --batch_size %BATCH_SIZE% ^
    --num_workers %NUM_WORKERS% ^
    --mean %MEAN% ^
    --std %STD% ^
    --save_predictions "%OUTPUT_DIR%\predictions.csv" ^
    --save_metrics "%OUTPUT_DIR%\metrics.txt"

echo.
echo ================================================================================
echo INFERENCIA CONCLUIDA!
echo ================================================================================
echo Resultados salvos em: %OUTPUT_DIR%
echo.

pause
