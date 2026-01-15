@echo off
REM ============================================================================
REM Inferencia do Experimento 9 - Base de Teste de Itapecuru
REM Executa inferencia com o melhor modelo encontrado pelo Optuna
REM ============================================================================

echo ============================================================================
echo INICIANDO INFERENCIA DO EXPERIMENTO 9
echo ============================================================================
echo.

REM Ativar ambiente virtual (se existir)
REM Descomente a linha abaixo se estiver usando venv
REM call venv\Scripts\activate.bat

REM Navegar para o diretorio do projeto
cd /d "%~dp0.."

REM Parametros do experimento
set MODEL_DIR=results_itapecuru_exp_9
set OUTPUT_DIR=%MODEL_DIR%\test_results
set TEST_CSV=data/revised/color/itapecuru_test_color_rev.csv
set TEST_DIR=C:/Users/Public/Documents/DATASETS/Itapecuru/RETINOGRAFO_EYER_PROCESSED
set LABEL_COLUMN=quality
set BATCH_SIZE=32
set NUM_WORKERS=6

REM Normalizacao especifica do Itapecuru
set MEAN=0.387444 0.208384 0.132099
set STD=0.282517 0.211356 0.166142

REM Criar diretorio de saida se nao existir
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Configuracoes:
echo   Modelo: %MODEL_DIR%
echo   Resultados: %OUTPUT_DIR%
echo   Test CSV: %TEST_CSV%
echo   Test Dir: %TEST_DIR%
echo   Label Column: %LABEL_COLUMN%
echo   Batch Size: %BATCH_SIZE%
echo   Normalizacao: mean=%MEAN%, std=%STD%
echo.

REM Executar inferencia com o melhor modelo (fold 1)
echo ============================================================================
echo EXECUTANDO INFERENCIA - MELHOR MODELO (FOLD 1)
echo ============================================================================
echo.

python inference.py ^
    --model_dir %MODEL_DIR% ^
    --fold 1 ^
    --test_csv %TEST_CSV% ^
    --test_dir %TEST_DIR% ^
    --label_column %LABEL_COLUMN% ^
    --batch_size %BATCH_SIZE% ^
    --num_workers %NUM_WORKERS% ^
    --mean %MEAN% ^
    --std %STD% ^
    --save_predictions %OUTPUT_DIR%\test_predictions_fold1.csv ^
    --save_metrics %OUTPUT_DIR%\test_metrics_fold1.txt ^
    --device auto

echo.
echo Inferencia concluida!
echo Predicoes salvas em: %OUTPUT_DIR%\test_predictions_fold1.csv
echo Metricas salvas em: %OUTPUT_DIR%\test_metrics_fold1.txt
echo.

echo.
echo ============================================================================
echo INFERENCIA DO EXPERIMENTO 9 FINALIZADA
echo ============================================================================
echo.
echo Resultados salvos em: %OUTPUT_DIR%\
echo.
echo Arquivos gerados:
echo   - test_predictions_fold1.csv (predicoes do modelo)
echo   - test_metrics_fold1.txt (metricas de teste)
echo.
echo Para visualizar as metricas:
echo   type %OUTPUT_DIR%\test_metrics_fold1.txt
echo.

pause
