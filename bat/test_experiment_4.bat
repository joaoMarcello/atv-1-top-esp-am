@echo off
REM ============================================================================
REM Teste do Experimento 4 - Inferência na base de teste de Itapecuru
REM Executa inferência com o melhor modelo encontrado pelo Optuna
REM ============================================================================

echo ============================================================================
echo INICIANDO TESTE DO EXPERIMENTO 4 - INFERENCIA
echo ============================================================================
echo.

REM Ativar ambiente virtual (se existir)
REM Descomente a linha abaixo se estiver usando venv
REM call venv\Scripts\activate.bat

REM Navegar para o diretorio do projeto
cd /d "%~dp0.."

REM Parâmetros do experimento
set MODEL_DIR=results_itapecuru_exp_4
set OUTPUT_DIR=%MODEL_DIR%\test_results
set TEST_CSV=data/revised/color/itapecuru_test_color_rev.csv
set TEST_DIR=C:/Users/Public/Documents/DATASETS/Itapecuru/RETINOGRAFO_EYER_PROCESSED
set LABEL_COLUMN=quality
set BATCH_SIZE=32
set NUM_WORKERS=6

REM Normalização específica do Itapecuru
set MEAN=0.387444 0.208384 0.132099
set STD=0.282517 0.211356 0.166142

REM Criar diretório de saída se não existir
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Configurações:
echo   Modelo: %MODEL_DIR%
echo   Resultados: %OUTPUT_DIR%
echo   Test CSV: %TEST_CSV%
echo   Test Dir: %TEST_DIR%
echo   Label Column: %LABEL_COLUMN%
echo   Batch Size: %BATCH_SIZE%
echo   Normalização: mean=%MEAN%, std=%STD%
echo.

REM Loop para executar inferência nos 5 folds
for %%f in (1 2 3 4 5) do (
    echo ============================================================================
    echo EXECUTANDO INFERENCIA - FOLD %%f
    echo ============================================================================
    echo.
    
    python inference.py ^
        --model_dir %MODEL_DIR% ^
        --fold %%f ^
        --test_csv %TEST_CSV% ^
        --test_dir %TEST_DIR% ^
        --label_column %LABEL_COLUMN% ^
        --batch_size %BATCH_SIZE% ^
        --num_workers %NUM_WORKERS% ^
        --mean %MEAN% ^
        --std %STD% ^
        --save_predictions %OUTPUT_DIR%\test_predictions_fold%%f.csv ^
        --save_metrics %OUTPUT_DIR%\test_metrics_fold%%f.txt ^
        --device auto
    
    echo.
    echo Fold %%f concluído!
    echo Predições salvas em: %OUTPUT_DIR%\test_predictions_fold%%f.csv
    echo Métricas salvas em: %OUTPUT_DIR%\test_metrics_fold%%f.txt
    echo.
)

echo.
echo ============================================================================
echo TESTE DO EXPERIMENTO 4 FINALIZADO
echo ============================================================================
echo.
echo Todos os 5 folds foram avaliados no conjunto de teste.
echo Resultados salvos em: %OUTPUT_DIR%\
echo.
echo Arquivos gerados:
echo   - test_predictions_fold1.csv ... fold5.csv
echo   - test_metrics_fold1.txt ... fold5.txt
echo.

pause
