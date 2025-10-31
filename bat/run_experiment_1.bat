@echo off
REM ============================================================================
REM Experimento 1 - Treinamento AnyNet para Retinopatia Diabetica
REM ============================================================================

echo ============================================================================
echo INICIANDO EXPERIMENTO 1
echo ============================================================================
echo.

REM Ativar ambiente virtual (se existir)
REM Descomente a linha abaixo se estiver usando venv
REM call venv\Scripts\activate.bat

REM Navegar para o diretorio do projeto
cd /d "%~dp0.."

REM Executar o main.py com os parametros padrao
python main.py ^
    --n_epochs 2 ^
    --k_folds 3 ^
    --n_trials 3 ^
    --random_seed 42 ^
    --num_workers 4 ^
    --save_study_every 1 ^
    --data_dir "C:/Users/Public/Documents/DATASETS/diabetic-retinopathy-detection/train" ^
    --csv_file "data/trainLabels.csv" ^
    --save_dir "best_model_data" ^
    --num_classes 5 ^
    --verbose

echo.
echo ============================================================================
echo EXPERIMENTO 1 FINALIZADO
echo ============================================================================
echo.

pause
