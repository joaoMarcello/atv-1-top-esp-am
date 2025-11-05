@echo off
REM ============================================================================
REM Experimento 1 - Treinamento AnyNet para Retinopatia Diabetica
REM ============================================================================

echo ============================================================================
echo INICIANDO EXPERIMENTO 2
echo ============================================================================
echo.

REM Ativar ambiente virtual (se existir)
REM Descomente a linha abaixo se estiver usando venv
REM call venv\Scripts\activate.bat

REM Navegar para o diretorio do projeto
cd /d "%~dp0.."

REM Executar o main.py com os parametros padrao
python main.py ^
    --n_epochs 40 ^
    --k_folds 3 ^
    --n_trials 50 ^
    --random_seed 24 ^
    --num_workers 6 ^
    --save_study_every 1 ^
    --data_dir "C:/Users/Public/Documents/DATASETS/diabetic-retinopathy-detection/train_processed_224" ^
    --csv_file "data/train_labels_v2.csv" ^
    --save_dir "best_model_data_v2" ^
    --num_classes 5 ^
    --verbose

echo.
echo ============================================================================
echo EXPERIMENTO 2 FINALIZADO
echo ============================================================================
echo.

pause
