@echo off
REM ============================================================================
REM Experimento 3 - Avaliação de Qualidade na base de Itapecuru
REM mean=[0.387444, 0.208384, 0.132099], std=[0.282517, 0.211356, 0.166142]
REM ============================================================================

echo ============================================================================
echo INICIANDO EXPERIMENTO 3
echo ============================================================================
echo.

REM Ativar ambiente virtual (se existir)
REM Descomente a linha abaixo se estiver usando venv
REM call venv\Scripts\activate.bat

REM Navegar para o diretorio do projeto
cd /d "%~dp0.."

REM Executar o main.py com os parametros padrao
python main.py ^
    --n_epochs 60 ^
    --k_folds 5 ^
    --n_trials 100 ^
    --random_seed 42 ^
    --num_workers 6 ^
    --save_study_every 1 ^
    --mean 0.387444 0.208384 0.132099 ^
    --std 0.282517 0.211356 0.166142 ^
    --label_column "quality" ^
    --data_dir "C:/Users/Public/Documents/DATASETS/Itapecuru/RETINOGRAFO_EYER_PROCESSED" ^
    --csv_file "data/revised/color/itapecuru_train_full_color_rev.csv" ^
    --save_dir "results_itapecuru_exp_3" ^
    --num_classes 3 ^
    --patience 10 ^
    --min_epochs 25 ^
    --use_scheduler ^
    --scheduler_eta_min 1e-7 ^
    --verbose

echo.
echo ============================================================================
echo EXPERIMENTO 3 FINALIZADO
echo ============================================================================
echo.

pause
