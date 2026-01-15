@echo off
REM ============================================================================
REM Experimento 5 - Avaliação de Qualidade na base de Itapecuru
REM Deixei os critérios de pruning menos agressivos. Não vou class_weights em CoralLoss. Removi batch size 16 e stem channels 16 do espaço de busca. Corrigi o dataleak usando a stratifiedGroupKFold e a coluna paciente. Troquei o scheduler para a versão do timm porque tem warm-up. Não estou fazendo validação cruzada.
REM mean=[0.387444, 0.208384, 0.132099], std=[0.282517, 0.211356, 0.166142]
REM ============================================================================

echo ============================================================================
echo INICIANDO EXPERIMENTO 5
echo ============================================================================
echo.

REM Ativar ambiente virtual (se existir)
REM Descomente a linha abaixo se estiver usando venv
REM call venv\Scripts\activate.bat

@REM "comment": "0.7204, 0.6902, 0.7121, 0.7388, 0.7105",
@REM             0.7127, 0.7056, 0.7359, 0.7025, 0.7245 - mean=0.7162

REM Navegar para o diretorio do projeto
cd /d "%~dp0.."

REM Executar o main.py com os parametros padrao
python main.py ^
    --n_epochs 60 ^
    --k_folds 1 ^
    --n_trials 100 ^
    --random_seed 24 ^
    --num_workers 6 ^
    --save_study_every 1 ^
    --mean 0.387444 0.208384 0.132099 ^
    --std 0.282517 0.211356 0.166142 ^
    --label_column "quality" ^
    --patient_column "paciente" ^
    --data_dir "C:/Users/Public/Documents/DATASETS/Itapecuru/RETINOGRAFO_EYER_PROCESSED" ^
    --csv_file "data/revised/color/itapecuru_train_full_color_rev.csv" ^
    --save_dir "results_itapecuru_exp_7" ^
    --num_classes 3 ^
    --patience 10 ^
    --min_epochs 25 ^
    --pruning_threshold 0.60 ^
    --min_trials_before_pruning 100 ^
    --use_scheduler ^
    --scheduler_eta_min 1e-7 ^
    --scheduler_warmup_epochs 5 ^
    --scheduler_warmup_lr 1e-6 ^
    --verbose

echo.
echo ============================================================================
echo EXPERIMENTO 5 FINALIZADO
echo ============================================================================
echo.

pause
