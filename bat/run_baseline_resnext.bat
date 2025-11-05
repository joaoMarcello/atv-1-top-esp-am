@echo off
REM Script para treinar baseline ResNeXt pré-treinado (ImageNet)

echo ========================================
echo   BASELINE: RESNEXT PRE-TREINADO
echo   ImageNet - Validation Fold 1
echo ========================================
echo.

REM Ativar ambiente Python (ajuste conforme necessário)
REM call conda activate seu_ambiente

REM Executar treinamento do baseline
python train_baseline_resnext.py ^
    --model_name "resnext50_32x4d.a1h_in1k" ^
    --pretrained imagenet ^
    --n_epochs 40 ^
    --batch_size 32 ^
    --lr 0.0001 ^
    --weight_decay 0.0001 ^
    --patience 6 ^
    --min_epochs 20 ^
    --k_folds 3 ^
    --num_workers 6 ^
    --save_dir baseline_results ^
    --random_seed 24 ^
    --device auto

echo.
echo ========================================
echo   Treinamento concluído!
echo ========================================
echo   Resultados salvos em: baseline_results/
echo ========================================

pause
