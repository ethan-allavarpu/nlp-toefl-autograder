#! /bin/bash


python run.py finetune --dataset ELL \
        --writing_params_path new-ell.params \
        --max_epochs 10 \
        --learning_rate 1e-05 \
        --model_type base \
        --val_losses_path val_losses/new-ell.txt \
        --outputs_path predictions/fce/new-ell.fce.predictions \
        --in_distribution_outputs_path predictions/own/new-ell.ell.predictions

python src/evaluation.py "predictions/fce/new-ell.fce.predictions" fce base

python src/evaluation.py "predictions/own/new-ell.ell.predictions" own base