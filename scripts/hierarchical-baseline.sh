#! /bin/bash
        
# Finetune the model
python run.py pretrain --dataset ELL \
        --writing_params_path hierarchical-baseline-ell.params

python run.py finetune --dataset ICNALE-EDITED \
        --reading_params_path hierarchical-baseline-ell.params \
        --writing_params_path hierarchical-baseline.params \
        --model_type hierarchical \
        --max_epochs 10

# Evaluate on the dev set; write to disk
python run.py evaluate --dataset FCE  \
        --reading_params_path hierarchical-baseline.params  \
        --outputs_path hierarchical-baseline.fce.predictions \
        --model_type hierarchical