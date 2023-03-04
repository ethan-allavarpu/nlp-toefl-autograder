#! /bin/bash
        
# Finetune the model
python run.py pretrain --dataset ETS \
        --writing_params_path double-pretrain-ets1.params \
        --model_type ets \
        --max_epochs 20 \
        --learning_rate 2e-07\
        --lr_decay False

python run.py pretrain --dataset ELL \
        --reading_params_path double-pretrain-ets1.params \
        --writing_params_path double-pretrain-ets2.params \
        --max_epochs 15 \
        --learning_rate 2e-07\
        --lr_decay False

python run.py finetune --dataset ICNALE-EDITED \
        --reading_params_path double-pretrain-ets2.params \
        --writing_params_path double-pretrain-ets3.params \
        --model_type hierarchical \
        --max_epochs 10 \
        --learning_rate 2e-08\
        --lr_decay False

# Evaluate on the dev set; write to disk
python run.py evaluate --dataset FCE  \
        --reading_params_path double-pretrain-ets3.params \
        --outputs_path double-pretrain.fce.predictions \
        --model_type hierarchical