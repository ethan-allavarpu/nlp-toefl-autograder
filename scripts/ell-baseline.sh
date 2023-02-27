#! /bin/bash
        
# Finetune the model
python run.py finetune --dataset ELL \
        --writing_params_path ell-baseline.params \

# Evaluate on the dev set; write to disk
python run.py evaluate --dataset ELL  \
        --reading_params_path ell-baseline.params \
        --outputs_path ell-baseline.predictions