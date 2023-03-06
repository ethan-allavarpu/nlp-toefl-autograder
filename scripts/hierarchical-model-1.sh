#! /bin/bash
        
# Finetune the model
python run.py finetune --dataset ELL-ICNALE \
        --writing_params_path hierarchical-model-1.params \
        --model_type hierarchical

# Evaluate on the dev set; write to disk
python run.py evaluate --dataset FCE  \
        --reading_params_path hierarchical-model-1.params  \
        --outputs_path hierarchical-model-1.fce.predictions \
        --model_type hierarchical