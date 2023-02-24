#! /bin/bash
        
# Finetune the model
python run_speech.py finetune --dataset SPEECHOCEAN \
        --writing_params_path speechocean-baseline.params \

Evaluate on the dev set; write to disk
python run_speech.py evaluate --dataset ICNALE-EDITED  \
        --reading_params_path speechocean-baseline.params \
        --outputs_path speechocean-baseline.predictions
