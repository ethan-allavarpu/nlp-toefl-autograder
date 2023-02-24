#! /bin/bash
        
# Finetune the model
python run.py finetune --dataset ICNALE-EDITED \
        --writing_params_path icnale-baseline.params \

# Evaluate on the dev set; write to disk
python run.py evaluate --dataset ICNALE-EDITED  \
        --reading_params_path icnale-baseline.params \
        --outputs_path icnale-baseline.predictions
