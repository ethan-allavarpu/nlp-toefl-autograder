#! /bin/bash
        
# Finetune the model
python run.py finetune --dataset ICNALE-EDITED \
        --writing_params_path og-icnale.params \
        --max_epochs 20 \
        --model_type base-og \
        --val_losses_path val_losses/og-icnale.txt \
        --outputs_path predictions/fce/og-icnale.fce.predictions \
        --in_distribution_outputs_path predictions/own/og-icnale.icnale.predictions

# Evaluate on the dev set; write to disk
python run.py evaluate --dataset FCE  \
        --reading_params_path /home/ubuntu/TESTING/nlp-toefl-autograder/icnale-baseline.params \
        --outputs_path icnale-simple-is-better.predictions

python src/evaluation.py "predictions/fce/og-icnale.fce.predictions" fce base


# python run.py finetune --dataset ICNALE-EDITED \
#         --writing_params_path new-icnale.params \
#         --max_epochs 20 \
#         --learning_rate 1e-05 \
#         --model_type base \
#         --val_losses_path val_losses/new-icnale.txt \
#         --outputs_path predictions/fce/new-icnale.fce.predictions \
#         --in_distribution_outputs_path predictions/own/new-icnale.icnale.predictions

# python src/evaluation.py "predictions/fce/new-icnale.fce.predictions" fce base

# python src/evaluation.py "predictions/own/new-icnale.icnale.predictions" own base

# python run.py finetune --dataset ICNALE-EDITED \
#         --writing_params_path new-conv-icnale.params \
#         --max_epochs 10 \
#         --learning_rate 1e-05 \
#         --model_type base \
#         --val_losses_path val_losses/new-icnale-conv.txt \
#         --outputs_path predictions/fce/new-icnale-conv.fce.predictions \
#         --in_distribution_outputs_path predictions/own/new-icnale-conv.icnale.predictions

# python src/evaluation.py "predictions/fce/new-icnale-conv.fce.predictions" fce base

