#! /bin/bash
        
# Finetune the model
python run_tuning.py --model ell-baseline

# Evaluate on the dev set; write to disk
python run.py evaluate --dataset FCE  \
        --reading_params_path /home/ubuntu/nlp-toefl-autograder/tuning/ell-baseline/trial_00000/best-model.params \
        --outputs_path ell-baseline-more-drop.fce.predictions \
        --model_type base
