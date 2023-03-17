#! /bin/bash
        
python run_tuning.py --model baseline

python run.py evaluate --dataset FCE  \
        --reading_params_path /home/ubuntu/nlp-toefl-autograder/tuning/baseline/trial_00001/final-model.params \
        --outputs_path icnale-baseline-more-drop.fce.predictions
