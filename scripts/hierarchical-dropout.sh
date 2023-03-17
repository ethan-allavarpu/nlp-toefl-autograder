#! /bin/bash
        
python run_tuning.py --model hierarchical

python run.py evaluate --dataset FCE  \
        --reading_params_path /home/ubuntu/nlp-toefl-autograder/tuning/hierarchical/trial_00001/best-model.params \
        --outputs_path hierarchical-more-drop.fce.predictions \
        --model_type hierarchical


python run.py evaluate --dataset FCE  \
        --reading_params_path /home/ubuntu/nlp-toefl-autograder/tuning/multitask/trial_00000/best-model.params \
        --outputs_path multitask.fce.predictions \
        --model_type multitask