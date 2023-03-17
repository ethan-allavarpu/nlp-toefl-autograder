#! /bin/bash

# Evaluate on the dev set; write to disk
python run.py evaluate --dataset FCE  \
        --reading_params_path tuning/baseline/trial_00001/best-model.params \
        --outputs_path icnale-baseline.predictions