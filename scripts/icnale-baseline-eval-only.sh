#! /bin/bash

# Evaluate on the dev set; write to disk
python run.py evaluate --dataset FCE  \
        --reading_params_path icnale-baseline.params \
        --outputs_path icnale-baseline.predictions