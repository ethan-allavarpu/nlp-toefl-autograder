#! /bin/bash

python src/evaluation.py icnale-baseline.fce.predictions data/fce-data-input-format.csv written

python src/evaluation.py ell-baseline.fce.predictions data/fce-data-input-format.csv written

python src/evaluation.py icnale-baseline-categories.fce.predictions data/fce-data-input-format.csv written

python src/evaluation.py hierarchical-baseline.fce.predictions data/fce-data-input-format.csv written
