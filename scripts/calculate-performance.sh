#! /bin/bash

python src/evaluation.py "predictions/fce/new-icnale-conv.fce.predictions" fce base 10 "images/output_icnale_conv_10.png"

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/icnale-baseline.fce.predictions" fce base 10 "images/output_icnale_10.png"

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/ell-baseline.fce.predictions" fce base 10 "images/output_ell_10.png"

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/icnale-baseline-categories.fce.predictions" fce base 10 "images/output_icnale_categories_10.png"

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/double-pretrain.fce.predictions" fce base 10 "images/output_double_pretrain_10.png"

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/hierarchical-baseline.fce.predictions" fce base 10 "images/output_hierarchical_10.png"

# Trial 1 final params
python src/evaluation.py icnale-baseline-more-drop.fce.predictions fce base 10 "images/output_icnale_more_drop_10.png"

# Trial 0 best params
python src/evaluation.py ell-baseline-more-drop.fce.predictions fce base 10 "images/output_ell_more_drop_10.png"

# Trial 1 best params
python src/evaluation.py hierarchical-more-drop.fce.predictions fce hierarchical 10 "images/output_hierarchical_more_drop_10.png"

# Trial 3 final params
python src/evaluation.py ell-baseline.fce.predictions fce base 10 "images/output_ell_10.png"

python src/evaluation.py icnale-baseline.predictions fce base   10 "images/output_icnale_10.png"

python src/evaluation.py hierarchical-model-standardized.fce.predictions fce hierarchical 10 "images/output_hierarchical-standardized_10.png" "images/output_hierarchical-standardized_10-hist.png" "Hierarchical Model Confusion Matrix"

# Trial 1 final params
python src/evaluation.py hierarchical-model-normalized.fce.predictions fce hierarchical 10 "images/output_hierarchical_10.png"

# Trial 3 final params
python src/evaluation.py ets-pretrain.fce.predictions fce base 10 "images/output_ets_10.png"

# MULTITASK
python src/evaluation.py multitask.fce.predictions fce hierarchical 5 "images/output_multitask_5.png"

python src/evaluation-ensemble.py 2 ets-pretrain.fce.predictions icnale-baseline-more-drop.fce.predictions 1 2 fce 10 "images/output_ets_icnale_10.png"

python src/evaluation-ensemble.py 2 ell-baseline.fce.predictions icnale-baseline-more-drop.fce.predictions 1 2 fce 10 "images/output_ell_icnale_10.png"

python src/evaluation-ensemble.py 2 ets-pretrain.fce.predictions ell-baseline.fce.predictions 1 2 fce 10 "images/output_ets_ell_10.png"

python src/evaluation-ensemble.py 3 ell-baseline.fce.predictions ets-pretrain.fce.predictions icnale-baseline.predictions 1 2 2 fce 10 "images/output_ets_ell_icnale_10.png"


# AGAIN BUT FOR 5 BUCKETS

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/icnale-baseline.fce.predictions" fce base 5 "images/output_icnale_5.png"

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/ell-baseline.fce.predictions" fce base 5 "images/output_ell_5.png"

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/icnale-baseline-categories.fce.predictions" fce base 5 "images/output_icnale_categories_5.png"

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/double-pretrain.fce.predictions" fce base 5 "images/output_double_pretrain_5.png"

python src/evaluation.py "PARAMS AND PREDICTIONS FROM BEFORE/hierarchical-baseline.fce.predictions" fce base 5 "images/output_hierarchical_5.png"

# Trial 1 final params
python src/evaluation.py icnale-baseline-more-drop.fce.predictions fce base 5 "images/output_icnale_more_drop_5.png"

# Trial 0 best params
python src/evaluation.py ell-baseline-more-drop.fce.predictions fce base 5 "images/output_ell_more_drop_5.png"

# Trial 1 best params
python src/evaluation.py hierarchical-more-drop.fce.predictions fce hierarchical 5 "images/output_hierarchical_more_drop_5.png"

# Trial 3 final params
python src/evaluation.py ell-baseline.fce.predictions fce base 5 "images/output_ell_5.png"

python src/evaluation.py icnale-baseline.predictions fce base   5 "images/output_icnale_5.png"

python src/evaluation.py hierarchical-model-standardized.fce.predictions fce hierarchical 5 "images/output_hierarchical-standardized_5.png"
# Trial 1 final params
python src/evaluation.py hierarchical-model-normalized.fce.predictions fce hierarchical 5 "images/output_hierarchical_5.png"

# Trial 3 final params
python src/evaluation.py ets-pretrain.fce.predictions fce base 5 "images/output_ets_5.png"

# MULTITASK
python src/evaluation.py multitask.fce.predictions fce hierarchical 10 "images/output_multitask_10.png"

python src/evaluation-ensemble.py 2 ets-pretrain.fce.predictions icnale-baseline-more-drop.fce.predictions 1 2 fce 5 "images/output_ets_icnale_5.png"

python src/evaluation-ensemble.py 2 ell-baseline.fce.predictions icnale-baseline-more-drop.fce.predictions 1 2 fce 5 "images/output_ell_icnale_5.png"

python src/evaluation-ensemble.py 2 ets-pretrain.fce.predictions ell-baseline.fce.predictions 1 2 fce 5 "images/output_ets_ell_5.png"

python src/evaluation-ensemble.py 3 ell-baseline.fce.predictions ets-pretrain.fce.predictions icnale-baseline.predictions 1 2 2 fce 5 "images/output_ets_ell_icnale_5.png"


python src/evaluation-ensemble.py 2 "PARAMS AND PREDICTIONS FROM BEFORE/hierarchical-baseline.fce.predictions" icnale-baseline-more-drop.fce.predictions 1 2 fce 10 "images/output_iter_icnale_10.png" "images/output_iter_icnale_10-hist.png" "Ensemble Model Confusion Matrix"

python src/evaluation-ensemble.py 2 "PARAMS AND PREDICTIONS FROM BEFORE/hierarchical-baseline.fce.predictions" icnale-baseline-more-drop.fce.predictions 1 2 fce 5 "images/output_iter_icnale_5.png"