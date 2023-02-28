import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  $ python src/evaluation.py <nlp_preds>")
        sys.exit(0)

    print(f"Evaluating {sys.argv[1]}")
    nlp_preds = pd.read_csv(os.path.join(os.getcwd(), sys.argv[1]))

    pred_cols = [col for col in nlp_preds.columns if "pred" in col]
    actual_cols = [col for col in nlp_preds.columns if "actual" in col]

    for col in pred_cols:
        rmse = np.sqrt(mean_squared_error(nlp_preds[col], nlp_preds[actual_cols[0]]))
        r2 = r2_score(nlp_preds[col], nlp_preds[actual_cols[0]])
        print(f"RMSE for {col}: {round(rmse, 4)}. R-squared for {col}: {round(r2, 4)}")

