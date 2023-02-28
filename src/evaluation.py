import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("  $ python src/evaluation.py <nlp_preds> <fce_data> <written_speech>")
        sys.exit(0)

    print(f"Evaluating {sys.argv[1]}")
    if sys.argv[3] == "written":
        nlp_preds = pd.read_csv(os.path.join(os.getcwd(), sys.argv[1]), header=None)
        nlp_preds.columns = ["preds", "overall_score"]
    else:
        nlp_preds = pd.read_csv(os.path.join(os.getcwd(), sys.argv[1]))
    fce = pd.read_csv(os.path.join(os.getcwd(), sys.argv[2]))
    nlp_preds["sortkey"] = fce.sortkey
    if sys.argv[3] == "written":
        nlp_preds = nlp_preds.groupby("sortkey").mean().reset_index(drop=True)
    # TODO: use training set mean here instead of prediction mean
    adjusted_preds = nlp_preds.preds - nlp_preds.preds.mean() + nlp_preds.overall_score.mean()

    rmse = np.sqrt(mean_squared_error(nlp_preds.overall_score, adjusted_preds))
    r2 = r2_score(nlp_preds.overall_score, adjusted_preds)

    print(
        f"RMSE for NLP model: {round(rmse, 4)}. R-squared for NLP model: {round(r2, 4)}"
    )
