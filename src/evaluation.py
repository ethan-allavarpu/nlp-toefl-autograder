import numpy as np
import os
import pandas as pd
import sys


def get_evaluation_metrics(pred_vals: np.array, true_vals=np.array) -> tuple:
    """
    Calculate RMSE, test R-squared metrics between predictions and true values

    Parameters
    ----------
    pred_vals: np.array
        NumPy array with predictions from a model
    true_vals: np.array
        NumPy array with ground truths

    Returns
    -------
    Tuple of length two: (RMSE, R-squared)
    """
    assert pred_vals.shape[0] == true_vals.shape[0]
    n = pred_vals.shape[0]
    rmse = np.sqrt(((true_vals - pred_vals) ** 2).sum() / n)
    test_r2 = np.corrcoef(pred_vals, true_vals)
    return (rmse, test_r2)


if __name__ == "__main__":
    if (len(sys.argv)) != 3:
        print("Usage:")
        print("  $ python src/evaluation.py <nlp_preds> <fce_data>")
        sys.exit(0)

    nlp_preds = pd.read_csv(os.path.join(os.getcwd(), sys.argv[1]), header=None)
    nlp_preds.columns = ["preds", "overall_score"]
    fce = pd.read_csv(os.path.join(os.getcwd(), sys.argv[2]))
    nlp_preds["sortkey"] = fce.sortkey
    nlp_preds = nlp_preds.groupby("sortkey").mean()

    nlp_rmse, nlp_r2 = get_evaluation_metrics(nlp_preds.preds, nlp_preds.overall_score)

    print(
        f"RMSE for NLP model: {round(nlp_rmse, 4)}. R-squared for NLP model: {round(nlp_r2[0, 1], 4)}"
    )
