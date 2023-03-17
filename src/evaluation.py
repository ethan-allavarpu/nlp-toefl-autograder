import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys
from evalutation_utils import get_confusion_matrix, get_performance, save_heatmap, save_histogram

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage:")
        print("  $ python src/evaluation.py <nlp_preds> <eval_set> <model_type> <n_buckets> <output_image_path> <hist_filepath> <plot_title>")
        sys.exit(0)
    print(f"Evaluating {sys.argv[1]}")
    nlp_preds = pd.read_csv(os.path.join(os.getcwd(), sys.argv[1]), header=None)
    nlp_preds.columns = ["preds", "overall_score"]
    if sys.argv[2] == "fce":
        fce = pd.read_csv("data/fce-data-input-format.csv")
        nlp_preds["sortkey"] = fce.sortkey
        nlp_preds = nlp_preds.groupby("sortkey").mean().reset_index(drop=True)
        model_type = sys.argv[3]
        n_buckets = int(sys.argv[4])
        image_filepath = sys.argv[5]
        hist_path = sys.argv[6]
        if model_type == "hierarchical":
            nlp_preds.preds = nlp_preds.preds * np.std(nlp_preds.overall_score, ddof=1)    
        # TODO: use training set mean here instead of prediction mean
        adjusted_preds = nlp_preds.preds - nlp_preds.preds.mean() + nlp_preds.overall_score.mean()

        rmse = np.sqrt(mean_squared_error(nlp_preds.overall_score, adjusted_preds))
        r2 = r2_score(nlp_preds.overall_score, adjusted_preds)
        r = np.corrcoef(nlp_preds.overall_score, adjusted_preds)[0, 1]
    else:
        rmse = np.sqrt(mean_squared_error(nlp_preds.overall_score, nlp_preds.preds))
        r2 = r2_score(nlp_preds.overall_score, nlp_preds.preds)
        r = np.corrcoef(nlp_preds.overall_score, nlp_preds.preds)[0, 1]

    print(
        f"RMSE for NLP model: {round(rmse, 4)}. R-squared for NLP model: {round(r2, 4)}"
    )

    conf_mat = get_confusion_matrix(adjusted_preds, nlp_preds.overall_score, n_buckets)
    if n_buckets == 5:
        x_quantiles = [0] + list(np.arange(60, 101, step=10))
    elif n_buckets == 10:
        x_quantiles = [0] + list(np.arange(55, 101, step=5))
    save_heatmap(conf_mat, image_filepath, labs=x_quantiles, title=sys.argv[7])
    save_histogram(adjusted_preds, nlp_preds.overall_score, hist_path)
    get_performance(conf_mat, n_buckets)
    

    print(f"Correlation (r) between the two predictions: {np.round(r, 4)}")
