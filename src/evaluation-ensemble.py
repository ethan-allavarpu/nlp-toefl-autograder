import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys
from evalutation_utils import get_confusion_matrix, get_performance, save_heatmap, save_histogram

if __name__ == "__main__":
    nmodels = int(sys.argv[1])
    models = [sys.argv[i] for i in range(2, 2 + nmodels)]
    multiplier = [int(sys.argv[i]) for i in range(2 + nmodels, 2 + 2 * nmodels)]
    print(multiplier)
    print(f"Evaluating {', '.join(models)}")
    def get_individual_preds(pred_path, fce):
        nlp_preds = pd.read_csv(os.path.join(os.getcwd(), pred_path), header=None)
        nlp_preds.columns = ["preds", "overall_score"]
        nlp_preds["sortkey"] = fce.sortkey
        nlp_preds = nlp_preds.groupby("sortkey").mean().reset_index(drop=True)
        return (nlp_preds.preds - nlp_preds.preds.mean() + nlp_preds.overall_score.mean(), nlp_preds.overall_score)
    if sys.argv[2 + 2 * nmodels] == "fce":
        fce = pd.read_csv(os.path.join(os.getcwd(), "data/fce-data-input-format.csv"))
    combo = [get_individual_preds(model, fce) for model in models]
    preds = [pred for pred, _, in combo]
    overall = combo[0][1]
    adjusted_preds = np.array([preds[i] * multiplier[i] for i in range(len(preds))]).sum(axis = 0) / sum(multiplier)

    rmse = np.sqrt(mean_squared_error(overall, adjusted_preds))
    r2 = r2_score(overall, adjusted_preds)

    print(
        f"RMSE for NLP model: {round(rmse, 4)}. R-squared for NLP model: {round(r2, 4)}"
    )
    n_buckets = int(sys.argv[3 + 2 * nmodels])
    print(n_buckets)
    image_filepath = sys.argv[4 + 2 * nmodels]
    hist_path = sys.argv[5 + 2 * nmodels]
    plot_title = sys.argv[6 + 2 * nmodels]
    if n_buckets == 5:
        x_quantiles = [0] + list(np.arange(60, 101, step=10))
    elif n_buckets == 10:
        x_quantiles = [0] + list(np.arange(55, 101, step=5))
    conf_mat = get_confusion_matrix(adjusted_preds, overall, n_buckets)
    save_heatmap(conf_mat, image_filepath, labs=x_quantiles, title=plot_title)
    save_histogram(adjusted_preds, overall, hist_path)
    get_performance(conf_mat, n_buckets)

    print(f"Correlation between the two predictions: {np.round(np.corrcoef(overall, adjusted_preds)[0, 1], 4)}")
