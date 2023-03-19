import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import sys
from evalutation_utils import get_confusion_matrix, get_performance, save_heatmap, save_histogram

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=False, default="/home/ubuntu/nlp-toefl-autograder/tuning/speech/baseline")
    parser.add_argument('--n_buckets', type=int, required=False, default=12)
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()

    path = args.path
    predictions_df_list = []
    df_list = []
    for seed in range(5):
        path_df = path + "/seed_" + str(seed) + f"/trial_00000/speechocean-{'train' if args.train else 'best'}.predictions"
        df = pd.read_csv(path_df)
        pred_cols = [col for col in df.columns if "pred" in col]
        actual_cols = [col for col in df.columns if "actual" in col]
        losses_dict = {}
        for col in pred_cols:
            rmse = np.sqrt(mean_squared_error(df[col], df[actual_cols[pred_cols.index(col)]]))
            r2 = r2_score(df[col], df[actual_cols[pred_cols.index(col)]])
            losses_dict[col] = {'rmse': rmse, 'r2': r2}
        losses_df = pd.DataFrame(losses_dict)
        losses_df['seed'] = seed
        df_list.append(losses_df)
        predictions_df_list.append(df)

    losses_df = pd.concat(df_list).reset_index().rename(columns={'index': 'metric'})
    # get mean r2 and rmse
    losses_df = losses_df.groupby(['metric']).mean().reset_index()

    losses_df.to_csv(path + f"/{'train_' if args.train else ''}losses.csv")

    if not args.train:
        # get correlation between the two predictions
        nlp_preds = pd.concat(predictions_df_list)
        conf_mat = get_confusion_matrix(nlp_preds['pred_total'], nlp_preds['actual_total'], args.n_buckets, custom_range = np.arange(-5, 106, 10))
        # labs=[f"[{max(n-10, 0)},{min(n, 100)}{')' if n<100 else ']'}" for n in np.arange(5, 106, 10)]
        save_heatmap(conf_mat, path + "/confusion_matrix.png", labs = [0] + [min(x, 100) for x in np.arange(5, 106, 10)],
                     title = "Granular-Output Speech Model Confusion Matrix")
        get_performance(conf_mat, args.n_buckets)
        
        r = np.corrcoef(nlp_preds['pred_total'], nlp_preds['actual_total'])[0, 1]
        print(f"Correlation (r) between the two predictions: {np.round(r, 4)}")
        save_histogram(nlp_preds['pred_total'], nlp_preds['actual_total'], path + "/histogram.png", 
                       title="Histogram of Prediction Errors")