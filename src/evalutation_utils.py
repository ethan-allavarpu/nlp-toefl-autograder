from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_bucket(score, quantiles):
    for q in quantiles:
        if score <= q:
            return q
    return 100

def bucket_values(x, n_buckets=5, custom_range=None):
    if custom_range is not None:
        return pd.cut(x, custom_range, labels=False, include_lowest=True).tolist()
    x_quantiles = int(100 / n_buckets) * np.arange(1, n_buckets)
    if n_buckets == 5:
        x_quantiles = np.arange(60, 100, step=10)
    elif n_buckets == 10:
        x_quantiles = np.arange(55, 100, step=5)
    return [get_bucket(val, x_quantiles) for val in x]

def get_confusion_matrix(preds, actual, n_buckets, custom_range=None):
    bucket_preds = bucket_values(preds, n_buckets, custom_range=custom_range)
    bucket_scores = bucket_values(actual, n_buckets, custom_range=custom_range)
    return confusion_matrix(bucket_scores, bucket_preds)

def get_performance(confusion_matrix, n_buckets):
    print(confusion_matrix)
    within_n = 0
    for i in range(confusion_matrix.shape[0]):
        within_n += confusion_matrix[i, i]
    print(f"Percent of Predictions in correct bucket: {np.round(within_n / confusion_matrix.sum(), 2)}")

    within_n = 0
    for i in range(confusion_matrix.shape[0]):
        for val in range(i - 1, i + 2):
            if val >= 0 and val < confusion_matrix.shape[0]:
                within_n += confusion_matrix[i, val]
    print(f"Percent of Predictions within +/- 1 bucket: {np.round(within_n / confusion_matrix.sum(), 2)}")

    within_n = 0
    for i in range(confusion_matrix.shape[0]):
        for val in range(i - 2, i + 3):
            if val >= 0 and val < confusion_matrix.shape[0]:
                within_n += confusion_matrix[i, val]
    print(f"Percent of Predictions within +/- 2 buckets: {np.round(within_n / confusion_matrix.sum(), 2)}")

def save_heatmap(conf_mat, image_filepath, labs=None, title="Confusion Matrix"):
    plt.figure(figsize = (6,4))
    ax = sn.heatmap(conf_mat, annot=True, fmt='g')
    ax.set_yticks(np.arange(conf_mat.shape[0] + 1), labels=labs)
    ax.set_xticks(np.arange(conf_mat.shape[0] + 1), labels=labs)
    plt.xlabel("Predicted")
    plt.ylabel("Expected")
    plt.yticks(rotation=0)
    plt.title(title)
    plt.savefig(image_filepath, dpi=300)

def save_histogram(preds, overall, image_filepath, title=""):
    plt.figure(figsize = (6,4))
    sn.set_palette("dark")
    differences = preds - overall
    sn.histplot(differences, bins=np.arange(-50, 51, 3))
    plt.xlabel("Predicted - Expected")
    plt.title(title)
    plt.savefig(image_filepath, dpi=300)
