import pandas as pd

predictions_1 = pd.read_csv("/home/ubuntu/nlp-toefl-autograder/speechocean-simpler.predictions")
predictions_2 = pd.read_csv("/home/ubuntu/nlp-toefl-autograder/speechocean-alpha2.predictions")

pred_cols = [c for c in predictions_1.columns if 'pred' in c]


predictions_1[pred_cols] = 0.5*predictions_1[pred_cols] + 0.5 * predictions_2[pred_cols]

predictions_1.to_csv("/home/ubuntu/nlp-toefl-autograder/speechocean-ensemble.predictions", index=False)


