import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pathlib
tuning = pathlib.Path("tuning")

files = list(tuning.rglob("*"))

df = pd.DataFrame()

for file_path in files:
    file_path = str(file_path)
    if "all-losses.txt" in file_path:
        with open(file_path, 'r+') as f:
            content = f.read().strip() # read content from file and remove whitespaces around
            tuples = eval(content)
            df_temp = pd.DataFrame(tuples, columns=["loss", "val_loss"])
            df_temp["model"] = [file_path.split("/")[-3] for i in range(len(df_temp))]
            df_temp["trial"] = [file_path.split("/")[-2] for i in range(len(df_temp))]
            df = pd.concat([df, df_temp], axis=0)



"plot the val loss column for each model"
for model in df.model.unique():
    df_temp = df[df.model == model]

    for trial in df_temp.trial.unique():
        plt.clf()
        df_temp2 = df_temp[df_temp.trial == trial]
        #df_temp = df_temp.sort_values(by=['val_loss'])
        
        if len(df_temp2["val_loss"]) > 0 and len(df_temp2["loss"])>0:
            df_temp2.plot()
            "make the plot have a log scale on the y axis"
            #plt.yscale("log")
            plt.title(model + " " + trial)
        plt.show()
        # save fig
        plt.savefig(f'images/{model}_{trial}.png')