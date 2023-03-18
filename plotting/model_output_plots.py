import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_model_output(models= ['baseline', 'simpler'],
                      metric='r2'):
    loss_list = []
    for model in models:
        losses = pd.read_csv(f'/home/ubuntu/nlp-toefl-autograder/tuning/speech/{model}/losses.csv', index_col=0)
        pred_cols = [col for col in losses.columns if 'pred' in col]
        df = losses.loc[losses['metric'] == metric, pred_cols]
        df['model'] = 'Granular-Output Model' if model=='simpler' else 'Sentence-Level Model'
        loss_list.append(df)

    cols = [c.replace('pred_', '').capitalize() for c in pred_cols]
    df = pd.concat(loss_list)
    df.columns = cols + ['model']
    print(metric)
    metric = ('Test R-squared' if metric=='r2' else metric)
    df = df.melt(id_vars='model', var_name='Category', value_name=metric)
    sns.set(rc={'figure.figsize':(6,3)})
    sns.set_theme(palette='pastel', style='white')
    sns.barplot(x = 'Category', y=metric, 
               hue = 'model',data=df)
    plt.title(f'{metric} by Speech Sub-Score Categories')
    # plt.legend([], [], frameon=False)
    plt.legend(loc=(0.25, -0.5))
    # save figure
    plt.savefig(f'/home/ubuntu/nlp-toefl-autograder/images/speech_{metric}.png',
                bbox_inches='tight', pad_inches=0.1, dpi=300)



if __name__ =='__main__':
    plot_model_output()