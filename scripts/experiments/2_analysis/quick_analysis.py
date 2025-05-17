import os
import re
import pandas as pd

RESULTS_DIR = 'assets/autoresultscsv'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def read_results(metric: str):
    files = os.listdir(RESULTS_DIR)

    results_list = []
    for file in files:
        df_ = pd.read_csv(f'{RESULTS_DIR}/{file}')
        df_['dataset'] = file
        results_list.append(df_)

    res = pd.concat(results_list)
    res = res.query(f'metric=="{metric}"')
    res = res.reset_index(drop=True)
    res['model'] = res['dataset'].apply(lambda x: re.sub('.csv', '', x).split('_')[-1])
    res['ds'] = res['dataset'].apply(lambda x: '_'.join(re.sub('.csv', '', x).split('_')[:-1]))
    res = res.drop(columns=['metric', 'unique_id','dataset'])
    # res['ds'] = res['ds'].map(DS_MAPPER)

    return res


df = read_results('mase')

df.mean(numeric_only=True).sort_values()
df.groupby('ds').mean(numeric_only=True)
df.groupby('ds').mean(numeric_only=True).rank(axis=1).mean().sort_values()


a = df.query('model=="NHITS"').groupby('ds').mean(numeric_only=True)


b=df.query('model=="NHITS"').drop(columns=['model','ds'])
b.rank(axis=1).mean().sort_values()

df.query('model=="NHITS"').groupby('ds').mean(numeric_only=True).rank(axis=1).mean().sort_values()

df.query('model=="NHITS"').drop(columns=['ds','model']).rank(axis=1).mean().sort_values()


import matplotlib.pyplot as plt
df.groupby('ds').mean(numeric_only=True).plot(kind='bar')
plt.title('Mean Performance per Dataset')
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()


