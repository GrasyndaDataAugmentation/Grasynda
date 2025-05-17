from utils.load_data.config import DATASETS
from utils.load_data.config import DATA_GROUPS
from utils.load_data.base import LoadDataset
from src.qgraph_ts import Grasynda

data_name, group = DATA_GROUPS[0]
print(data_name, group)
N_QUANTILES = 25

# LOADING DATA AND SETUP
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group,
                                                                      min_n_instances=min_samples)

print(df['unique_id'].value_counts())
print(df.shape)

# DATA SPLITS
train, _ = LoadDataset.train_test_split(df, horizon)

# QGTS
qgts_gen = Grasynda(n_quantiles=N_QUANTILES,
                    quantile_on='remainder',
                    period=freq_int,
                    ensemble_transitions=False)

qgts_gen.transition_mats['ID90']

qgts_df = qgts_gen.transform(train)

qgts_gen.matrix_to_edgelist('ID90').to_csv('assets/results/plot_data/example.csv', index=False)

df_uid = train.query('unique_id=="ID90"')
# df_uid = qgts_df.query('unique_id=="Grasynda_ID90"')

import pandas as pd
import plotnine as p9
from plotnine.geoms.geom_hline import geom_hline
from numerize import numerize

aes_ = {'x': 'ds', 'y': 'y', 'group': 1}

plot = \
    p9.ggplot(df_uid) + \
    p9.aes(**aes_) + \
    p9.theme_minimal(base_family='Palatino', base_size=12) + \
    p9.theme(plot_margin=.0125,
             axis_text=p9.element_text(size=12),
             legend_title=p9.element_blank(),
             legend_position=None)

plot += p9.geom_line(color='#228B22', size=1)

plot = \
    plot + \
    p9.xlab('') + \
    p9.ylab('') + \
    p9.ggtitle('') + \
    p9.scale_y_continuous(labels=lambda lst: [numerize.numerize(x)
                                              for x in lst])

plot.save('plot_example_syn.pdf', width=7.5, height=4.15)
