
import os
import sys


sys.path.append(r'C:\Users\lhenr\Desktop\graph_based_time_series_aug')
import pandas as pd
import plotnine as p9
from plotnine.geoms.geom_hline import geom_hline
from numerize import numerize

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

qgts_df = qgts_gen.transform(train)

qgts_gen.transition_mats['ID9']


qgts_gen.matrix_to_edgelist('ID9').to_csv('assets/results/plot_data/example.csv', index=False)

#df_uid = train.query('unique_id=="ID10"')
# df_uid = qgts_df.query('unique_id=="Grasynda_ID90"')


df_original = train.query('unique_id == "ID9"').copy()
df_generated = qgts_df.query('unique_id == "Grasynda_ID9"').copy()

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, theme_minimal, theme, element_text, xlab, ylab,
    ggtitle, scale_y_continuous, scale_color_manual, labs
)
from numerize import numerize



df_original['label'] = 'Original'
df_original['linetype'] = 'solid'

df_generated['label'] = 'Synthetic'
df_generated['linetype'] = 'dashed'

# Combine just for plotting convenience (we won’t break the structure)
df_plot = pd.concat([df_original, df_generated], axis=0)

# Plot with legend, correct colors, and correct linetypes
plot = (
    ggplot(df_plot)
    + aes(x='ds', y='y', color='label', linetype='label')
    + geom_line(size=1)
    + scale_color_manual(values={'Original': '#1f77b4', 'Synthetic': '#228B22'})  # Blue, Green
    + theme_minimal(base_family='Palatino', base_size=12)
    + theme(
        plot_margin=0.0125,
        axis_text=element_text(size=12),
        legend_title=element_text(size=11),
        legend_position='right'
    )
    + xlab('')
    + ylab('')
    + ggtitle('')
    + labs(color='Series Type', linetype='Series Type')
    + scale_y_continuous(labels=lambda lst: [numerize.numerize(x) for x in lst])
)

plot.save('plot_ID9_vs_synthetic_with_legend_fixed.pdf', width=7.5, height=4.15)
