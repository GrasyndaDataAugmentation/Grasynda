import sys

sys.path.append(r'C:\Users\lhenr\Desktop\graph_based_time_series_aug')

import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate
from functools import partial

from utils.load_data.config import DATASETS
from utils.load_data.config import DATA_GROUPS
from utils.config import SYNTH_METHODS, MODEL_CONFIG, MODELS
from src.workflow import ExpWorkflow
from utils.load_data.base import LoadDataset
from src.qgraph_ts import QuantileGraphTimeSeriesGenerator as QGTSGen
from src.qgraph_ts import QuantileDerivedTimeSeriesGenerator as DerivedGen
from pytorch_lightning import Trainer


dataset_indices = [0,1,2,3,4,5]

for i in dataset_indices:
    data_name, group = DATA_GROUPS[i]
    print(f"Loading dataset {i}: {data_name}, {group}")
    
    data_loader = DATASETS[data_name]
    min_samples = data_loader.min_samples[group]
    print(f'min_samples: {min_samples}')
    
    globals()[f'df{i}'], horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples
    )

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = []
kan_paths = [
    "assets/results/M3_Monthly_KAN.csv",
    "assets/results/Tourism_Monthly_KAN.csv",
    "assets/results/Gluonts_m1_quarterly_KAN.csv",
    "assets/results/M3_Quarterly_KAN.csv",
    "assets/results/Tourism_Quarterly_KAN.csv",
    "assets/results/Gluonts_m1_monthly_KAN"
]

for i in range(6):
    data_name, group = DATA_GROUPS[i]
    print(f"Loading dataset {i}: {data_name}, {group}")
    
    data_loader = DATASETS[data_name]
    min_samples = data_loader.min_samples[group]
    print(f'min_samples: {min_samples}')
    
    df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(
        group, min_n_instances=min_samples
    )
    datasets.append(df)

def get_lengths(df):
    lengths = df['unique_id'].value_counts().reset_index()
    lengths.columns = ['unique_id', 'length']
    df = df.merge(lengths, on='unique_id', how='left')
    return df[['unique_id', 'length']].drop_duplicates()

length_dfs = []
for i, df in enumerate(datasets):
    lengths = get_lengths(df)
    lengths['dataset'] = f'Dataset{i+1}'
    length_dfs.append(lengths)

kan_dfs = []
for i, path in enumerate(kan_paths):
    kan_df = pd.read_csv(path)
    kan_df = kan_df[kan_df['metric'] == 'mase']
    kan_df['model'] = 'KAN'
    kan_df['dataset'] = f'Dataset{i+1}'
    kan_df = kan_df[['unique_id', 'qgts(25)']].rename(columns={'qgts(25)': 'mase'})
    kan_dfs.append(kan_df)

merged_dfs = []
for i in range(len(datasets)):
    merged = pd.merge(kan_dfs[i], length_dfs[i], on='unique_id', how='left')
    merged_dfs.append(merged)

all_data = pd.concat(merged_dfs)

plt.figure(figsize=(14, 6))
all_data['length_bin'] = pd.qcut(all_data['length'], q=10, duplicates='drop')
bin_intervals = all_data['length_bin'].cat.categories

ax = sns.violinplot(
    data=all_data,
    x='length_bin',
    y='mase',
    color='lightblue',
    inner='quartile',
    cut=0
)

ax.set_xticklabels(
    [f"[{int(i.left)}-{int(i.right)}]" for i in bin_intervals],
    rotation=45,
    ha='right'
)

plt.xlabel("Length Intervals (by Percentile)")
plt.ylabel("MASE")
plt.title("MASE Distribution by Length Percentiles (All Datasets)")
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()