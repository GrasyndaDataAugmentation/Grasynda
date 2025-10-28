import os
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
from src.qgraph_ts import Grasynda

N_QUANTILES = 25
ENSEMBLE_SIZE = 50
RESULTS_DIR = 'assets/results/csv/{ds}-{group},{model}.csv'

for data_name, group in DATA_GROUPS:
    # data_name, group = DATA_GROUPS[0]
    print(data_name, group)

    for model in [*MODELS]:

        fp = RESULTS_DIR.format(ds=data_name, group=group, model=model)

        if os.path.exists(fp):
            continue

        # MODEL = 'NHITS'

        # LOADING DATA AND SETUP
        data_loader = DATASETS[data_name]
        min_samples = data_loader.min_samples[group]
        print('min_samples', min_samples)
        df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group,
                                                                              min_n_instances=min_samples)

        print(df['unique_id'].value_counts())
        print(df.shape)

        # DATA SPLITS
        train, test = LoadDataset.train_test_split(df, horizon)

        # AUGMENTATION PARAMETERS
        max_len = df['unique_id'].value_counts().max() - (2 * horizon)
        min_len = df['unique_id'].value_counts().min() - (2 * horizon)
        n_uids = df['unique_id'].nunique()
        max_n_uids = int(np.round(np.log(n_uids), 0))
        max_n_uids = 2 if max_n_uids < 2 else max_n_uids

        input_data = {'input_size': n_lags, 'h': horizon}

        augmentation_params = {
            'seas_period': freq_int,
            'max_n_uids': max_n_uids,
            'max_len': max_len,
            'min_len': min_len,
        }

        # AUGMENTED TRAIN SETS

        training_sets = {}
        for tsgen in SYNTH_METHODS:
            print(tsgen)
            train_df_ = ExpWorkflow.get_offline_augmented_data(train_=train,
                                                               generator_name=tsgen,
                                                               augmentation_params=augmentation_params,
                                                               n_series_by_uid=1)

            training_sets[tsgen] = train_df_

        training_sets['original'] = train.copy()

   


        qgtse_gen = Grasynda(n_quantiles=N_QUANTILES,
                             quantile_on='remainder',
                             period=freq_int,
                             ensemble_size=ENSEMBLE_SIZE,
                             ensemble_transitions=True)

        qgtse_df = qgtse_gen.transform(train)

    
        train_qgtse = pd.concat([train, qgtse_df]).reset_index(drop=True)
        training_sets['GrasyndaE'] = train_qgtse

#SAVING
        save_dir = os.path.join("assets", "results", "training_sets")
        os.makedirs(save_dir, exist_ok=True)

        for name, df_out in training_sets.items():
            save_path = os.path.join(save_dir, f"{data_name}_{group}_{name}.csv")
            df_out.to_csv(save_path, index=False)
            print(f"✅ Saved training set: {save_path}")
