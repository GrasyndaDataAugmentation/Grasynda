import os

import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate
from functools import partial

from utils.load_data.config import DATASETS
from utils.load_data.config import DATA_GROUPS
from utils.config import MODEL_CONFIG, MODELS
from utils.load_data.base import LoadDataset
from src.qgraph_ts import Grasynda

N_QUANTILES_LIST = [3, 5, 7, 10, 15, 20, 25, 50, 75, 100]
ENSEMBLE_SIZE = 50
RESULTS_DIR = 'assets/results/csv/sensitivity-{ds}-{group},{model}.csv'

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
        for n_quants in N_QUANTILES_LIST:
            print(n_quants)
            gsd = Grasynda(n_quantiles=n_quants,
                           quantile_on='remainder',
                           period=freq_int,
                           ensemble_transitions=False)

            synth_df = gsd.transform(train)

            train_augmented = pd.concat([train, synth_df]).reset_index(drop=True)

            training_sets[f'Grasynda({n_quants})'] = train_augmented

        training_sets['Original'] = train.copy()

        # MODELING

        test_with_fcst = test.copy()
        for tsgen, train_df_ in training_sets.items():
            model_params = MODEL_CONFIG.get(model)
            model_conf = {**input_data, **model_params}
            # model_conf['max_steps']=300
            # model_conf['accelerator']='cpu'

            if model.startswith('Auto'):
                # model_conf = {'h': horizon, 'num_samples':2}
                model_conf = {'h': horizon}

            nf = NeuralForecast(models=[MODELS[model](**model_conf, alias=tsgen)], freq=freq_str)
            nf.fit(df=train_df_, val_size=horizon)

            fcst = nf.predict()

            test_with_fcst = test_with_fcst.merge(fcst.reset_index(), on=['unique_id', 'ds'], how="left")

        # EVALUATION
        evaluation_df = evaluate(test_with_fcst, [partial(mase, seasonality=freq_int), smape], train_df=train)

        evaluation_df.to_csv(fp, index=False)

        print(evaluation_df.query('metric=="mase"').mean(numeric_only=True).sort_values())
        print(evaluation_df.query('metric=="smape"').mean(numeric_only=True).sort_values())
        print(evaluation_df.mean(numeric_only=True).sort_values())
