import os

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

        # Derivation method

        derived_gen = DerivedGen(n_quantiles=N_QUANTILES, ensemble_transitions=False)
        derived_gen_e = DerivedGen(n_quantiles=N_QUANTILES, ensemble_transitions=True, ensemble_size=ENSEMBLE_SIZE)

        derived_gen = derived_gen.transform(train)
        derived_gen_ensemble_df = derived_gen_e.transform(train)

        train_derived = pd.concat([train, derived_gen]).reset_index(drop=True)
        train_derived_e = pd.concat([train, derived_gen_ensemble_df]).reset_index(drop=True)

        training_sets['derived'] = train_derived
        training_sets['derived_ensemble'] = train_derived_e

        # QGTS
        qgts_gen = Grasynda(n_quantiles=N_QUANTILES,
                            quantile_on='remainder',
                            period=freq_int,
                            ensemble_transitions=False)

        qgtse_gen = Grasynda(n_quantiles=N_QUANTILES,
                             quantile_on='remainder',
                             period=freq_int,
                             ensemble_size=ENSEMBLE_SIZE,
                             ensemble_transitions=True)

        qgts_df = qgts_gen.transform(train)
        qgtse_df = qgtse_gen.transform(train)

        train_qgts = pd.concat([train, qgts_df]).reset_index(drop=True)
        train_qgtse = pd.concat([train, qgtse_df]).reset_index(drop=True)

        training_sets['QGTS'] = train_qgts
        training_sets['QGTSE'] = train_qgtse

        # MODELING

        test_with_fcst = test.copy()
        for tsgen, train_df_ in training_sets.items():
            model_params = MODEL_CONFIG.get(model)
            model_conf = {**input_data, **model_params}
            if model.startswith("Auto"):
                # model_conf = {'h': horizon, 'num_samples':2}
                model_conf = {'h': horizon}

            nf = NeuralForecast(models=[MODELS[model](**model_conf, alias=tsgen)], freq=freq_str)
            nf.fit(df=train_df_, val_size=horizon)

            fcst = nf.predict()

            test_with_fcst = test_with_fcst.merge(fcst.reset_index(), on=['unique_id', 'ds'], how="left")

        # BASELINE
        stats_models = [SeasonalNaive(season_length=freq_int)]
        sf = StatsForecast(models=stats_models, freq=freq_str, n_jobs=1)
        sf.fit(train)
        sf_fcst = sf.predict(h=horizon)
        test_with_fcst = test_with_fcst.merge(sf_fcst.reset_index(), on=['unique_id', 'ds'], how="left")

        # EVALUATION
        evaluation_df = evaluate(test_with_fcst, [partial(mase, seasonality=freq_int), smape], train_df=train)

        evaluation_df.to_csv(fp, index=False)

        print(evaluation_df.query('metric=="mase"').mean(numeric_only=True).sort_values())
        print(evaluation_df.query('metric=="smape"').mean(numeric_only=True).sort_values())
        print(evaluation_df.mean(numeric_only=True).sort_values())
