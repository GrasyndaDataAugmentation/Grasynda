import copy
from typing import Dict
from itertools import combinations

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL

from metaforecast.synth.generators.base import SemiSyntheticGenerator


class Grasynda(SemiSyntheticGenerator):

    def __init__(self,
                 n_quantiles: int,
                 quantile_on: str,
                 period: int,
                 ensemble_transitions: bool,
                 ensemble_size: int = 5,
                 robust: bool = False):

        super().__init__(alias='Grasynda')

        self.n_quantiles = n_quantiles
        # todo qunatile on original does not work
        self.quantile_on = quantile_on
        self.transition_mats = {}
        self.period = period
        self.robust = robust
        self.ensemble_transitions = ensemble_transitions
        self.uid_pw_distance = {}
        self.ensemble_size = ensemble_size
        self.ensemble_transition_mats = {}

    def matrix_to_edgelist(self, uid: str, threshold=0):

        adj_matrix = self.transition_mats[uid]

        # mask for edges above threshold
        mask = adj_matrix > threshold

        from_idx, to_idx = np.nonzero(mask)

        weights = adj_matrix[from_idx, to_idx]

        edge_list = pd.DataFrame({
            'from': from_idx,
            'to': to_idx,
            'weight': weights
        })

        return edge_list

    def transform(self, df: pd.DataFrame, **kwargs):

        # df2 = self.diff(df)

        df_ = df.copy()

        df_ = self.decompose_tsd(df_, period=self.period, robust=self.robust)

        df_['Quantile'] = self._get_quantiles(df_)

        self._calc_transition_matrix(df_)
        if self.ensemble_transitions:
            self.ensemble_transition_mats = self._get_ensemble_transition_mats()

        synth_ts_dict = self._create_synthetic_ts(df_)

        synth_df = self._postprocess_df(df_, synth_ts_dict)

        # df2 = self.undo_diff(df)

        return synth_df

    def _postprocess_df(self, df: pd.DataFrame, synth_ts: Dict):
        synth_list = []
        for uid, uid_df in df.groupby('unique_id'):
            uid_df[self.quantile_on] = synth_ts[uid].values
            synth_list.append(uid_df)

        synth_df = pd.concat(synth_list)

        synth_df['y'] = synth_df[['trend', 'seasonal', 'remainder']].sum(axis=1)
        synth_df = synth_df.drop(columns=['trend', 'seasonal', 'remainder', 'Quantile'])

        synth_df['unique_id'] = synth_df['unique_id'].apply(lambda x: f'{self.alias}_{x}')
        synth_df = synth_df[['ds', 'unique_id', 'y']]

        return synth_df

    def _create_synthetic_ts(self, df: pd.DataFrame) -> Dict:
        quantile_series = self._generate_quantile_series(df)

        generated_time_series = {}

        uids = df['unique_id'].unique().tolist()
        for uid in uids:
            uid_df = df.query(f'unique_id=="{uid}"')
            uid_s = uid_df[self.quantile_on]
            uid_quantiles = uid_df['Quantile']

            uid_q_vals = {q: uid_s[uid_quantiles == q].values for q in range(self.n_quantiles)}

            synth_ts = np.zeros(len(uid_s))
            synth_ts[0] = uid_s.values[0]

            for i in range(1, len(uid_quantiles)):
                current_quantile = quantile_series[uid][i]

                if len(uid_q_vals[current_quantile]) > 0:
                    synth_ts[i] = np.random.choice(uid_q_vals[current_quantile])
                else:
                    synth_ts[i] = np.nan

            generated_time_series[uid] = pd.Series(synth_ts, index=uid_df.index)

        return generated_time_series

    def _generate_quantile_series(self, df: pd.DataFrame):
        uids = df['unique_id'].unique().tolist()

        quantile_series = {}
        for uid in uids:
            if self.ensemble_transitions:
                transition_mat = self.ensemble_transition_mats[uid]
            else:
                transition_mat = self.transition_mats[uid]

            uid_df = df.query(f'unique_id=="{uid}"')

            series = uid_df[self.quantile_on]
            q_series = np.zeros(len(series), dtype=int)

            q_series[0] = uid_df['Quantile'].values[0]  # starts with 1st q

            for t in range(1, len(q_series)):
                current_quantile = q_series[t - 1]
                probs = transition_mat[current_quantile]
                if np.sum(probs) == 0 or np.isnan(np.sum(probs)):
                    probs = np.ones(self.n_quantiles) / self.n_quantiles
                else:
                    probs = probs / np.sum(probs)
                next_quantile = np.random.choice(np.arange(self.n_quantiles), p=probs)
                q_series[t] = next_quantile

            quantile_series[uid] = q_series

        return quantile_series

    def _get_ensemble_transition_mats(self):
        mats = copy.deepcopy(self.transition_mats)

        uid_pairs = combinations([*mats], 2)

        for uid in mats:
            self.uid_pw_distance[(uid, uid)] = 0.0

        for uid1, uid2 in uid_pairs:
            mat1 = mats[uid1]
            mat2 = mats[uid2]
            dist = np.linalg.norm(mat1 - mat2)

            self.uid_pw_distance[(uid1, uid2)] = dist
            self.uid_pw_distance[(uid2, uid1)] = dist

        ensemble_mats = {}
        for uid in mats:
            uid_dists = pd.Series(
                {other_uid: self.uid_pw_distance[(uid, other_uid)]
                 for other_uid in mats})

            similar_uids = uid_dists.sort_values().head(self.ensemble_size).index.tolist()

            avg_mat = np.sum(
                mats[uid]
                for uid in similar_uids
            ) / self.ensemble_size

            ensemble_mats[uid] = avg_mat

        return ensemble_mats

    def _calc_transition_matrix(self, df: pd.DataFrame):
        assert 'Quantile' in df.columns

        for unique_id, group in df.groupby('unique_id'):
            quantiles = group['Quantile'].values

            t_count_matrix = np.zeros((self.n_quantiles, self.n_quantiles))

            # Loop through the quantiles and count transitions
            for i in range(len(quantiles) - 1):
                current_quantile = quantiles[i]
                next_quantile = quantiles[i + 1]
                t_count_matrix[current_quantile, next_quantile] += 1

            t_prob_matrix = t_count_matrix / t_count_matrix.sum(axis=1, keepdims=True)
            t_prob_matrix = np.nan_to_num(t_prob_matrix)

            # rows where the sum is zero
            for row in range(self.n_quantiles):
                if np.sum(t_count_matrix[row]) == 0:
                    t_prob_matrix[row] = np.ones(self.n_quantiles) / self.n_quantiles
                else:
                    t_prob_matrix[row] = t_count_matrix[row] / np.sum(t_count_matrix[row])

            self.transition_mats[unique_id] = t_prob_matrix

        return self.transition_mats

    def _get_quantiles(self, df: pd.DataFrame):
        assert self.quantile_on in df.columns

        quantiles = df.groupby('unique_id')[self.quantile_on].transform(
            lambda x: pd.qcut(x, self.n_quantiles, labels=False, duplicates='drop')
        )

        return quantiles

    @staticmethod
    def decompose_tsd(df: pd.DataFrame, period: int, robust: bool):
        seasonal_components = []
        trend_components = []
        remainder_components = []

        for unique_id, group in df.groupby('unique_id'):
            stl = STL(group['y'], period=period, robust=robust)
            result = stl.fit()

            seasonal_components.append(pd.DataFrame({
                'unique_id': unique_id,
                'ds': group['ds'],
                'seasonal': result.seasonal
            }))

            trend_components.append(pd.DataFrame({
                'unique_id': unique_id,
                'ds': group['ds'],
                'trend': result.trend
            }))

            remainder_components.append(pd.DataFrame({
                'unique_id': unique_id,
                'ds': group['ds'],
                'remainder': result.resid
            }))

        seasonal_df = pd.concat(seasonal_components)
        trend_df = pd.concat(trend_components)
        remainder_df = pd.concat(remainder_components)

        decomposed_df = pd.merge(seasonal_df, trend_df, on=['unique_id', 'ds'])
        decomposed_df = pd.merge(decomposed_df, remainder_df, on=['unique_id', 'ds'])

        return decomposed_df


class GrasyndaDiff(SemiSyntheticGenerator):
    def __init__(self, n_quantiles: int, ensemble_transitions: bool, ensemble_size: int = 0):

        super().__init__(alias='GrasyndaDiff')
        self.n_quantiles = n_quantiles
        self.ensemble_transitions = ensemble_transitions
        self.ensemble_size = ensemble_size
        self.transition_mats = {}
        self.ensemble_transition_mats = {}
        self.uid_pw_distance = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ = df.copy()

        df_diff = self._difference_series(df_)

        df_diff['Quantile'] = self._get_quantiles(df_diff)

        self._calc_transition_matrix(df_diff)
        if self.ensemble_transitions:
            self.ensemble_transition_mats = self._get_ensemble_transition_mats()

        synth_ts_dict = self._create_synthetic_ts(df_diff)
        synth_df = self._integrate_series(df_diff, synth_ts_dict)

        return synth_df

    def _difference_series(self, df: pd.DataFrame) -> pd.DataFrame:

        df_diff = df.copy()
        df_diff['diff'] = df.groupby('unique_id')['y'].diff()
        df_diff.dropna(subset=['diff'], inplace=True)
        return df_diff

    def _get_quantiles(self, df: pd.DataFrame):
        assert 'diff' in df.columns

        quantiles = df.groupby('unique_id')['diff'].transform(
            lambda x: pd.qcut(x, self.n_quantiles, labels=False, duplicates='drop')
        )

        return quantiles

    def _calc_transition_matrix(self, df: pd.DataFrame):
        assert 'Quantile' in df.columns

        for unique_id, group in df.groupby('unique_id'):
            quantiles = group['Quantile'].values

            t_count_matrix = np.zeros((self.n_quantiles, self.n_quantiles))

            for i in range(len(quantiles) - 1):
                current_quantile = quantiles[i]
                next_quantile = quantiles[i + 1]
                t_count_matrix[current_quantile, next_quantile] += 1

            t_prob_matrix = t_count_matrix / t_count_matrix.sum(axis=1, keepdims=True)
            t_prob_matrix = np.nan_to_num(t_prob_matrix)

            for row in range(self.n_quantiles):
                if np.sum(t_count_matrix[row]) == 0:
                    t_prob_matrix[row] = np.ones(self.n_quantiles) / self.n_quantiles

            self.transition_mats[unique_id] = t_prob_matrix

        return self.transition_mats

    def _create_synthetic_ts(self, df: pd.DataFrame) -> Dict:
        quantile_series = self._generate_quantile_series(df)

        generated_time_series = {}

        uids = df['unique_id'].unique().tolist()
        for uid in uids:
            uid_df = df.query(f'unique_id=="{uid}"')
            uid_s = uid_df['diff']
            uid_quantiles = uid_df['Quantile']

            uid_q_vals = {q: uid_s[uid_quantiles == q].values for q in range(self.n_quantiles)}

            synth_ts = np.zeros(len(uid_s))
            synth_ts[0] = uid_s.values[0]

            for i in range(1, len(uid_quantiles)):
                current_quantile = quantile_series[uid][i]

                if len(uid_q_vals[current_quantile]) > 0:
                    synth_ts[i] = np.random.choice(uid_q_vals[current_quantile])
                else:
                    synth_ts[i] = np.nan

            generated_time_series[uid] = pd.Series(synth_ts, index=uid_df.index)

        return generated_time_series

    def _get_ensemble_transition_mats(self):
        mats = copy.deepcopy(self.transition_mats)

        uid_pairs = combinations([*mats], 2)

        for uid in mats:
            self.uid_pw_distance[(uid, uid)] = 0.0

        for uid1, uid2 in uid_pairs:
            mat1 = mats[uid1]
            mat2 = mats[uid2]
            dist = np.linalg.norm(mat1 - mat2)

            self.uid_pw_distance[(uid1, uid2)] = dist
            self.uid_pw_distance[(uid2, uid1)] = dist

        ensemble_mats = {}
        for uid in mats:
            uid_dists = pd.Series(
                {other_uid: self.uid_pw_distance[(uid, other_uid)]
                 for other_uid in mats})

            similar_uids = uid_dists.sort_values().head(self.ensemble_size).index.tolist()

            avg_mat = np.sum(
                mats[uid]
                for uid in similar_uids
            ) / self.ensemble_size

            ensemble_mats[uid] = avg_mat

        return ensemble_mats

    def _integrate_series(self, df: pd.DataFrame, synth_diff_dict: Dict) -> pd.DataFrame:
        """
        Integrate the synthetic differenced series to reconstruct the original scale.
        """
        synth_list = []
        for uid, group in df.groupby('unique_id'):
            original_y = group['y'].iloc[0]
            synth_diff = synth_diff_dict[uid]

            synthetic_y = np.cumsum(np.insert(synth_diff.values, 0, original_y))[:-1]

            group['y'] = synthetic_y
            group['y'] = group['y'].fillna(method='ffill')
            synth_list.append(group)

        synth_df = pd.concat(synth_list)
        synth_df['unique_id'] = synth_df['unique_id'].apply(lambda x: f'{self.alias}_{x}')
        synth_df = synth_df[['ds', 'unique_id', 'y']]
        synth_df['y'] = synth_df['y'].fillna(method='ffill')
        return synth_df[['ds', 'unique_id', 'y']]

    def _generate_quantile_series(self, df: pd.DataFrame):
        uids = df['unique_id'].unique().tolist()

        quantile_series = {}
        for uid in uids:
            if self.ensemble_transitions:
                transition_mat = self.ensemble_transition_mats[uid]
            else:
                transition_mat = self.transition_mats[uid]

            uid_df = df.query(f'unique_id=="{uid}"')

            series = uid_df['diff']
            q_series = np.zeros(len(series), dtype=int)

            q_series[0] = uid_df['Quantile'].values[0]  # starts with 1st q

            for t in range(1, len(q_series)):
                current_quantile = q_series[t - 1]
                probs = transition_mat[current_quantile]
                if np.sum(probs) == 0 or np.isnan(np.sum(probs)):
                    probs = np.ones(self.n_quantiles) / self.n_quantiles
                else:
                    probs = probs / np.sum(probs)
                next_quantile = np.random.choice(np.arange(self.n_quantiles), p=probs)
                q_series[t] = next_quantile

            quantile_series[uid] = q_series

        return quantile_series
