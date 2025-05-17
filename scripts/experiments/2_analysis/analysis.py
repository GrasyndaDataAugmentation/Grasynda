import pandas as pd
import plotnine as p9

from utils.analysis import to_latex_tab, read_results, THEME

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

df = read_results('mase')
df = df.drop(columns=['derived_ensemble', 'derived'])


def add_model_averages_and_ranks(df):
    # Calculate ranks for each metric (lower values are better)
    # Using axis=1 to rank across columns for each row
    rank_df = df.rank(axis=1, ascending=True)

    # Create a new DataFrame to store the result
    result_dfs = []

    # Process each model separately to keep data together
    for model in df.index.get_level_values('model').unique():
        # Extract model data
        model_data = df.xs(model, level='model')
        model_ranks = rank_df.xs(model, level='model')

        # Calculate averages
        model_avg = model_data.mean()
        model_rank_avg = model_ranks.mean()

        # Create average and average rank DataFrames with MultiIndex
        avg_df = pd.DataFrame([model_avg], index=pd.MultiIndex.from_tuples([(model, 'Avg.')],
                                                                           names=['model', 'ds']))
        avg_rank_df = pd.DataFrame([model_rank_avg], index=pd.MultiIndex.from_tuples([(model, 'Avg. Rank')],
                                                                                     names=['model', 'ds']))

        # Add original model data with MultiIndex
        model_with_index = pd.DataFrame(model_data.values,
                                        index=pd.MultiIndex.from_product([[model], model_data.index],
                                                                         names=['model', 'ds']),
                                        columns=model_data.columns)

        # Combine model data with averages (keeping them together)
        model_result = pd.concat([model_with_index, avg_df.round(4), avg_rank_df.round(2)])
        result_dfs.append(model_result)

    # Combine all model results
    final_result = pd.concat(result_dfs)

    return final_result


COLUMN_MAP = {
    'MagnitudeWarping': 'M-Warp',
    'TimeWarping': 'T-Warp',
    'SeasonalMBB': 'MBB',
    'Jittering': 'Jitter',
    'original': 'Original',
    'SeasonalNaive': 'SNaive',
    'derived': 'QGTS(D)',
    'QGTSE': 'Grasynda(E)',
    'QGTS': 'Grasynda',
}

APPROACH_COLORS = [
    '#2c3e50',  # Dark slate blue
    '#34558b',  # Royal blue
    '#4b7be5',  # Bright blue
    '#6db1bf',  # Light teal
    '#bf9b7a',  # Warm tan
    '#d17f5e',  # Warm coral
    '#c44536',  # Burnt orange red
    '#8b1e3f',  # Deep burgundy
    '#472d54',  # Deep purple
    '#855988',  # Muted mauve
    '#2d5447',  # Forest green
    '#507e6d'  # Sage green
]

wilc = pd.read_csv('assets/results/wilcoxon.csv')
wilc['method'] = wilc['method'].map(lambda x: COLUMN_MAP.get(x, x))
wilc = wilc.drop(columns=['mean_mase'])
wilc = wilc.rename(columns={'dataset': 'ds'})

wilc_binary = pd.pivot(wilc, columns='method', index=['model', 'ds']) < 0.05
wilc_binary.columns = [x[1] for x in wilc_binary.columns]

df = df.rename(columns=COLUMN_MAP)
df = df[['Original', 'Grasynda', 'Grasynda(E)', 'DBA',
         'Jitter', 'M-Warp', 'MBB', 'Scaling', 'T-Warp', 'TSMixup',
         'SNaive', 'ds', 'model']]

# overall details on table
perf_by_all = df.groupby(['model', 'ds']).mean(numeric_only=True)
og = perf_by_all['Original']
effectiveness = perf_by_all.apply(lambda x: (x < og).astype(int), axis=0).mean()
perf_by_all_ext = add_model_averages_and_ranks(df=perf_by_all).sort_index(level=['model'], ascending=[False])

avg_perf = perf_by_all.reset_index().groupby('model').mean(numeric_only=True)
avg_rank = perf_by_all.rank(axis=1).reset_index(level='model').groupby('model').mean(numeric_only=True).round(2)
avg_rank_ds = perf_by_all.rank(axis=1).reset_index(level='ds').groupby('ds').mean(numeric_only=True).round(2)
avg_perf_ds = perf_by_all.reset_index().groupby('ds').mean(numeric_only=True)

# perf_by_mod = df.groupby(['model']).mean(numeric_only=True)
# avg_score = perf_by_mod.mean().values

# avg_rank = perf_by_all.rank(axis=1).mean().round(2).values

# perf_by_mod.loc[('All', 'Average'), :] = avg_score
# perf_by_all.loc[('All', 'Avg. Rank'), :] = avg_rank
perf_by_all_ext.loc[('All', 'Effectiveness'), :] = effectiveness.round(2)

#
df = perf_by_all_ext.copy()

df = df.round(4)

annotated_res_list = []
for i, r in df.iterrows():
    print(i)
    top_2 = r.sort_values().unique()[:2]
    if len(top_2) < 2:
        raise ValueError('only one score')

    best1 = r[r == top_2[0]].values[0]
    best2 = r[r == top_2[1]].values[0]

    r[r == top_2[0]] = f'\\textbf{{{best1}}}'
    r[r == top_2[1]] = f'\\underline{{{best2}}}'


    star_models = []
    if i[1] not in ['Avg.', 'Avg. Rank', 'Effectiveness']:
        wilc_i = wilc_binary.loc[i]

        for m_ in r.index:
            if m_ == 'Original':
                continue

            if df.loc[i][m_] < df.loc[i]['Original']:
                # if wilc_i[m_]:
                #     r[m_] += '??'
                star_models.append(m_)

    # star_models = ['Scaling', 'MBB']
    # for model_key in star_models:
    #     if model_key in r.index:  # Ensure the key exists
            # r[model_key] = str(r[model_key]) + '*'

    if len(star_models) > 0:
        r[star_models] = [f'{x}*' for x in r[star_models]]

    r = r.astype(str)
    print(r)

    annotated_res_list.append(r)

annotated_res = pd.DataFrame(annotated_res_list).astype(str)

annotated_res.index = pd.MultiIndex.from_tuples(
    [(f'\\rotatebox{{90}}{{{x[0]}}}', x[1]) for x in annotated_res.index]
)
annotated_res.columns = [f'\\rotatebox{{60}}{{{x}}}' for x in annotated_res.columns]

text_tab = annotated_res.to_latex(caption='CAPTION', label='tab:scores_by_ds')

#

tex_tab = to_latex_tab(text_tab, 4, rotate_cols=True)
print(tex_tab)
