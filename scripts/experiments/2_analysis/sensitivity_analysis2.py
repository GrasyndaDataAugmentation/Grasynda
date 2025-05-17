import re

import pandas as pd
import plotnine as p9

from utils.analysis import to_latex_tab, read_results, THEME

RESULTS_DIR = 'assets/results/sensitivity_results'

df = read_results('mase', data_directory=RESULTS_DIR)

# overall details on table
# avg_perf = df.groupby(['model']).mean(numeric_only=True)
avg_perf = df.drop(columns='ds').groupby(['model']).apply(lambda x: x.rank(axis=1).mean())
# avg_perf = df.drop(columns='model').groupby(['ds']).apply(lambda x: x.rank(axis=1).mean())
avg_perf = avg_perf.drop(columns='Original')

# ord = avg_rank.mean().sort_values().index.tolist()
ord = avg_perf.mean().index.tolist()
# scores_df = avg_rank.reset_index().melt('model')
scores_df = avg_perf.reset_index().melt('model')
# scores_df = avg_perf.reset_index().melt('ds')
scores_df.columns = ['Model', 'Method', 'Average Rank']
scores_df['Method'] = pd.Categorical(scores_df['Method'], categories=ord)


def extract_method_info(method_string):
    # Look for digits inside parentheses
    match = re.search(r'\((\d+)\)', method_string)
    if match:
        return int(match.group(1))
    else:
        # Return the original string if no parentheses are found
        return method_string


# Apply the function to create a new column
scores_df['Method'] = scores_df['Method'].apply(extract_method_info)

plot = p9.ggplot(data=scores_df,
                 mapping=p9.aes(x='Method',
                                y='Average Rank',
                                group='Model',
                                color='Model')) + \
       p9.geom_line(size=1, alpha=0.8) + \
       p9.geom_point(size=3) + \
       THEME + \
       p9.labs(title='', y='Average Rank', x='Number of quantiles') + \
       p9.theme(figure_size=(10, 6),
                axis_text_x=p9.element_text(size=12),
                legend_position="top")
plot.save('assets/results/outputs/mase_by_model_sens.pdf', height=5, width=10)
