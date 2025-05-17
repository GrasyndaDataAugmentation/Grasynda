import re

import pandas as pd
import plotnine as p9

from utils.analysis import to_latex_tab, read_results, THEME

RESULTS_DIR = 'assets/results/sensitivity_results'

df = read_results('mase', data_directory=RESULTS_DIR)

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

# overall details on table
# avg_perf = df.groupby(['model']).mean(numeric_only=True)
avg_perf = df.drop(columns='ds').groupby(['model']).apply(lambda x: x.rank(axis=1).mean())
# avg_perf = df.drop(columns='model').groupby(['ds']).apply(lambda x: x.rank(axis=1).mean())

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

plot = \
    p9.ggplot(data=scores_df,
              mapping=p9.aes(x='Model',
                             y='Average Rank',
                             fill='Method')) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9) + \
    THEME + \
    p9.theme(axis_title_y=p9.element_text(size=12),
             axis_title_x=p9.element_blank(),
             axis_text=p9.element_text(size=12)) + \
    p9.scale_fill_manual(values=APPROACH_COLORS)

plot.save('assets/results/outputs/mase_by_model_sens.pdf', height=5, width=12)
