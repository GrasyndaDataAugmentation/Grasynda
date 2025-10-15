import re
import sys
import os
sys.path.append(r'C:\Users\lhenr\Desktop\graph_based_time_series_aug')
import pandas as pd

import plotnine as p9

from utils.analysis import to_latex_tab, read_results, THEME

RESULTS_DIR = 'assets/results/ensemble_sens'


file_path = os.path.join(RESULTS_DIR, "main_results.csv") 

df = read_results('mase', data_directory=RESULTS_DIR)
# Compute MEDIAN MASE (instead of mean or ranks)
avg_perf = df.drop(columns=['ds', 'Original']).groupby(['model']).median()  # Changed to median()

# Melt for plotting
scores_df = avg_perf.reset_index().melt(id_vars='model')
scores_df.columns = ['Model', 'Method', 'Median MASE']  # Updated column name

# Extract ensemble parameter (e.g., quantiles or sizes)
def extract_method_info(method_string):
    s = str(method_string)
    if s.startswith(('mean mase', 'p value')):
        return None
    match = re.search(r'\((\d+)\)', s)
    if match:
        return int(match.group(1))
    return None


scores_df['Method'] = scores_df['Method'].apply(extract_method_info)

scores_df["Median MASE"] = pd.to_numeric(scores_df["Median MASE"], errors='coerce')

print(scores_df['Method'])
# Define desired order
desired_order = [3, 5, 7, 10, 15, 20, 25, 50, 75, 100]

# Filter scores_df to keep only those methods you want (optional but recommended)
scores_df = scores_df[scores_df['Method'].isin(desired_order)]

# Convert Method to categorical with that order
scores_df['Method'] = pd.Categorical(scores_df['Method'], categories=desired_order, ordered=True)

# Now plot using scale_x_discrete for even spacing
plot = p9.ggplot(data=scores_df,
                 mapping=p9.aes(x='Method',
                                y='Median MASE',
                                group='Model',
                                color='Model')) + \
       p9.geom_line(size=1, alpha=0.8) + \
       p9.geom_point(size=3) + \
       THEME + \
       p9.labs(title='', y='Median MASE', x='Ensemble Parameter') + \
       p9.theme(figure_size=(10, 6),
                axis_text_x=p9.element_text(size=12),
                legend_position="top") + \
       p9.scale_x_discrete(drop=False)  # drop=False keeps all categories even if missing in data

plot.save('assets/results/outputs/mase_by_model_sens.pdf', height=5, width=10)
