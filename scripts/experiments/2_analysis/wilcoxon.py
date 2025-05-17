import pandas as pd
from scipy.stats import wilcoxon
import sys

# sys.path.append(r'C:\Users\lhenr\Desktop\graph_based_time_series_aug')

from utils.analysis import to_latex_tab, read_results, THEME

df = read_results('mase')
df = df.drop(columns=['derived_ensemble', 'derived'])

results = {}

# Group by dataset, model
for (dataset, model), group in df.groupby(['ds', 'model']):
    original = group['original']
    
    for method in group.columns.difference(['ds', 'model', 'original']):
        # Wilcoxon test
        p_value = wilcoxon(group[method], original).pvalue

        mean_mase = group[method].mean()

        results[(method, model, dataset)] = {
            'p_value': p_value,
            'mean_mase': mean_mase
        }

results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=['method', 'model', 'dataset'])
results_df.reset_index(inplace=True)

# Save to CSV
csv_path = "assets/results/wilcoxon.csv"
results_df.to_csv(csv_path, index=False)

print(f"Results saved to {csv_path}")
print(f"\nPreview:\n{results_df.head()}")

total_entries = len(results)
print(total_entries)

df_m1q = df.query('ds=="M3-Q"')
df_m1q_nhits = df_m1q.query('model=="NHITS"')

x = df_m1q['QGTSE']
y = df_m1q['original']

st = wilcoxon(x=x,y=y)
st.pvalue

