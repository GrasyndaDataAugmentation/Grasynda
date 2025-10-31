import os
import pandas as pd
import numpy as np
import tsfel
import pymdma
from pymdma.time_series.measures.synthesis_val import ImprovedPrecision, ImprovedRecall

TRAIN_DIR = "assets/results/training_sets"
PYMDMA_DIR = "assets/results/pymdma_metrics"
os.makedirs(PYMDMA_DIR, exist_ok=True)

DATASETS_CONFIG = [
    {"name": "Tourism", "groups": ["Monthly"]},
]

def df_to_array(df):
    time_series_list = []
    
    print(f"Before - Mean: {df['y'].mean():.2f}, Std: {df['y'].std():.2f}")
    
    for unique_id, group in df.groupby('unique_id'):
        group = group.sort_values('ds')
        ts_values = group['y'].values
        time_series_list.append(ts_values)
    
    # Calculate after stats
    all_values = np.concatenate(time_series_list)
    print(f"After  - Mean: {all_values.mean():.2f}, Std: {all_values.std():.2f}")
    
    return time_series_list

def extract_features(time_series_data, fs=12):
    cfg = tsfel.get_features_by_domain('temporal')
    features_list = []
   # print(time_series_data)
 
    for i, single_ts in enumerate(time_series_data):
        
        features = tsfel.time_series_features_extractor(cfg, single_ts, fs=fs, verbose=0)
        features_list.append(features.values.flatten())
    
    return np.array(features_list)  

def extract_main_value(metric_result):
    if isinstance(metric_result.value, tuple):
        return metric_result.value[0]
    else:
        return metric_result.value

def evaluate_dataset(dataset_name, group):
    real_path = os.path.join(TRAIN_DIR, f"{dataset_name}_{group}_original.csv")
    synthetic_files = [f for f in os.listdir(TRAIN_DIR) 
                      if f.startswith(f"{dataset_name}_{group}_") and "original" not in f]
    
    real_df = pd.read_csv(real_path)
    real_data = df_to_array(real_df)
    real_features = extract_features(real_data, fs=12)
   # print(f"Real features shape: {real_features.shape}")
    
    results = {}
    
    for fname in synthetic_files:
        synth_name = fname.replace(f"{dataset_name}_{group}_", "").replace(".csv", "")
        print(synth_name)
        synth_df = pd.read_csv(os.path.join(TRAIN_DIR, fname))
        synth_data = df_to_array(synth_df)
      #  print(f"\nSynthetic data shape ({synth_name}): {synth_data.shape}")
        synth_features = extract_features(synth_data, fs=12)
       # print(f"Synthetic features shape ({synth_name}): {synth_features.shape}")
        
        precision = ImprovedPrecision()
        recall = ImprovedRecall()
        
        results[synth_name] = {
            'precision': round(extract_main_value(precision.compute(real_features, synth_features)), 3),
            'recall': round(extract_main_value(recall.compute(real_features, synth_features)), 3)
        }
        
        print(f"✅ FINISHED {synth_name}: Precision={results[synth_name]['precision']}, Recall={results[synth_name]['recall']}")
    
    return results

all_results = {}

for dataset_config in DATASETS_CONFIG:
    dataset_name = dataset_config["name"]
    
    for group in dataset_config["groups"]:
        results = evaluate_dataset(dataset_name, group)
        all_results[dataset_name] = {group: results}
        
        output_path = os.path.join(PYMDMA_DIR, f"{dataset_name}_{group}_FEATTEST.csv")
        pd.DataFrame.from_dict(results, orient='index').to_csv(output_path)

print("Evaluation completed!")
