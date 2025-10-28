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
    pivot = df.pivot(index="ds", columns="unique_id", values="y")
    pivot = pivot.ffill().bfill()
    return pivot.T.values

def extract_features(time_series_data, fs=12):
    cfg = tsfel.get_features_by_domain()
    features_list = []
    
    for i in range(time_series_data.shape[0]):
        single_ts = time_series_data[i, :]
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
    
    results = {}
    
    for fname in synthetic_files:
        synth_name = fname.replace(f"{dataset_name}_{group}_", "").replace(".csv", "")
        
        synth_df = pd.read_csv(os.path.join(TRAIN_DIR, fname))
        synth_data = df_to_array(synth_df)
        synth_features = extract_features(synth_data, fs=12)
        
        precision = ImprovedPrecision()
        recall = ImprovedRecall()
        
        results[synth_name] = {
            'precision': round(extract_main_value(precision.compute(real_features, synth_features)), 3),
            'recall': round(extract_main_value(recall.compute(real_features, synth_features)), 3)
        }
    
    return results

all_results = {}

for dataset_config in DATASETS_CONFIG:
    dataset_name = dataset_config["name"]
    
    for group in dataset_config["groups"]:
        results = evaluate_dataset(dataset_name, group)
        all_results[dataset_name] = {group: results}
        
        output_path = os.path.join(PYMDMA_DIR, f"{dataset_name}_{group}_pymdma_results.csv")
        pd.DataFrame.from_dict(results, orient='index').to_csv(output_path)

print("Evaluation completed!")