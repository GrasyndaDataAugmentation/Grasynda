import os
import pandas as pd
import numpy as np
import pymdma
from pymdma.time_series.measures.synthesis_val import (
    ImprovedPrecision, 
    Density, 
    FrechetDistance,
    Authenticity
)

TRAIN_DIR = "assets/results/training_sets"
PYMDMA_DIR = "assets/results/pymdma_metrics"
os.makedirs(PYMDMA_DIR, exist_ok=True)

DATASETS_CONFIG = [
    {"name": "Tourism", "groups": ["Monthly", "Quarterly"]},
    {"name": "M3", "groups": ["Monthly", "Quarterly"]},
    {"name": "Gluons-m1", "groups": ["Monthly", "Quarterly"]},
]

def df_to_array(df):
    pivot = df.pivot(index="ds", columns="unique_id", values="y")
    pivot = pivot.fillna(method="ffill").fillna(method="bfill")
    return pivot.T.values

def extract_main_value(metric_result):
    if isinstance(metric_result.value, tuple):
        return metric_result.value[0]
    else:
        return metric_result.value

def evaluate_dataset(dataset_name, group):
    real_path = os.path.join(TRAIN_DIR, f"{dataset_name}_{group}_original.csv")
    
    if not os.path.exists(real_path):
        print(f"Real data not found: {real_path}")
        return None
    
    synthetic_files = [
        f for f in os.listdir(TRAIN_DIR)
        if f.startswith(f"{dataset_name}_{group}_") and "original" not in f
    ]
    
    print(f"Evaluating {dataset_name}-{group}: Found {len(synthetic_files)} synthetic datasets")
    
    real_df = pd.read_csv(real_path)
    real_data = df_to_array(real_df)
    print(f"Real data shape: {real_data.shape}")
    
    results = {}
    
    for fname in synthetic_files:
        synth_name = fname.replace(f"{dataset_name}_{group}_", "").replace(".csv", "")
        print(f"  Evaluating {synth_name}...")
        
        try:
            synth_df = pd.read_csv(os.path.join(TRAIN_DIR, fname))
            synth_data = df_to_array(synth_df)
            print(f"  Synthetic data shape: {synth_data.shape}")
            
            metrics = {}
            
            precision = ImprovedPrecision()
            density = Density()
            fd = FrechetDistance()
            authenticity = Authenticity()
            
            metrics['precision'] = round(extract_main_value(precision.compute(real_data, synth_data)), 3)
            metrics['density'] = round(extract_main_value(density.compute(real_data, synth_data)), 3)
            metrics['frechet'] = round(extract_main_value(fd.compute(real_data, synth_data)), 3)
            metrics['authenticity'] = round(extract_main_value(authenticity.compute(real_data, synth_data)), 3)
            
            results[synth_name] = metrics
            print(f"  Computed metrics: {metrics}")
            
        except Exception as e:
            print(f"  Error evaluating {synth_name}: {e}")
            results[synth_name] = {
                'precision': np.nan,
                'density': np.nan,
                'frechet': np.nan,
                'authenticity': np.nan
            }
    
    return results

all_results = {}

for dataset_config in DATASETS_CONFIG:
    dataset_name = dataset_config["name"]
    all_results[dataset_name] = {}
    
    for group in dataset_config["groups"]:
        results = evaluate_dataset(dataset_name, group)
        if results:
            all_results[dataset_name][group] = results
            
            output_path = os.path.join(PYMDMA_DIR, f"{dataset_name}_{group}_pymdma_results.csv")
            summary_df = pd.DataFrame.from_dict(results, orient='index')
            summary_df.to_csv(output_path)
            print(f"Saved {dataset_name}-{group} results to: {output_path}")

print("CALCULATING AVERAGES ACROSS ALL DATASETS")

all_methods = set()
for dataset_name, groups in all_results.items():
    for group, methods in groups.items():
        all_methods.update(methods.keys())

method_metrics_accumulator = {method: {
    'precision': [], 'density': [], 'frechet': [], 'authenticity': []} 
    for method in all_methods}

for dataset_name, groups in all_results.items():
    for group, methods in groups.items():
        for method_name, metrics in methods.items():
            if not np.isnan(metrics['precision']):
                method_metrics_accumulator[method_name]['precision'].append(metrics['precision'])
                method_metrics_accumulator[method_name]['density'].append(metrics['density'])
                method_metrics_accumulator[method_name]['frechet'].append(metrics['frechet'])
                method_metrics_accumulator[method_name]['authenticity'].append(metrics['authenticity'])

average_results = {}
for method_name, metrics_dict in method_metrics_accumulator.items():
    if metrics_dict['precision']:
        avg_precision = np.mean(metrics_dict['precision'])
        avg_density = np.mean(metrics_dict['density'])
        avg_frechet = np.mean(metrics_dict['frechet'])
        avg_authenticity = np.mean(metrics_dict['authenticity'])
        
        n_datasets = len(metrics_dict['precision'])
        
        average_results[method_name] = {
            'precision': round(avg_precision, 3),
            'density': round(avg_density, 3),
            'frechet': round(avg_frechet, 3),
            'authenticity': round(avg_authenticity, 3),
            'n_datasets': n_datasets
        }

if average_results:
    avg_df = pd.DataFrame.from_dict(average_results, orient='index')
    avg_output_path = os.path.join(PYMDMA_DIR, "OVERALL_AVERAGE_RESULTS.csv")
    avg_df.to_csv(avg_output_path)
    print(f"SAVED OVERALL AVERAGE RESULTS TO: {avg_output_path}")
    print("Average Results Summary:")
    print(avg_df)
else:
    print("No valid results to average")

print("All evaluations completed!")