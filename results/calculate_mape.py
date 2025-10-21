import os
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_mape(true_values, pred_values):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    true_values, pred_values = np.array(true_values), np.array(pred_values)
    # Avoid division by zero
    mask = true_values != 0
    return np.mean(np.abs((true_values[mask] - pred_values[mask]) / true_values[mask])) * 100

def process_results_directory(results_dir):
    results_dir = Path(results_dir)
    print(results_dir)
    output_file = results_dir.parent / 'mape_results.csv'
    
    # Get all CSV files in the results directory
    csv_files = list(results_dir.glob('*.csv'))
    print(csv_files)
    
    results = []
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Extract relevant columns (assuming they're named 'true' and 'pred')
            if 'true' in df.columns and 'pred' in df.columns:
                mape = calculate_mape(df['true'], df['pred'])
                results.append({
                    'dataset': csv_file.stem,
                    'mape': mape,
                    'num_samples': len(df)
                })
                print(f"Processed {csv_file.name}: MAPE = {mape:.2f}%")
            else:
                print(f"Skipping {csv_file.name}: Missing required columns ('true' and/or 'pred')")
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        return results_df
    else:
        print("No valid results to save.")
        return None

if __name__ == "__main__":
    # Assuming the script is run from the project root
    results_dir = './results/'
    process_results_directory(results_dir)
