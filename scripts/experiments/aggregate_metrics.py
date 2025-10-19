import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
import re


def aggregate_metrics(metrics_dir: str, output_file: str = "aggregated_metrics.csv"):
    """
    Aggregate metrics from multiple experiment runs into a single CSV file.

    Args:
        metrics_dir: Path to the directory containing metrics CSV files
        output_file: Name of the output CSV file
    """
    metrics_dir = Path(metrics_dir)

    # Dictionary to store all metrics by dataset and learning rate
    metrics_data = []

    # Find all CSV files in the metrics directory
    csv_files = list(metrics_dir.glob("*max_steps=5000.csv"))

    # Filter out any existing aggregated files to avoid processing them
    csv_files = [
        f
        for f in csv_files
        if not f.name.startswith("aggregated_") and "mean" not in f.name
    ]

    if not csv_files:
        print(f"No CSV files found in {metrics_dir}")
        return None

    print(f"Found {len(csv_files)} CSV files to process")

    # Load all metrics files
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Extract dataset name and configuration from filename
            # Format: {dataset_name}_max_steps={max_steps}.csv
            filename = csv_file.stem
            dataset_name = filename.split("_max_steps=")[0]

            # Convert all numeric columns to float, ignore errors for non-numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Add dataset name to the dataframe
            df["dataset"] = dataset_name

            metrics_data.append(
                {
                    "dataset": dataset_name,
                    **df.mean().to_dict(),
                }
            )

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    if not metrics_data:
        print("No valid data found to aggregate")
        return None

    # Combine all dataframes
    combined_df = pd.DataFrame(metrics_data)
    combined_df.set_index("dataset", inplace=True)

    # Also save the detailed version
    detailed_path = metrics_dir / f"detailed_{output_file}"
    combined_df.to_csv(detailed_path)
    print(f"Detailed metrics saved to: {detailed_path}")

    return combined_df


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description='Aggregate metrics from multiple experiment runs')
#     parser.add_argument('--metrics-dir', type=str, default='../../output/metrics',
#                        help='Path to the directory containing metrics JSON files')
#     parser.add_argument('--output', type=str, default='aggregated_metrics.csv',
#                        help='Output CSV filename')

#     args = parser.parse_args()

#     # Run the aggregation
#     aggregate_metrics(args.metrics_dir, args.output)
