import argparse
import os
import sys
from typing import List, Optional

import pandas as pd
import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract contexts and compute MSE for Chronos2 predictions.", add_help=False
    )
    parser.add_argument("--config-file", type=str, default="./configs/demand_aus_datasets.yaml")
    parser.add_argument("--predictions-dir", type=str, default="./outputs/chronos2/demand_aus")
    parser.add_argument("--output-csv", type=str, default="./outputs/chronos2/demand_aus_contexts_mse.csv")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    parser.add_argument("--target", type=str, default="")
    return parser.parse_args()


def load_yaml_config(config_path: str) -> List[dict]:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("Expected YAML to contain a list of run configurations.")
    return data


def select_configs(configs: List[dict]) -> List[dict]:
    selected: List[dict] = []
    for cfg in configs:
        cfg_name = str(cfg.get("name", ""))
        if cfg_name.startswith("demand_aus_"):
            selected.append(cfg)
    return selected


def infer_timestamp_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred and preferred in df.columns:
        return preferred
    candidates = ["timestamp", "datetime", "date", "time", "ds"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a timestamp column. Columns are: {df.columns.tolist()}")


def ensure_id_column(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    if id_column in df.columns:
        return df
    df = df.copy()
    df[id_column] = "series_0"
    return df


def find_prediction_file(base_dir: str, config_name: str) -> Optional[str]:
    target_name = f"{config_name}_rolling_predictions.csv"
    for root, dirs, files in os.walk(base_dir):
        if target_name in files:
            return os.path.join(root, target_name)
    return None


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    configs = load_yaml_config(args.config_file)
    cfgs = select_configs(configs)

    records = []

    for cfg in tqdm(cfgs, desc="Processing configs"):
        config_name = str(cfg.get("name", ""))
        pred_file = find_prediction_file(args.predictions_dir, config_name)
        if not pred_file:
            continue

        covariates_config = config_name.split("_")[3]

        csv_path = cfg["filename"]
        if not os.path.isabs(csv_path):
            csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            continue

        prediction_length = int(cfg["prediction_length"])
        target_col = args.target if args.target else str(cfg.get("series_fields", "actual"))

        # Check if covariates parameter acts implicitly or if list is empty
        covariates = cfg.get("covariates_fields", [])
        if not covariates:
            covariates = []

        freq = str(cfg.get("freq", ""))

        df = pd.read_csv(csv_path)
        df = ensure_id_column(df, args.id_column)
        ts_col = infer_timestamp_column(df, args.timestamp_column)

        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.sort_values([args.id_column, ts_col])

        pred_df = pd.read_csv(pred_file)
        if pred_df.empty:
            continue

        pred_df["cutoff"] = pd.to_datetime(pred_df["cutoff"])

        if "datetime" in pred_df.columns:
            pred_datetime_col = "datetime"
        elif "timestamp" in pred_df.columns:
            pred_datetime_col = "timestamp"
        elif ts_col in pred_df.columns:
            pred_datetime_col = ts_col
        else:
            # Fallback
            pred_datetime_col = pred_df.columns[1]

        pred_df[pred_datetime_col] = pd.to_datetime(pred_df[pred_datetime_col])

        joined = pred_df.merge(
            df[[args.id_column, ts_col, target_col]],
            how="inner",
            left_on=[args.id_column, pred_datetime_col],
            right_on=[args.id_column, ts_col],
        )

        # In case the columns have suffixes, let's select correctly or just drop one.
        if target_col + "_y" in joined.columns:
            joined = joined.rename(columns={target_col + "_y": "actual"})
        elif target_col in joined.columns and "actual" != target_col:
            joined = joined.rename(columns={target_col: "actual"})

        if "actual" not in joined.columns and target_col in joined.columns:
            joined = joined.rename(columns={target_col: "actual"})

        # Target column could be the same as 'actual', no renaming needed.

        joined["squared_err"] = (joined["predictions"] - joined["actual"]) ** 2

        mse_df = (
            joined.groupby([args.id_column, "cutoff"], dropna=False).agg(mse=("squared_err", "mean")).reset_index()
        )

        df_grouped = dict(tuple(df.groupby(args.id_column)))

        import numpy as np

        for _, mse_row in mse_df.iterrows():
            series_id = mse_row[args.id_column]
            cutoff = mse_row["cutoff"]
            mse = mse_row["mse"]

            group = df_grouped.get(series_id)
            if group is None:
                continue

            ts_values = group[ts_col].values
            idx = np.searchsorted(ts_values, np.datetime64(cutoff), side="right")

            target_vals = group[target_col].values
            start_idx = max(0, idx - 512)
            ctx_vals = target_vals[start_idx:idx].tolist()

            records.append(
                {
                    "location": "aus",
                    "context": ctx_vals,
                    "covariates_config": covariates_config,
                    "covariates_used": covariates,
                    "prediction_length": prediction_length,
                    "frequency": freq,
                    "MSE": mse,
                }
            )

    out_df = pd.DataFrame(records)

    # Determine the best configuration of covariates for optimal MSE
    out_df["context"] = out_df["context"].apply(tuple)
    best_idx = out_df.groupby(["location", "context", "prediction_length", "frequency"])["MSE"].idxmin()
    out_df = out_df.loc[best_idx].drop(columns=["MSE"]).reset_index(drop=True)

    # Output file name was derived from args.output_csv natively
    out_file = args.output_csv.replace(".csv", ".parquet")
    out_df.to_parquet(out_file, index=False)
    print(f"Saved {len(out_df)} contexts to {out_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
