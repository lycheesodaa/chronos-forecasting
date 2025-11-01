import argparse
import os
import sys
from typing import List, Optional

import pandas as pd
import yaml
from tqdm import tqdm

try:
	from chronos import BaseChronosPipeline, Chronos2Pipeline
except Exception as e:
	print("Error: chronos-forecasting is not installed. Install with: pip install -U 'chronos-forecasting>=2.0' 'pandas[pyarrow]'")
	raise


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run Chronos-2 zero-shot forecasting for demand_aus using configs.", add_help=False)
	parser.add_argument("--config-file", type=str, default="./configs/demand_aus_datasets.yaml")
	parser.add_argument("--name", type=str, default=None)
	parser.add_argument("--output-dir", type=str, default="./outputs/chronos2/demand_aus")
	parser.add_argument("--device", type=str, default="cuda:1")
	parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
	parser.add_argument("--id-column", type=str, default="id")
	parser.add_argument("--timestamp-column", type=str, default="timestamp")
	parser.add_argument("--target", type=str, default=None)
	parser.add_argument("--predict-batches-jointly", action="store_true")
	parser.add_argument("--batch-size", type=int, default=64)
	# Rolling evaluation is always enabled; metrics are always computed.
	return parser.parse_args()


def load_yaml_config(config_path: str) -> List[dict]:
	with open(config_path, "r") as f:
		data = yaml.safe_load(f)
	if not isinstance(data, list):
		raise ValueError("Expected YAML to contain a list of run configurations.")
	return data


def select_configs(configs: List[dict], name: Optional[str]) -> List[dict]:
    if name:
        for cfg in configs:
            if str(cfg.get("name")) == name:
                return [cfg]
        raise ValueError(f"No config with name '{name}' found in the YAML file.")
    selected: List[dict] = []
    for cfg in configs:
        cfg_name = str(cfg.get("name", ""))
        if cfg_name.startswith("demand_aus_"):
            selected.append(cfg)
    if not selected:
        raise ValueError("No 'demand_aus_' entry found in the YAML file.")
    return selected


def infer_timestamp_column(df: pd.DataFrame, preferred: str) -> str:
	if preferred in df.columns:
		return preferred
	candidates = [
		"timestamp",
		"datetime",
		"date",
		"time",
		"ds",
	]
	for col in candidates:
		if col in df.columns:
			return col
	raise ValueError(
		"Could not find a timestamp column. Provide --timestamp-column or ensure one of these exists: "
		+ ", ".join(candidates)
	)


def ensure_id_column(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
	if id_column in df.columns:
		return df
	df = df.copy()
	df[id_column] = "series_0"
	return df


def prepare_context_df(df: pd.DataFrame, id_column: str, timestamp_column: str, target: str, covariates: List[str]) -> pd.DataFrame:
	cols = [c for c in [id_column, timestamp_column, target] + list(covariates) if c in df.columns]
	missing = [c for c in [id_column, timestamp_column, target] if c not in cols]
	if missing:
		raise ValueError(f"Missing required columns in CSV: {missing}")
	context_df = df[cols].copy()
	context_df[timestamp_column] = pd.to_datetime(context_df[timestamp_column])
	context_df = context_df.sort_values([id_column, timestamp_column])
	return context_df


def compute_metrics(pred_df: pd.DataFrame, df_full: pd.DataFrame, id_column: str, timestamp_column: str, target: str) -> pd.DataFrame:
	joined = pred_df.merge(
		df_full[[id_column, timestamp_column, target]],
		how="left",
		on=[id_column, timestamp_column],
		validate="m:1",
	)
	joined = joined.rename(columns={target: "actual"})
	joined = joined.dropna(subset=["actual", "predictions"]).copy()
	if joined.empty:
		return pd.DataFrame()
	joined["abs_err"] = (joined["predictions"] - joined["actual"]).abs()
	joined["squared_err"] = (joined["predictions"] - joined["actual"]) ** 2
	joined["ape"] = (joined["abs_err"] / joined["actual"].replace(0, pd.NA)).astype(float)

	metrics = joined.groupby(["cutoff"], dropna=False).agg(
		mae=("abs_err", "mean"),
		rmse=("squared_err", lambda x: (x.mean()) ** 0.5),
		mape=("ape", "mean"),
	).reset_index()
	# overall row
	overall = pd.DataFrame({
		"cutoff": ["OVERALL"],
		"mae": [joined["abs_err"].mean()],
		"rmse": [(joined["squared_err"].mean()) ** 0.5],
		"mape": [joined["ape"].mean()],
	})
	return pd.concat([metrics, overall], ignore_index=True)


def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch  # noqa: F401
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    # Pass through explicit device strings like "cuda", "cpu", or "cuda:1"
    return device_arg


def main() -> None:
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)

	configs = load_yaml_config(args.config_file)
	cfgs = select_configs(configs, args.name)

	quantiles = [float(x) for x in args.quantiles.split(",")]
	device = get_device(args.device)
	print(f"Using device: {device}")

	# Load Chronos-2 pipeline once
	pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)

	for cfg in cfgs:
		base_name = str(cfg.get("name", "demand_aus_run"))
		print(f"\n=== Running config: {base_name} ===")
		csv_path = cfg["filename"]
		prediction_length = int(cfg["prediction_length"])  # we prioritize this
		target_col = args.target if args.target else str(cfg.get("series_fields", "target"))
		covariates = list(cfg.get("covariates_fields", []))
		offset = int(cfg.get("offset", 0))
		num_rolls = int(cfg.get("num_rolls", 0))
		freq = str(cfg.get("freq", ""))

		if not os.path.isabs(csv_path):
			csv_path = os.path.abspath(csv_path)
		if not os.path.exists(csv_path):
			raise FileNotFoundError(f"CSV not found: {csv_path}")

		print(f"Loading data from: {csv_path}")
		df = pd.read_csv(csv_path)

		# Ensure id/timestamp presence
		df = ensure_id_column(df, args.id_column)
		ts_col = infer_timestamp_column(df, args.timestamp_column)

		context_df = prepare_context_df(df, args.id_column, ts_col, target_col, covariates)

		# Rolling window evaluation only
		if num_rolls <= 0 or not freq:
			raise ValueError("Rolling eval requires valid 'num_rolls' (>0) and 'freq' in YAML.")

		df_full = df.copy()
		df_full[ts_col] = pd.to_datetime(df_full[ts_col])
		df_full = df_full.sort_values([args.id_column, ts_col])

		all_preds = []

		for series_id, group in df_full.groupby(args.id_column):
			ts_list = group[ts_col].tolist()
			n = len(ts_list)
			# offset marks the START index (relative to the end) of evaluation
			eval_start_idx = n + offset
			if eval_start_idx < 0 or eval_start_idx >= n:
				raise ValueError(f"Computed eval_start_idx out of bounds for id={series_id}: {eval_start_idx} (n={n}, offset={offset})")
			end_idx = min(n, eval_start_idx + num_rolls)
			cutoff_indices = list(range(eval_start_idx, end_idx))
			cutoffs = [ts_list[i] for i in cutoff_indices]

			# Print first evaluation timestamp for this series for verification
			if cutoff_indices:
				first_cutoff = ts_list[cutoff_indices[0]]
				first_eval_ts = group.loc[group[ts_col] > first_cutoff, ts_col].min()
				if pd.notna(first_eval_ts):
					print(f"[{base_name}] id={series_id} first eval timestamp: {first_eval_ts}")

			for cutoff in tqdm(cutoffs, desc=f"{base_name} id={series_id}"):
				ctx_df = context_df[context_df[ts_col] <= cutoff]
				pred = pipeline.predict_df(
					ctx_df,
					future_df=None,
					prediction_length=prediction_length,
					quantile_levels=quantiles,
					id_column=args.id_column,
					timestamp_column=ts_col,
					target=target_col,
					predict_batches_jointly=args.predict_batches_jointly,
					batch_size=args.batch_size,
				)
				pred["cutoff"] = cutoff
				all_preds.append(pred)

		combined = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

		out_csv = os.path.join(args.output_dir, f"{base_name}_rolling_predictions.csv")
		combined.to_csv(out_csv, index=False)
		print(f"Saved rolling predictions to: {out_csv}")
		# try:
		# 	out_parquet = os.path.join(args.output_dir, f"{base_name}_rolling_predictions.parquet")
		# 	combined.to_parquet(out_parquet, index=False)
		# except Exception:
		# 	out_parquet = None
		# if out_parquet and os.path.exists(out_parquet):
		# 	print(f"Saved rolling predictions to: {out_parquet}")

		if not combined.empty:
			metrics_df = compute_metrics(
				combined,
				df_full[[args.id_column, ts_col, target_col]].rename(columns={ts_col: ts_col}),
				args.id_column,
				ts_col,
				target_col,
			)
			metrics_csv = os.path.join(args.output_dir, f"{base_name}_rolling_metrics.csv")
			metrics_df.to_csv(metrics_csv, index=False)
			print(f"Saved rolling metrics to: {metrics_csv}")


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"Error: {e}", file=sys.stderr)
		sys.exit(1)
