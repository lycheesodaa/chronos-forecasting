import datasets
import numpy as np
import pandas as pd
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.dataset.common import ListDataset
from typing import List
from .m5 import get_m5_dataset
from .epf import get_epf_dataset


def to_gluonts_univariate(
    hf_dataset: datasets.Dataset,
    series_fields: List[str],
    covariates_fields: List[str] = None,
):
    if isinstance(series_fields, str):
        series_fields = [series_fields]

    # dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
    dataset_length = len(hf_dataset) * len(series_fields)

    # Assumes that all time series in the dataset have the same frequency
    dataset_freq = pd.DatetimeIndex(hf_dataset[0]["timestamp"]).to_period()[0].freqstr

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            entry = {
                "start": pd.Period(
                    hf_entry["timestamp"][0],
                    freq=dataset_freq,
                ),
                "target": hf_entry[field],
            }

            if covariates_fields:
                covariates = np.array([hf_entry[field] for field in covariates_fields])
                is_nan_covariates = np.isnan(covariates)
                covariates[is_nan_covariates] = -1
                covariates = np.vstack([covariates, is_nan_covariates]).astype(
                    np.float32
                )
                entry.update({FieldName.FEAT_DYNAMIC_REAL: covariates})

            gts_dataset.append(entry)

    assert len(gts_dataset) == dataset_length

    return gts_dataset


def load_and_split_dataset(backtest_config: dict):
    hf_repo = backtest_config.get("hf_repo", None)
    filename = backtest_config.get("filename", None)
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]
    series_fields = backtest_config.get("series_fields")
    covariates_fields = backtest_config.get("covariates_fields")

    if dataset_name == "m5_with_covariates":
        ds = get_m5_dataset()

    elif dataset_name in [
        "epf_electricity_be_paper",
        "epf_electricity_de_paper",
        "epf_electricity_fr_paper",
        "epf_electricity_np_paper",
        "epf_electricity_pjm_paper",
    ]:
        ds = get_epf_dataset(dataset_name)

    elif 'synthetic_datasets' in dataset_name:
        ds = datasets.load_from_disk(filename)
    elif 'demand' in dataset_name:
        df = pd.read_csv(filename)
        df = df.rename(columns={'datetime': 'timestamp'})
        df['id'] = dataset_name

        # Convert to list format, since the ChronosX code requires it this way
        ts_data = []
        for id_val, group in df.groupby('id'):
            ts_entry = {
                'timestamp': group['timestamp'].values,
                series_fields: group[series_fields].values.astype(np.float32),
                'id': np.array([id_val] * len(group)),
            }
            # Add other features as needed
            for col in group.columns:
                if col not in ['timestamp', series_fields, 'id'] and col in covariates_fields:
                    ts_entry[col] = group[col].values.astype(np.float32)
            ts_data.append(ts_entry)

        ds = datasets.Dataset.from_list(ts_data)
    else:
        ds = datasets.load_dataset(
            hf_repo, dataset_name, split="train", trust_remote_code=True
        )

    ds.set_format("numpy")

    gts_dataset = to_gluonts_univariate(
        ds, covariates_fields=covariates_fields, series_fields=series_fields
    )

    # Split dataset for evaluation
    train_dataset, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls, distance=1)

    return train_dataset, test_data


def split_train_val(dataset, val_ratio=0.25):
    """
    Split dataset into train and validation sets.
    Validation contains only the final val_ratio portion.
    """
    train_data = []
    val_data = []

    for entry in dataset:
        freq = entry["start"].freq
        target_length = len(entry["target"])
        split_point = int(target_length * (1 - val_ratio))

        # Training set: first 75%
        train_entry = {
            "start": entry["start"],
            "target": entry["target"][:split_point],
        }

        if "feat_dynamic_real" in entry:
            train_entry["feat_dynamic_real"] = entry["feat_dynamic_real"][
                :, :split_point
            ]

        for key in entry:
            if key not in ["target", "feat_dynamic_real", "start"]:
                train_entry[key] = entry[key]

        train_data.append(train_entry)

        # Validation set: only final 25%
        val_entry = {
            "start": entry["start"] + split_point,  # Adjust start time
            "target": entry["target"][split_point:],
        }

        if "feat_dynamic_real" in entry:
            val_entry["feat_dynamic_real"] = entry["feat_dynamic_real"][:, split_point:]

        for key in entry:
            if key not in ["target", "feat_dynamic_real", "start"]:
                val_entry[key] = entry[key]

        val_data.append(val_entry)

    return ListDataset(train_data, freq=freq), ListDataset(val_data, freq=freq)
