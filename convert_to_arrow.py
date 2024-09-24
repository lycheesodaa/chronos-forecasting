from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import numpy as np
from gluonts.dataset.arrow import ArrowWriter


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
    )

    # Set an arbitrary start time
    if start_times is None:
        # Set an arbitrary start time
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)
        
    assert len(time_series) == len(start_times)

    dataset = [
        {"start": start, "target": ts} for ts, start in zip(time_series, start_times)
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


if __name__ == "__main__":
    # Generate 20 random time series of length 1024
    df = pd.read_csv("data/demand_data_all_cleaned_numerical.csv")
    ts_data = [df['actual'].values]
    start_times = [df['datetime'][0]]
    print(ts_data)
    print(start_times)

    # Convert to GluonTS arrow format
    convert_to_arrow("./data/processed/demand_data_all_cleaned_numerical.arrow",
                     time_series=ts_data, start_times=start_times)