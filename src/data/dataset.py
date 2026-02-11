"""
Scalable Data Loader for Large Genomic Datasets

Implements PyTorch IterableDataset to stream data from Parquet files.
Enables training on 100GB+ datasets without loading everything into RAM.
"""

from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


class GenomicBigDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        feature_cols: List[str],
        target_cols: Dict[str, str],  # {"lifespan": "age_death", "cvd": "has_cvd"}
        batch_size: int = 1024,
        shuffle_buffer_size: int = 10000,
    ):
        """
        Args:
            data_dir: Directory containing .parquet files
            feature_cols: List of column names to use as input features (genotypes)
            target_cols: Dictionary mapping model targets to dataframe columns
            batch_size: Number of samples to yield at once (internal optimization)
            shuffle_buffer_size: Size of buffer for local shuffling
        """
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.parquet")))

        if not self.files:
            print(f"Warning: No .parquet files found in {data_dir}")

        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size

    def _parse_file(self, filepath: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """Read a parquet file in batches"""
        try:
            parquet_file = pq.ParquetFile(filepath)

            # Iterate through row groups
            for i in range(parquet_file.num_row_groups):
                df = parquet_file.read_row_group(i).to_pandas()

                # Check if columns exist
                available_feats = [c for c in self.feature_cols if c in df.columns]

                # Fill missing features with 0 (Ref)
                # In production, this should be handled more carefully (imputation)
                X = df[available_feats].fillna(0).values.astype(np.float32)

                # Pad if features are missing from this specific file
                if len(available_feats) < len(self.feature_cols):
                    # Create full matrix
                    full_X = np.zeros((len(df), len(self.feature_cols)), dtype=np.float32)
                    # Map available columns to their positions
                    for col_idx, col_name in enumerate(self.feature_cols):
                        if col_name in df.columns:
                            full_X[:, col_idx] = df[col_name].fillna(0).values
                    X = full_X

                # Extract targets
                targets = {}
                for target_key, col_name in self.target_cols.items():
                    if col_name in df.columns:
                        targets[target_key] = df[col_name].fillna(0).values.astype(np.float32)
                    else:
                        targets[target_key] = np.zeros(len(df), dtype=np.float32)

                # Yield row by row (buffered shuffle happens in __iter__)
                for j in range(len(df)):
                    yield {
                        "genomic": torch.tensor(X[j]),
                        "targets": {k: torch.tensor(v[j]) for k, v in targets.items()},
                    }

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Distribute files among workers
        if worker_info is None:  # Single-process
            my_files = self.files
        else:
            # Per-worker split
            per_worker = int(np.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.files))
            my_files = self.files[start:end]

        # Shuffle files
        np.random.shuffle(my_files)

        buffer = []

        for filepath in my_files:
            for sample in self._parse_file(filepath):
                buffer.append(sample)

                if len(buffer) >= self.shuffle_buffer_size:
                    # Yield a random item from buffer
                    idx = np.random.randint(0, len(buffer))
                    yield buffer.pop(idx)

        # Yield remaining
        np.random.shuffle(buffer)
        for sample in buffer:
            yield sample
