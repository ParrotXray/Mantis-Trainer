import os
from pathlib import Path
from typing import Final, List, Optional

import numpy as np
import pandas as pd
import ujson

from model import (
    SEQUENCE_META_COLUMNS,
    UNIFIED_FEATURE_NAMES,
    DatasetConfig,
    PreprocessConfig,
)
from utils import Logger


class DataPreprocess:
    def __init__(
        self,
        dataset_configs: List[DatasetConfig],
        paths: List[str],
        config: Optional[PreprocessConfig] = None,
    ) -> None:
        if len(dataset_configs) != len(paths):
            raise ValueError(
                f"Number of dataset configs ({len(dataset_configs)}) "
                f"must match number of paths ({len(paths)})"
            )

        self.dataset_configs: List[DatasetConfig] = dataset_configs
        self.paths: List[str] = paths
        self.config: PreprocessConfig = config or PreprocessConfig()

        self.datasets: List[pd.DataFrame] = []
        self.combined_data: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None

        self.log: Logger = Logger(__name__)

    # Explicit format list covers all known CIC/UNSW dataset variants.
    # Tried in order; first one where >50% of values parse is used.
    _TS_FORMATS: Final[List[str]] = [
        "%Y/%m/%d %H:%M:%S",  # 2010/6/12 03:01:06
        "%Y/%m/%d %I:%M:%S %p",  # 2010/6/12 03:01:06 AM
        "%Y/%m/%d %H:%M",  # 2010/6/12 03:54
        "%Y/%m/%d %I:%M %p",  # 2010/6/12 03:54 AM
        "%m/%d/%Y %H:%M:%S",  # 7/7/2017 8:08:46
        "%m/%d/%Y %I:%M:%S %p",  # 7/7/2017 8:08:46 AM
        "%m/%d/%Y %H:%M",  # 4/7/2017 12:43
        "%d/%m/%Y %H:%M:%S",  # 14/02/2018 10:00:00
        "%d/%m/%Y %H:%M",  # 14/02/2018 10:00
        "%Y-%m-%d %H:%M:%S",  # 2018-02-14 10:00:00
        "%Y-%m-%d %H:%M",  # 2018-02-14 10:00
    ]

    @staticmethod
    def _parse_timestamp_ms(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() > 0.5:
            return numeric.fillna(-1).astype("int64")

        for fmt in DataPreprocess._TS_FORMATS:
            ts = pd.to_datetime(series, format=fmt, errors="coerce")
            if ts.notna().mean() > 0.5:
                return (ts.astype("int64") // 1_000_000).where(ts.notna(), other=-1)

        ts = pd.to_datetime(series, errors="coerce")
        if ts.notna().mean() > 0.5:
            return (ts.astype("int64") // 1_000_000).where(ts.notna(), other=-1)

        def _time_to_ms(val: str) -> int:
            try:
                parts = str(val).strip().split(":")
                if len(parts) == 3:
                    return int(
                        (
                            float(parts[0]) * 3600
                            + float(parts[1]) * 60
                            + float(parts[2])
                        )
                        * 1000
                    )
                elif len(parts) == 2:
                    return int((float(parts[0]) * 60 + float(parts[1])) * 1000)
            except Exception:
                pass
            return -1

        return series.map(_time_to_ms).astype("int64")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.datasets.clear()
        self.combined_data = None
        self.feature_matrix = None
        self.labels = None
        return False

    def _load_csv_files(
        self,
        path: str,
        csv_glob: str = "*.csv",
        header_file: str = "",
        header_name_column: str = "",
    ) -> List[pd.DataFrame]:
        p = Path(path)
        frames = []

        if p.is_file() and p.suffix.lower() == ".csv":
            csv_files = [p]
        elif p.is_dir():
            csv_files = sorted(p.rglob(csv_glob))
        else:
            self.log.warning(f"Invalid path: {path}")
            return frames

        col_names = None
        if header_file and p.is_dir():
            header_path = p / header_file
            if header_path.exists():
                hdr_df = pd.read_csv(
                    str(header_path), encoding="utf-8", encoding_errors="replace"
                )
                col_names = hdr_df[header_name_column].str.strip().tolist()
                self.log.info(
                    f"Using column names from {header_file} ({len(col_names)} columns)"
                )

        for csv_path in csv_files:
            try:
                self.log.info(f"Loading {csv_path.name}")
                if col_names is not None:
                    df = pd.read_csv(
                        str(csv_path),
                        header=None,
                        names=col_names,
                        encoding="utf-8",
                        encoding_errors="replace",
                        low_memory=False,
                    )
                else:
                    df = pd.read_csv(
                        str(csv_path),
                        encoding="utf-8",
                        encoding_errors="replace",
                        low_memory=False,
                    )
                df.columns = df.columns.str.strip()
                frames.append(df)
                self.log.info(f"{csv_path.name} Shape: {df.shape}")
            except Exception as e:
                self.log.error(f"Error loading {csv_path}: {e}")

        return frames

    def load_datasets(self) -> None:
        all_frames: List[pd.DataFrame] = []

        for ds_config, path in zip(self.dataset_configs, self.paths):
            self.log.info(
                f"Loading dataset '{ds_config.kaggle_dataset_id}' from {path}..."
            )

            raw_frames = self._load_csv_files(
                path,
                ds_config.csv_glob,
                ds_config.header_file,
                ds_config.header_name_column,
            )
            if not raw_frames:
                self.log.warning(
                    f"No CSV files found for '{ds_config.kaggle_dataset_id}' at {path}"
                )
                continue

            combined_raw = pd.concat(raw_frames, ignore_index=True)
            self.log.info(
                f"{ds_config.kaggle_dataset_id}: {len(combined_raw):,} rows, "
                f"{combined_raw.shape[1]} columns"
            )

            # Extract labels BEFORE column mapping (uses original column names)
            labels = ds_config.extract_labels(combined_raw)

            mapped = ds_config.map_columns(combined_raw)

            # Compute derived features (e.g. UNSW needs flow_bytes_per_sec)
            mapped = ds_config.compute_derived(mapped)

            mapped["_label"] = labels.values
            mapped["_source"] = ds_config.kaggle_dataset_id

            all_frames.append(mapped)

        if not all_frames:
            raise ValueError("No datasets were successfully loaded!")

        self.combined_data = pd.concat(all_frames, ignore_index=True)
        self.log.info(
            f"Combined: {len(self.combined_data):,} rows from "
            f"{len(all_frames)} dataset(s)"
        )

    def statistics_dataset(self) -> None:
        if self.combined_data is None:
            raise ValueError("No combined data. Call load_datasets() first!")

        self.log.info("Dataset statistics...")
        self.labels = self.combined_data["_label"].copy()

        lines = ["\nLabel distribution:", "=" * 60]
        counts = self.labels.value_counts()
        for label, count in counts.items():
            lines.append(f"  {label:<35} {count:>10,}")
        lines.append(f"  {'TOTAL':<35} {len(self.labels):>10,}")
        lines.append("=" * 60)
        self.log.info("\n".join(lines))

        if "_source" in self.combined_data.columns:
            lines = ["\nPer-dataset breakdown:"]
            for source in self.combined_data["_source"].unique():
                mask = self.combined_data["_source"] == source
                source_labels = self.labels[mask]
                n_benign = source_labels.isin(
                    self.dataset_configs[0].benign_labels
                ).sum()
                n_attack = len(source_labels) - n_benign
                lines.append(
                    f"  {source}: {len(source_labels):,} total "
                    f"(Normal: {n_benign:,}, Attack: {n_attack:,})"
                )
            self.log.info("\n".join(lines))

    def feature_preparation(self) -> None:
        if self.combined_data is None:
            raise ValueError("No combined data. Call load_datasets() first!")

        self.log.info("Feature preparation (unified schema)...")

        available = [
            f for f in UNIFIED_FEATURE_NAMES if f in self.combined_data.columns
        ]
        missing = set(UNIFIED_FEATURE_NAMES) - set(available)

        self.log.info(
            f"Available features: {len(available)}/{len(UNIFIED_FEATURE_NAMES)}"
        )
        if missing:
            self.log.info(f"Missing features (will be NaN): {sorted(missing)}")

        self.feature_matrix = pd.DataFrame(index=self.combined_data.index)
        for feat in UNIFIED_FEATURE_NAMES:
            if feat in self.combined_data.columns:
                self.feature_matrix[feat] = pd.to_numeric(
                    self.combined_data[feat], errors="coerce"
                )
            else:
                self.feature_matrix[feat] = np.nan

        # Clean inf values (but preserve NaN for missing features)
        self.feature_matrix = self.feature_matrix.replace([np.inf, -np.inf], np.nan)

        for col in self.feature_matrix.columns:
            mask = self.feature_matrix[col].notna()
            self.feature_matrix.loc[mask, col] = self.feature_matrix.loc[
                mask, col
            ].clip(self.config.clip_min, self.config.clip_max)

        # Timestamp converted to int64 ms so sort_values("timestamp") gives correct temporal order.
        for meta_col in SEQUENCE_META_COLUMNS:
            if meta_col not in self.combined_data.columns:
                continue
            if meta_col == "timestamp":
                self.feature_matrix[meta_col] = self._parse_timestamp_ms(
                    self.combined_data[meta_col]
                )
            else:
                self.feature_matrix[meta_col] = self.combined_data[meta_col].values

        n_feature_cols = len(UNIFIED_FEATURE_NAMES)
        n_total = self.feature_matrix[UNIFIED_FEATURE_NAMES].size
        n_nan = self.feature_matrix[UNIFIED_FEATURE_NAMES].isna().sum().sum()
        pct_nan = n_nan / n_total * 100 if n_total > 0 else 0
        meta_present = [
            c for c in SEQUENCE_META_COLUMNS if c in self.feature_matrix.columns
        ]
        self.log.info(
            f"Feature matrix: {self.feature_matrix.shape} "
            f"({n_feature_cols} flow features + {len(meta_present)} metadata cols), "
            f"NaN: {n_nan:,} ({pct_nan:.1f}%)"
        )
        if meta_present:
            self.log.info(f"Sequence metadata preserved: {meta_present}")

    def output_result(self) -> None:
        if self.feature_matrix is None or self.labels is None:
            raise ValueError("No feature matrix. Call feature_preparation() first!")

        self.log.info("Saving processed data...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        output = self.feature_matrix.copy()
        output["Label"] = self.labels.values

        invalid_labels = ["Unknown", "0", "", "nan"]
        benign_labels = set()
        for cfg in self.dataset_configs:
            benign_labels.update(cfg.benign_labels)

        benign_mask = output["Label"].isin(benign_labels)
        attack_mask = ~benign_mask & (~output["Label"].isin(invalid_labels))
        output_benign = output[benign_mask]
        output_attack = output[attack_mask]

        dropped = len(output) - len(output_benign) - len(output_attack)
        if dropped > 0:
            self.log.warning(f"Dropped {dropped:,} rows with invalid labels")

        benign_csv_path = Path("outputs") / "preprocessing_benign.csv"
        attack_csv_path = Path("outputs") / "preprocessing_attack.csv"
        benign_parquet_path = Path("outputs") / "preprocessing_benign.parquet"
        attack_parquet_path = Path("outputs") / "preprocessing_attack.parquet"

        output_benign.to_csv(benign_csv_path, index=False)
        output_attack.to_csv(attack_csv_path, index=False)
        output_benign.to_parquet(benign_parquet_path, index=False)
        output_attack.to_parquet(attack_parquet_path, index=False)

        self.log.info(
            f"Normal samples:  {len(output_benign):>10,} -> "
            f"{benign_csv_path}, {benign_parquet_path}"
        )
        self.log.info(
            f"Attack samples:  {len(output_attack):>10,} -> "
            f"{attack_csv_path}, {attack_parquet_path}"
        )

        stats = {
            "total_samples": len(self.combined_data),
            "total_features": len(UNIFIED_FEATURE_NAMES),
            "benign_samples": len(output_benign),
            "attack_samples": len(output_attack),
            "datasets": [cfg.kaggle_dataset_id for cfg in self.dataset_configs],
            "label_distribution": self.labels.value_counts().to_dict(),
        }

        stats_path = Path("metadata") / "preprocessing_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            ujson.dump(stats, f, indent=2, ensure_ascii=False)
        self.log.info(f"Statistics saved: {stats_path}")
