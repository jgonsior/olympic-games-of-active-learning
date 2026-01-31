"""
OGAL I/O Utilities

Common functions for loading, writing, and validating OGAL experiment results.
Use these helpers to ensure consistent data handling across scripts.

Example usage:
    from misc.io_utils import load_results, compute_run_id, validate_schema

    # Load completed experiments
    df = load_results(config, "05_done_workload.csv")

    # Compute run identity for deduplication
    run_id = compute_run_id(row)

    # Validate a results directory
    is_valid, errors = validate_schema(results_path)

Source: misc/io_utils.py
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import yaml


# Identity columns that define a unique experiment run
IDENTITY_COLUMNS = [
    "EXP_DATASET",
    "EXP_STRATEGY",
    "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    "EXP_START_POINT",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "EXP_RANDOM_SEED",
]

# Required columns in workload files
WORKLOAD_REQUIRED_COLUMNS = [
    "EXP_UNIQUE_ID",
    *IDENTITY_COLUMNS,
    "EXP_NUM_QUERIES",
]


def load_results(
    output_path: Union[str, Path],
    filename: str,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a results CSV file from the output directory.

    Args:
        output_path: Base output directory (e.g., config.OUTPUT_PATH)
        filename: Name of file to load (e.g., "05_done_workload.csv")
        usecols: Optional list of columns to load (for memory efficiency)

    Returns:
        DataFrame with loaded results

    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.ParserError: If file is corrupted

    Example:
        >>> from misc.io_utils import load_results
        >>> df = load_results("/path/to/output/exp_title", "05_done_workload.csv")
        >>> print(len(df))
        1234
    """
    filepath = Path(output_path) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    # Handle compressed files
    if filepath.suffix == ".xz" or str(filepath).endswith(".csv.xz"):
        return pd.read_csv(filepath, usecols=usecols)
    elif filepath.suffix == ".parquet":
        return pd.read_parquet(filepath, columns=usecols)
    else:
        return pd.read_csv(filepath, usecols=usecols)


def write_results(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    filename: str,
    compress: bool = True,
) -> Path:
    """
    Write results DataFrame to the output directory.

    Args:
        df: DataFrame to write
        output_path: Base output directory
        filename: Name of file to write
        compress: If True, compress CSV files with xz

    Returns:
        Path to written file

    Example:
        >>> from misc.io_utils import write_results
        >>> write_results(df, "/path/to/output", "my_results.csv")
        PosixPath('/path/to/output/my_results.csv.xz')
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename.endswith(".parquet"):
        filepath = output_dir / filename
        df.to_parquet(filepath, index=False)
    elif filename.endswith(".csv"):
        if compress:
            filepath = output_dir / f"{filename}.xz"
            df.to_csv(filepath, index=False, compression="xz")
        else:
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
    else:
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)

    return filepath


def compute_run_id(
    row: Union[pd.Series, Dict[str, Any]],
    identity_columns: Optional[List[str]] = None,
) -> str:
    """
    Compute a unique run identity hash from hyperparameter values.

    This is used for deduplication and merge operations. The hash is
    deterministic for the same hyperparameter combination.

    Args:
        row: Series or dict containing hyperparameter values
        identity_columns: Columns to use for identity (default: IDENTITY_COLUMNS)

    Returns:
        SHA256 hash string (first 16 chars) representing the run identity

    Example:
        >>> from misc.io_utils import compute_run_id
        >>> row = {"EXP_DATASET": 1, "EXP_STRATEGY": 2, ...}
        >>> run_id = compute_run_id(row)
        >>> print(run_id)
        'a1b2c3d4e5f67890'
    """
    if identity_columns is None:
        identity_columns = IDENTITY_COLUMNS

    # Build identity string from sorted column values
    identity_parts = []
    for col in sorted(identity_columns):
        if col in row:
            val = row[col]
            # Handle pandas Series indexing
            if hasattr(val, "iloc"):
                val = val.iloc[0] if len(val) > 0 else val
            identity_parts.append(f"{col}={val}")

    identity_str = "|".join(identity_parts)
    return hashlib.sha256(identity_str.encode()).hexdigest()[:16]


def compute_run_id_tuple(
    row: Union[pd.Series, Dict[str, Any]],
    identity_columns: Optional[List[str]] = None,
) -> Tuple:
    """
    Compute a tuple of identity values for DataFrame operations.

    This is useful for groupby or merge operations where a hashable
    identity is needed.

    Args:
        row: Series or dict containing hyperparameter values
        identity_columns: Columns to use for identity (default: IDENTITY_COLUMNS)

    Returns:
        Tuple of (column, value) pairs

    Example:
        >>> from misc.io_utils import compute_run_id_tuple
        >>> row = {"EXP_DATASET": 1, "EXP_STRATEGY": 2}
        >>> run_id = compute_run_id_tuple(row)
        >>> print(run_id)
        (('EXP_BATCH_SIZE', 5), ('EXP_DATASET', 1), ...)
    """
    if identity_columns is None:
        identity_columns = IDENTITY_COLUMNS

    identity_parts = []
    for col in sorted(identity_columns):
        if col in row:
            val = row[col]
            if hasattr(val, "iloc"):
                val = val.iloc[0] if len(val) > 0 else val
            identity_parts.append((col, val))

    return tuple(identity_parts)


def validate_schema(
    results_path: Union[str, Path],
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate a results directory for schema compliance.

    Args:
        results_path: Path to results directory
        strict: If True, treat warnings as errors

    Returns:
        Tuple of (is_valid, list of error/warning messages)

    Example:
        >>> from misc.io_utils import validate_schema
        >>> is_valid, messages = validate_schema("/path/to/results")
        >>> if not is_valid:
        ...     for msg in messages:
        ...         print(msg)
    """
    results_path = Path(results_path)
    messages = []
    has_errors = False

    # Check required files
    workload_path = results_path / "05_done_workload.csv"
    if not workload_path.exists():
        messages.append("ERROR: 05_done_workload.csv not found")
        return False, messages

    # Load and check workload
    try:
        df = pd.read_csv(workload_path)
    except Exception as e:
        messages.append(f"ERROR: Cannot read workload: {e}")
        return False, messages

    # Check required columns
    missing_cols = set(WORKLOAD_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        messages.append(f"ERROR: Missing required columns: {missing_cols}")
        has_errors = True

    # Check for duplicate EXP_UNIQUE_ID
    if "EXP_UNIQUE_ID" in df.columns:
        if df["EXP_UNIQUE_ID"].duplicated().any():
            dup_count = df["EXP_UNIQUE_ID"].duplicated().sum()
            messages.append(f"ERROR: {dup_count} duplicate EXP_UNIQUE_ID values")
            has_errors = True

    # Check config file (warning only)
    config_path = results_path / "00_config.yaml"
    if not config_path.exists():
        msg = "WARNING: 00_config.yaml not found"
        messages.append(msg)
        if strict:
            has_errors = True

    return not has_errors, messages


def load_metric_file(
    output_path: Union[str, Path],
    strategy: str,
    dataset: str,
    metric: str,
) -> Optional[pd.DataFrame]:
    """
    Load a specific metric file from the results structure.

    Args:
        output_path: Base output directory
        strategy: Strategy name (e.g., "ALIPY_RANDOM")
        dataset: Dataset name (e.g., "Iris")
        metric: Metric name (e.g., "accuracy")

    Returns:
        DataFrame with metric data, or None if file doesn't exist

    Example:
        >>> from misc.io_utils import load_metric_file
        >>> df = load_metric_file(config.OUTPUT_PATH, "ALIPY_RANDOM", "Iris", "accuracy")
        >>> if df is not None:
        ...     print(df.head())
    """
    base_path = Path(output_path) / strategy / dataset

    # Try different file extensions
    for ext in [".csv.xz", ".csv", ".csv.xz.parquet"]:
        filepath = base_path / f"{metric}{ext}"
        if filepath.exists():
            try:
                if ext.endswith(".parquet"):
                    return pd.read_parquet(filepath)
                else:
                    return pd.read_csv(filepath)
            except Exception:
                continue

    return None


def get_strategy_dataset_pairs(output_path: Union[str, Path]) -> List[Tuple[str, str]]:
    """
    Get all (strategy, dataset) pairs that have results.

    Args:
        output_path: Base output directory

    Returns:
        List of (strategy_name, dataset_name) tuples

    Example:
        >>> from misc.io_utils import get_strategy_dataset_pairs
        >>> pairs = get_strategy_dataset_pairs(config.OUTPUT_PATH)
        >>> print(pairs[:3])
        [('ALIPY_RANDOM', 'Iris'), ('ALIPY_RANDOM', 'wine_origin'), ...]
    """
    output_path = Path(output_path)
    pairs = []

    # Directories to exclude
    excluded = {"workloads", "plots", "metrics", "_TS"}

    for strategy_dir in output_path.iterdir():
        if not strategy_dir.is_dir():
            continue
        if strategy_dir.name in excluded:
            continue
        if strategy_dir.name.startswith("0"):
            continue

        for dataset_dir in strategy_dir.iterdir():
            if dataset_dir.is_dir():
                pairs.append((strategy_dir.name, dataset_dir.name))

    return sorted(pairs)


def merge_workloads(
    primary_path: Union[str, Path],
    secondary_path: Union[str, Path],
    output_path: Union[str, Path],
    check_duplicates: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge two workload files, checking for duplicates.

    Args:
        primary_path: Path to primary results directory
        secondary_path: Path to secondary results directory
        output_path: Path for merged output
        check_duplicates: If True, check for duplicate identities

    Returns:
        Tuple of (merged DataFrame, list of warning messages)

    Raises:
        ValueError: If duplicate EXP_UNIQUE_ID values found

    Example:
        >>> from misc.io_utils import merge_workloads
        >>> merged, warnings = merge_workloads(
        ...     "/path/to/exp1", "/path/to/exp2", "/path/to/merged"
        ... )
        >>> print(f"Merged {len(merged)} experiments")
    """
    warnings = []

    # Load workloads
    primary_df = load_results(primary_path, "05_done_workload.csv")
    secondary_df = load_results(secondary_path, "05_done_workload.csv")

    # Check for EXP_UNIQUE_ID overlap
    if check_duplicates:
        primary_ids = set(primary_df["EXP_UNIQUE_ID"].tolist())
        secondary_ids = set(secondary_df["EXP_UNIQUE_ID"].tolist())
        overlap = primary_ids & secondary_ids

        if overlap:
            raise ValueError(
                f"Cannot merge: {len(overlap)} overlapping EXP_UNIQUE_ID values. "
                f"Examples: {list(overlap)[:5]}"
            )

    # Check for identity overlap
    identity_cols = [c for c in IDENTITY_COLUMNS if c in primary_df.columns and c in secondary_df.columns]
    if identity_cols:
        primary_identities = set(
            tuple(row) for row in primary_df[identity_cols].values.tolist()
        )
        secondary_identities = set(
            tuple(row) for row in secondary_df[identity_cols].values.tolist()
        )
        identity_overlap = primary_identities & secondary_identities

        if identity_overlap:
            warnings.append(
                f"WARNING: {len(identity_overlap)} overlapping run identities found"
            )

    # Merge
    merged = pd.concat([primary_df, secondary_df], ignore_index=True)

    # Write output
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path / "05_done_workload.csv", index=False)

    return merged, warnings


def load_config(output_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load experiment configuration from 00_config.yaml.

    Args:
        output_path: Base output directory

    Returns:
        Configuration dictionary, or None if file doesn't exist

    Example:
        >>> from misc.io_utils import load_config
        >>> config = load_config("/path/to/results")
        >>> if config:
        ...     print(config.get("EXP_TITLE"))
    """
    config_path = Path(output_path) / "00_config.yaml"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None
