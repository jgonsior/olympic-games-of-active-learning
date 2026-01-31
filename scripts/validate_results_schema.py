#!/usr/bin/env python
"""
OGAL Results Schema Validator

Validates experiment results for schema compliance before merging into the shared dataset.

Usage:
    python scripts/validate_results_schema.py --results_path /path/to/results
    python scripts/validate_results_schema.py --results_path /path/to/new --compare_with /path/to/existing
    python scripts/validate_results_schema.py --results_path /path/to/results --strict

Exit codes:
    0 - All checks passed
    1 - Errors found (must fix before proceeding)
    2 - Warnings only (review before proceeding)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd
import yaml


# Required columns in workload files
WORKLOAD_REQUIRED_COLUMNS = [
    "EXP_UNIQUE_ID",
    "EXP_DATASET",
    "EXP_STRATEGY",
    "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    "EXP_RANDOM_SEED",
    "EXP_START_POINT",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "EXP_NUM_QUERIES",
]

# Identity columns (primary key components)
IDENTITY_COLUMNS = [
    "EXP_DATASET",
    "EXP_STRATEGY",
    "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    "EXP_START_POINT",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "EXP_RANDOM_SEED",
]


class ValidationResult:
    """Stores validation results."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def add_error(self, message: str):
        self.errors.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def add_info(self, message: str):
        self.info.append(message)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def print_summary(self):
        if self.info:
            for msg in self.info:
                print(f"  ✓ {msg}")
        if self.warnings:
            for msg in self.warnings:
                print(f"  ⚠ WARNING: {msg}")
        if self.errors:
            for msg in self.errors:
                print(f"  ✗ ERROR: {msg}")


def check_required_files(results_path: Path) -> ValidationResult:
    """Check that required files exist."""
    result = ValidationResult()

    # Required files
    required_files = [
        ("05_done_workload.csv", "Completed experiments workload"),
    ]

    # Optional but recommended files
    optional_files = [
        ("00_config.yaml", "Experiment configuration"),
        ("01_workload.csv", "Full workload definition"),
    ]

    for filename, description in required_files:
        filepath = results_path / filename
        if filepath.exists():
            result.add_info(f"{filename} exists ({description})")
        else:
            result.add_error(f"{filename} missing ({description})")

    for filename, description in optional_files:
        filepath = results_path / filename
        if filepath.exists():
            result.add_info(f"{filename} exists ({description})")
        else:
            result.add_warning(f"{filename} missing ({description})")

    return result


def check_workload_schema(results_path: Path) -> Tuple[ValidationResult, Optional[pd.DataFrame]]:
    """Check workload file schema and return the dataframe."""
    result = ValidationResult()
    workload_path = results_path / "05_done_workload.csv"

    if not workload_path.exists():
        result.add_error("Cannot check workload schema - file missing")
        return result, None

    try:
        df = pd.read_csv(workload_path)
    except Exception as e:
        result.add_error(f"Cannot read workload CSV: {e}")
        return result, None

    # Check required columns
    missing_cols = set(WORKLOAD_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        result.add_error(f"Missing required columns: {missing_cols}")
    else:
        result.add_info("All required columns present")

    # Check identity columns
    missing_identity_cols = set(IDENTITY_COLUMNS) - set(df.columns)
    if missing_identity_cols:
        result.add_error(f"Missing identity columns: {missing_identity_cols}")
    else:
        result.add_info("All identity columns present")

    # Check for EXP_UNIQUE_ID
    if "EXP_UNIQUE_ID" in df.columns:
        unique_count = df["EXP_UNIQUE_ID"].nunique()
        total_count = len(df)
        result.add_info(f"{unique_count} unique experiments (total rows: {total_count})")

        if unique_count != total_count:
            duplicates = df[df.duplicated(subset=["EXP_UNIQUE_ID"], keep=False)]
            dup_ids = duplicates["EXP_UNIQUE_ID"].unique()[:5]
            result.add_error(
                f"Duplicate EXP_UNIQUE_ID values found: {len(df) - unique_count} duplicates. "
                f"Examples: {list(dup_ids)}"
            )
    else:
        result.add_error("EXP_UNIQUE_ID column not found")
        df = None

    return result, df


def check_metric_files(
    results_path: Path, workload_df: Optional[pd.DataFrame]
) -> ValidationResult:
    """Check metric files for schema compliance."""
    result = ValidationResult()

    # Find strategy directories (exclude special directories)
    excluded_dirs = {"workloads", "plots", "metrics", "_TS"}
    strategy_dirs = [
        d
        for d in results_path.iterdir()
        if d.is_dir() and d.name not in excluded_dirs and not d.name.startswith("0")
    ]

    if not strategy_dirs:
        result.add_warning("No strategy directories found")
        return result

    result.add_info(f"{len(strategy_dirs)} strategy directories found")

    # Get valid EXP_UNIQUE_IDs from workload
    valid_ids: Optional[Set[int]] = None
    if workload_df is not None and "EXP_UNIQUE_ID" in workload_df.columns:
        valid_ids = set(workload_df["EXP_UNIQUE_ID"].astype(int).tolist())

    # Sample check: verify a few metric files
    checked_files = 0
    max_checks = 50  # Limit checks for performance
    metric_ids_found: Set[int] = set()

    for strategy_dir in strategy_dirs:
        if checked_files >= max_checks:
            break

        for dataset_dir in strategy_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            if checked_files >= max_checks:
                break

            # Check for required metric files
            for metric_name in ["accuracy", "weighted_f1-score"]:
                for ext in [".csv.xz", ".csv"]:
                    metric_path = dataset_dir / f"{metric_name}{ext}"
                    if metric_path.exists():
                        try:
                            df = pd.read_csv(metric_path)
                            if "EXP_UNIQUE_ID" not in df.columns:
                                result.add_error(
                                    f"Missing EXP_UNIQUE_ID in {metric_path.relative_to(results_path)}"
                                )
                            else:
                                # Collect IDs for cross-check
                                metric_ids_found.update(
                                    df["EXP_UNIQUE_ID"].astype(int).tolist()
                                )
                                result.add_info(
                                    f"Checking {metric_path.relative_to(results_path)}... OK"
                                )
                            checked_files += 1
                        except Exception as e:
                            result.add_error(
                                f"Cannot read {metric_path.relative_to(results_path)}: {e}"
                            )
                        break  # Only check one extension

    if checked_files == 0:
        result.add_warning("No metric files found to check")

    # Cross-check: verify metric IDs exist in workload
    if valid_ids is not None and metric_ids_found:
        orphan_ids = metric_ids_found - valid_ids
        if orphan_ids:
            sample_orphans = list(orphan_ids)[:5]
            result.add_warning(
                f"{len(orphan_ids)} EXP_UNIQUE_ID values in metrics not found in workload. "
                f"Examples: {sample_orphans}"
            )
        else:
            result.add_info("All EXP_UNIQUE_ID values in metrics exist in workload")

    return result


def check_config_file(results_path: Path) -> ValidationResult:
    """Check config file for validity."""
    result = ValidationResult()
    config_path = results_path / "00_config.yaml"

    if not config_path.exists():
        result.add_warning("00_config.yaml not found - provenance tracking limited")
        return result

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if config is None:
            result.add_warning("00_config.yaml is empty")
            return result

        # Check for recommended fields
        if "GIT_COMMIT_HASH" in config:
            result.add_info(f"Git commit: {config['GIT_COMMIT_HASH']}")
        else:
            result.add_warning("GIT_COMMIT_HASH not in config (provenance tracking)")

        result.add_info("00_config.yaml is valid YAML")

    except yaml.YAMLError as e:
        result.add_error(f"Invalid YAML in 00_config.yaml: {e}")
    except Exception as e:
        result.add_error(f"Cannot read 00_config.yaml: {e}")

    return result


def check_duplicate_identities(
    results_path: Path, compare_path: Optional[Path]
) -> ValidationResult:
    """Check for duplicate run identities within and across datasets."""
    result = ValidationResult()

    workload_path = results_path / "05_done_workload.csv"
    if not workload_path.exists():
        result.add_warning("Cannot check duplicates - workload missing")
        return result

    try:
        df = pd.read_csv(workload_path)
    except Exception as e:
        result.add_error(f"Cannot read workload: {e}")
        return result

    # Check for duplicate identities within this dataset
    identity_cols = [c for c in IDENTITY_COLUMNS if c in df.columns]
    if identity_cols:
        duplicates = df[df.duplicated(subset=identity_cols, keep=False)]
        if len(duplicates) > 0:
            result.add_warning(
                f"{len(duplicates)} rows have duplicate run identities (same hyperparameters)"
            )
        else:
            result.add_info("No duplicate run identities within dataset")

    # Compare with existing dataset if provided
    if compare_path is not None:
        compare_workload = compare_path / "05_done_workload.csv"
        if compare_workload.exists():
            try:
                existing_df = pd.read_csv(compare_workload)

                # Check for EXP_UNIQUE_ID overlap
                if "EXP_UNIQUE_ID" in df.columns and "EXP_UNIQUE_ID" in existing_df.columns:
                    new_ids = set(df["EXP_UNIQUE_ID"].tolist())
                    existing_ids = set(existing_df["EXP_UNIQUE_ID"].tolist())
                    overlap = new_ids & existing_ids

                    if overlap:
                        sample = list(overlap)[:5]
                        result.add_warning(
                            f"{len(overlap)} EXP_UNIQUE_ID values overlap with comparison dataset. "
                            f"Examples: {sample}"
                        )
                    else:
                        result.add_info("No EXP_UNIQUE_ID overlap with comparison dataset")

                # Check for identity overlap
                identity_cols = [
                    c for c in IDENTITY_COLUMNS if c in df.columns and c in existing_df.columns
                ]
                if identity_cols:
                    # Create identity tuples
                    new_identities = set(
                        tuple(row) for row in df[identity_cols].values.tolist()
                    )
                    existing_identities = set(
                        tuple(row) for row in existing_df[identity_cols].values.tolist()
                    )
                    identity_overlap = new_identities & existing_identities

                    if identity_overlap:
                        result.add_warning(
                            f"{len(identity_overlap)} run identities overlap with comparison dataset"
                        )
                    else:
                        result.add_info("No run identity overlap with comparison dataset")

            except Exception as e:
                result.add_warning(f"Cannot compare with existing dataset: {e}")
        else:
            result.add_warning(
                f"Comparison workload not found: {compare_workload}"
            )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate OGAL experiment results schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        required=True,
        help="Path to results directory to validate",
    )
    parser.add_argument(
        "--compare_with",
        type=Path,
        default=None,
        help="Path to existing results for duplicate checking",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    args = parser.parse_args()

    print("=== OGAL Results Schema Validator ===")
    print(f"Results path: {args.results_path}")
    if args.compare_with:
        print(f"Comparing with: {args.compare_with}")
    print()

    # Verify paths exist
    if not args.results_path.exists():
        print(f"ERROR: Results path does not exist: {args.results_path}")
        sys.exit(1)

    if args.compare_with and not args.compare_with.exists():
        print(f"WARNING: Comparison path does not exist: {args.compare_with}")

    all_results: List[ValidationResult] = []

    # Check 1: Required files
    print("[CHECK] Required files...")
    result = check_required_files(args.results_path)
    result.print_summary()
    all_results.append(result)
    print()

    # Check 2: Config file
    print("[CHECK] Config file...")
    result = check_config_file(args.results_path)
    result.print_summary()
    all_results.append(result)
    print()

    # Check 3: Workload schema
    print("[CHECK] Workload schema...")
    result, workload_df = check_workload_schema(args.results_path)
    result.print_summary()
    all_results.append(result)
    print()

    # Check 4: Metric files
    print("[CHECK] Metric files...")
    result = check_metric_files(args.results_path, workload_df)
    result.print_summary()
    all_results.append(result)
    print()

    # Check 5: Duplicate identities
    print("[CHECK] Duplicate identities...")
    result = check_duplicate_identities(args.results_path, args.compare_with)
    result.print_summary()
    all_results.append(result)
    print()

    # Summary
    total_errors = sum(len(r.errors) for r in all_results)
    total_warnings = sum(len(r.warnings) for r in all_results)

    if total_errors > 0:
        print(f"=== VALIDATION FAILED ({total_errors} errors, {total_warnings} warnings) ===")
        sys.exit(1)
    elif total_warnings > 0:
        if args.strict:
            print(f"=== VALIDATION FAILED (strict mode: {total_warnings} warnings) ===")
            sys.exit(1)
        else:
            print(f"=== VALIDATION PASSED WITH WARNINGS ({total_warnings} warnings) ===")
            sys.exit(2)
    else:
        print("=== VALIDATION PASSED ===")
        sys.exit(0)


if __name__ == "__main__":
    main()
