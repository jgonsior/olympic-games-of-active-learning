#!/usr/bin/env python
"""
OGAL Strategy Catalog Exporter

Exports the complete strategy catalog from the code registry to JSON/CSV/Markdown.
This ensures documentation stays synchronized with the actual strategy definitions.

Usage:
    python scripts/export_strategy_catalog.py --format json
    python scripts/export_strategy_catalog.py --format csv
    python scripts/export_strategy_catalog.py --format markdown

Source: scripts/export_strategy_catalog.py
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the parent directory to the path to allow imports from resources
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class StrategyInfo:
    """Represents a single AL strategy entry."""

    strategy_id: int
    strategy_name: str
    strategy_family: str
    backend_framework: str
    adapter_entrypoint: str
    upstream_url: str
    fork_url: Optional[str]
    python_class: str
    default_params: Dict[str, Any]
    notes: str


# Framework information with upstream and fork URLs
FRAMEWORK_INFO = {
    "ALIPY": {
        "upstream": "https://github.com/NUAA-AL/ALiPy",
        "fork": "https://github.com/jgonsior/ALiPy",
        "adapter": "framework_runners/alipy_runner.py::ALIPY_AL_Experiment",
    },
    "LIBACT": {
        "upstream": "https://github.com/ntucllab/libact",
        "fork": "https://github.com/jgonsior/libact",
        "adapter": "framework_runners/libact_runner.py::LIBACT_Experiment",
    },
    "PLAYGROUND": {
        "upstream": "https://github.com/google/active-learning",
        "fork": "https://github.com/jgonsior/active-learning",
        "adapter": "framework_runners/playground_runner.py::PLAYGROUND_AL_Experiment",
    },
    "SMALLTEXT": {
        "upstream": "https://github.com/webis-de/small-text",
        "fork": None,  # Uses PyPI package, no fork
        "adapter": "framework_runners/smalltext_runner.py::SMALLTEXT_AL_Experiment",
    },
    "SKACTIVEML": {
        "upstream": "https://github.com/scikit-activeml/scikit-activeml",
        "fork": "https://github.com/jgonsior/scikit-activeml",
        "adapter": "framework_runners/skactiveml_runner.py::SKACTIVEML_AL_Experiment",
    },
    "OPTIMAL": {
        "upstream": None,  # OGAL-native implementation
        "fork": None,
        "adapter": "framework_runners/optimal_runner.py::OPTIMAL_AL_Experiment",
    },
}

# Strategy family classification (based on paper taxonomy)
STRATEGY_FAMILIES = {
    # Uncertainty-based
    "RANDOM": "random",
    "UNCERTAINTY": "uncertainty",
    "LC": "uncertainty",
    "MM": "uncertainty",
    "ENTROPY": "uncertainty",
    "DTB": "uncertainty",
    "SM": "uncertainty",
    "LEASTCONFIDENCE": "uncertainty",
    "PREDICTIONENTROPY": "uncertainty",
    "BREAKINGTIES": "uncertainty",
    "MARGIN": "uncertainty",
    "US_": "uncertainty",
    # Committee-based
    "QBC": "committee",
    # Diversity/Density-based
    "DENSITY": "density",
    "GRAPH_DENSITY": "density",
    "CORESET": "diversity",
    "KCENTER": "diversity",
    "CLUSTERING": "diversity",
    "CLUSTER": "diversity",
    "MCM": "diversity",
    "UNIFORM": "random",
    "EMBEDDINGKMEANS": "diversity",
    # Hybrid/Meta
    "MIXTURE": "hybrid",
    "INFORMATIVE_DIVERSE": "hybrid",
    "BANDIT": "meta",
    "ALBL": "meta",
    "HIERARCHICAL": "hybrid",
    "CONTRASTIVE": "contrastive",
    "DISCRIMINATIVE": "discriminative",
    "DAL": "discriminative",
    # Model-based
    "EER": "expected_error",
    "VOI": "expected_error",
    "MC_EER": "expected_error",
    "EXPECTED": "expected_error",
    "PROBABILISTIC": "probabilistic",
    "MCPAL": "probabilistic",
    "COST_EMBEDDING": "cost_sensitive",
    # Query-specific
    "QUIRE": "query_by_committee",
    "DWUS": "density_weighted",
    # Optimal (oracle)
    "OPTIMAL": "oracle",
    "GREEDY": "oracle",
    "BSO": "oracle",
    "TRUE": "oracle",
    # Other
    "BMDR": "other",
    "SPAL": "other",
    "LAL": "learning_to_learn",
}


def classify_strategy_family(strategy_name: str) -> str:
    """Classify a strategy into a family based on its name."""
    name_upper = strategy_name.upper()
    for pattern, family in STRATEGY_FAMILIES.items():
        if pattern in name_upper:
            return family
    return "TODO(verify)"


def get_framework_from_strategy(strategy_name: str) -> str:
    """Extract the framework name from a strategy name."""
    # Derive prefixes from FRAMEWORK_INFO keys to maintain single source of truth
    prefixes = [f"{key}_" for key in FRAMEWORK_INFO.keys()]
    for prefix in prefixes:
        if strategy_name.startswith(prefix):
            return prefix.rstrip("_")
    return "UNKNOWN"


def export_strategy_catalog() -> List[StrategyInfo]:
    """Export the complete strategy catalog from the code registry."""
    # Import the strategy mappings
    try:
        from resources.data_types import (
            AL_STRATEGY,
            al_strategy_to_python_classes_mapping,
            al_strategies_which_only_support_binary_classification,
        )
    except ImportError as e:
        print(f"Error importing strategy definitions: {e}", file=sys.stderr)
        print("Make sure you're running from the repository root.", file=sys.stderr)
        sys.exit(1)

    # Convert to set for O(1) lookup
    binary_only_strategies = set(al_strategies_which_only_support_binary_classification)

    strategies = []

    for strategy in AL_STRATEGY:
        strategy_name = strategy.name
        strategy_id = strategy.value

        # Get framework
        framework = get_framework_from_strategy(strategy_name)
        framework_info = FRAMEWORK_INFO.get(framework, {})

        # Get class and params
        if strategy in al_strategy_to_python_classes_mapping:
            python_class, default_params = al_strategy_to_python_classes_mapping[strategy]
            class_name = f"{python_class.__module__}.{python_class.__name__}"
        else:
            class_name = "TODO(verify)"
            default_params = {}

        # Classify family
        family = classify_strategy_family(strategy_name)

        # Build notes using actual binary-only classification from code
        notes = ""
        if strategy in binary_only_strategies:
            notes = "Binary classification only"

        strategies.append(
            StrategyInfo(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                strategy_family=family,
                backend_framework=framework,
                adapter_entrypoint=framework_info.get("adapter", "TODO(verify)"),
                upstream_url=framework_info.get("upstream", ""),
                fork_url=framework_info.get("fork"),
                python_class=class_name,
                default_params=default_params,
                notes=notes,
            )
        )

    return sorted(strategies, key=lambda s: (s.backend_framework, s.strategy_id))


def output_json(strategies: List[StrategyInfo], output_file: Optional[str] = None):
    """Output strategies as JSON."""
    data = [
        {
            "id": s.strategy_id,
            "name": s.strategy_name,
            "family": s.strategy_family,
            "framework": s.backend_framework,
            "adapter": s.adapter_entrypoint,
            "upstream": s.upstream_url,
            "fork": s.fork_url,
            "class": s.python_class,
            "params": s.default_params,
            "notes": s.notes,
        }
        for s in strategies
    ]

    output = json.dumps(data, indent=2, default=str)
    if output_file:
        Path(output_file).write_text(output)
        print(f"Written to {output_file}")
    else:
        print(output)


def output_csv(strategies: List[StrategyInfo], output_file: Optional[str] = None):
    """Output strategies as CSV."""
    fieldnames = [
        "id",
        "name",
        "family",
        "framework",
        "adapter",
        "upstream",
        "fork",
        "class",
        "notes",
    ]

    if output_file:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in strategies:
                writer.writerow(
                    {
                        "id": s.strategy_id,
                        "name": s.strategy_name,
                        "family": s.strategy_family,
                        "framework": s.backend_framework,
                        "adapter": s.adapter_entrypoint,
                        "upstream": s.upstream_url or "",
                        "fork": s.fork_url or "",
                        "class": s.python_class,
                        "notes": s.notes,
                    }
                )
        print(f"Written to {output_file}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for s in strategies:
            writer.writerow(
                {
                    "id": s.strategy_id,
                    "name": s.strategy_name,
                    "family": s.strategy_family,
                    "framework": s.backend_framework,
                    "adapter": s.adapter_entrypoint,
                    "upstream": s.upstream_url or "",
                    "fork": s.fork_url or "",
                    "class": s.python_class,
                    "notes": s.notes,
                }
            )


def output_markdown(strategies: List[StrategyInfo], output_file: Optional[str] = None):
    """Output strategies as Markdown table."""
    lines = []
    lines.append("| ID | Strategy Name | Family | Framework | Adapter | Fork? |")
    lines.append("|---:|---------------|--------|-----------|---------|-------|")

    for s in strategies:
        fork_status = "Yes" if s.fork_url else "No"
        lines.append(
            f"| {s.strategy_id} | `{s.strategy_name}` | {s.strategy_family} | {s.backend_framework} | `{s.adapter_entrypoint}` | {fork_status} |"
        )

    output = "\n".join(lines)
    if output_file:
        Path(output_file).write_text(output)
        print(f"Written to {output_file}")
    else:
        print(output)


def main():
    parser = argparse.ArgumentParser(
        description="Export OGAL strategy catalog from code registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    strategies = export_strategy_catalog()

    if args.format == "json":
        output_json(strategies, args.output)
    elif args.format == "csv":
        output_csv(strategies, args.output)
    elif args.format == "markdown":
        output_markdown(strategies, args.output)


if __name__ == "__main__":
    main()
