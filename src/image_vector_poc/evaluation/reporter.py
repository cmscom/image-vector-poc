"""Evaluation result reporting and persistence."""

import json
from pathlib import Path

from .metrics import EvaluationMetrics


class EvaluationReporter:
    """Save and load evaluation results to/from JSON files."""

    def __init__(self, output_dir: Path | str = "data/evaluations"):
        """Initialize the reporter.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, metrics: EvaluationMetrics) -> Path:
        """Save evaluation metrics to JSON file.

        Args:
            metrics: EvaluationMetrics instance to save

        Returns:
            Path to the saved file
        """
        # Generate safe filename from model name
        safe_name = metrics.model_name.replace("/", "_").replace(":", "_")
        date_str = metrics.timestamp[:10]
        filename = f"{safe_name}_{date_str}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

        return filepath

    def load(self, filepath: Path | str) -> EvaluationMetrics:
        """Load evaluation metrics from JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            EvaluationMetrics instance
        """
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return EvaluationMetrics(**data)

    def load_all(self) -> list[EvaluationMetrics]:
        """Load all evaluation results from output directory.

        Returns:
            List of EvaluationMetrics sorted by timestamp
        """
        results = []
        for filepath in self.output_dir.glob("*.json"):
            try:
                results.append(self.load(filepath))
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Failed to load {filepath}: {e}")
        return sorted(results, key=lambda x: x.timestamp)

    def list_files(self) -> list[Path]:
        """List all evaluation result files.

        Returns:
            List of file paths
        """
        return sorted(self.output_dir.glob("*.json"))
