"""ModelGate registry and lineage.

Handles MLflow logging, champion promotion, and model reproduction.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from modelgate.verdict import Verdict

logger = logging.getLogger(__name__)


def _get_active_tracking_uri(tracking_uri: str | None = None) -> str:
    """Resolve active MLflow tracking URI."""
    if tracking_uri:
        return tracking_uri
    if "MLFLOW_TRACKING_URI" in os.environ:
        return os.environ["MLFLOW_TRACKING_URI"]
    return "sqlite:///mlflow.db"


def log_run_to_mlflow(
    project: str,
    verdict: Verdict,
    lineage_info: dict[str, Any],
    model: Any,
    seed: int,
    metrics: dict[str, float] | None = None,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> str | None:
    """Log a ModelGate run to MLflow.

    Args:
        project: Project/experiment name.
        verdict: The run verdict.
        lineage_info: Lineage metadata dict.
        model: The trained model.
        seed: Random seed used.
        metrics: Additional metrics to log.
        tracking_uri: Configured MLflow tracking URI.
        artifact_location: Optional directory for artifacts.

    Returns:
        MLflow run ID, or None if MLflow is unavailable.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("MLflow not available; skipping run logging")
        return None

    active_uri = _get_active_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(active_uri)

    # Ensure experiment exists with specified artifact location if needed
    try:
        client = mlflow.tracking.MlflowClient(active_uri)
        exp = client.get_experiment_by_name(project)
        if exp is None:
            client.create_experiment(project, artifact_location=artifact_location)
    except Exception:
        pass

    mlflow.set_experiment(project)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("seed", seed)
        mlflow.log_param("verdict", verdict.status)
        mlflow.log_param("git_sha", lineage_info.get("git_sha", ""))

        dvc_hash = lineage_info.get("dvc_hash", "")
        if dvc_hash:
            mlflow.log_param("dvc_hash", dvc_hash)
            mlflow.set_tag("dvc_hash", dvc_hash)

        # Log metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # Log check results as metrics
        for cr in verdict.check_results:
            for key, val in cr.metrics.items():
                try:
                    mlflow.log_metric(f"{cr.name}_{key}", float(val))
                except (ValueError, TypeError):
                    pass

        # Log the model
        try:
            mlflow.sklearn.log_model(model, "model", serialization_format="pickle")
        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")

        # Log verdict JSON as artifact
        verdict_json = json.dumps(verdict.to_dict(), indent=2)
        mlflow.log_text(verdict_json, "verdict.json")

        # Log lineage
        lineage_json = json.dumps(lineage_info, indent=2, default=str)
        mlflow.log_text(lineage_json, "lineage.json")

        # Tags
        mlflow.set_tag("verdict", verdict.status)
        mlflow.set_tag("git_sha", lineage_info.get("git_sha", ""))
        mlflow.set_tag("seed", str(seed))

        for dname, dhash in lineage_info.get("data_hashes", {}).items():
            mlflow.set_tag(f"data_hash_{dname}", dhash)

        return run.info.run_id


def promote_to_champion(
    project: str,
    run_id: str,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> bool:
    """Promote a model version to champion in the MLflow registry.

    Args:
        project: Project/model name in the registry.
        run_id: MLflow run ID to promote.
        alias: Alias to set (default: "champion").
        tracking_uri: Optional MLflow tracking URI.

    Returns:
        True if promotion succeeded.
    """
    try:
        import mlflow

        active_uri = _get_active_tracking_uri(tracking_uri)
        mlflow.set_tracking_uri(active_uri)
        client = mlflow.tracking.MlflowClient(active_uri)

        # Find the model version associated with this run
        model_name = project

        # Ensure the registered model exists
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)

        # Register the model from the run
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)

        # Set the alias
        client.set_registered_model_alias(model_name, alias, mv.version)

        logger.info(f"Promoted {model_name} version {mv.version} to '{alias}'")
        return True

    except ImportError:
        logger.error("MLflow not available; cannot promote model")
        return False
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        return False


def get_reproduce_command(run_id: str) -> str:
    """Generate the CLI command to reproduce a run.

    Args:
        run_id: MLflow run ID.

    Returns:
        CLI command string.
    """
    return f"modelgate reproduce {run_id}"
