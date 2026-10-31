"""Seed script for ModelGate Docker Compose environment.

Waits for API and MLflow services to become healthy, then generates demo data
and runs ModelGate quality checks so demo data appears in the dashboard and registry.
"""

from __future__ import annotations

import os
import subprocess
import time
import httpx

API_URL = os.getenv("API_URL", "http://api:8000")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def wait_for_service(url: str, name: str, max_retries: int = 30) -> bool:
    print(f"Waiting for {name} ({url})...")
    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
            if resp.status_code in (200, 301, 302):
                print(f"✓ {name} is ready.")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"⚠️ {name} did not become ready after {max_retries}s, proceeding anyway.")
    return False


def main() -> None:
    wait_for_service(API_URL, "ModelGate API")
    wait_for_service(MLFLOW_URL, "MLflow Server")

    print("\n1. Generating synthetic demo dataset...")
    subprocess.run(["python", "demo/train.py"], check=False)

    print("\n2. Executing ModelGate quality gate...")
    subprocess.run(["modelgate", "run", "--config", "modelgate.yaml"], check=False)

    print("\n✓ Demo data seeding completed successfully.")


if __name__ == "__main__":
    main()
