"""
client.py  —  Client-side study aggregation wrapper

Sends all images in a study to the SageMaker endpoint, collects
per-image probabilities, averages them, and returns a single
study-level prediction.

Supports:
  - Single study prediction
  - Batch prediction over a directory of studies
  - Concurrent requests for throughput (ThreadPoolExecutor)
  - Retry logic with exponential backoff

Usage:
  from client import MURAClient

  client = MURAClient(endpoint_name="mura-efficientnet-b0")

  # Single study
  result = client.predict_study("MURA-v1.1/valid/XR_WRIST/patient00042/study1_positive")
  print(result)

  # Whole validation set
  results = client.predict_directory("MURA-v1.1/valid", workers=8)
"""

import base64
import io
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from PIL import Image


# ── Config ─────────────────────────────────────────────────────────────────────

REGION        = os.environ.get("AWS_REGION",    "us-east-1")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "mura-efficientnet-b0")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
BODY_PARTS = {
    "XR_ELBOW", "XR_FINGER", "XR_FOREARM",
    "XR_HAND",  "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _encode_image(image_path: Path) -> str:
    """Read an image file and return base64-encoded JPEG bytes."""
    img = Image.open(image_path).convert("RGB")

    # Re-encode as JPEG to keep payload size predictable
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _body_part_from_path(path: Path) -> str:
    for part in path.parts:
        if part in BODY_PARTS:
            return part
    return "UNKNOWN"


def _label_from_path(path: Path) -> int | None:
    """Return ground-truth label if available (validation/test sets), else None."""
    for part in path.parts:
        if "positive" in part:
            return 1
        if "negative" in part:
            return 0
    return None


# ── MURAClient ─────────────────────────────────────────────────────────────────

class MURAClient:
    """
    Wraps the SageMaker runtime client with study-level aggregation.

    Args:
        endpoint_name:  Name of the deployed SageMaker endpoint
        region:         AWS region
        threshold:      Classification threshold (default 0.5)
        max_retries:    Number of retries on throttling / transient errors
        retry_delay:    Base delay in seconds (doubled on each retry)
    """

    def __init__(
        self,
        endpoint_name: str = ENDPOINT_NAME,
        region: str = REGION,
        threshold: float = 0.5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.endpoint_name = endpoint_name
        self.threshold     = threshold
        self.max_retries   = max_retries
        self.retry_delay   = retry_delay

        self._runtime = boto3.client("sagemaker-runtime", region_name=region)

    # ── Single image ───────────────────────────────────────────────────────────

    def _invoke_image(self, image_path: Path, study_id: str) -> float:
        """
        Send one image to the endpoint. Returns the probability score.
        Retries on ThrottlingException and ServiceUnavailableException.
        """
        payload = json.dumps({
            "image":    _encode_image(image_path),
            "study_id": study_id,
        })

        delay = self.retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                response = self._runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Accept="application/json",
                    Body=payload,
                )
                result = json.loads(response["Body"].read())
                return result["probability"]

            except self._runtime.exceptions.ThrottlingException:
                if attempt == self.max_retries:
                    raise
                print(f"  Throttled — retrying in {delay:.1f}s (attempt {attempt+1})")
                time.sleep(delay)
                delay *= 2

            except Exception as e:
                if attempt == self.max_retries:
                    raise
                print(f"  Error: {e} — retrying in {delay:.1f}s")
                time.sleep(delay)
                delay *= 2

    # ── Single study ───────────────────────────────────────────────────────────

    def predict_study(
        self,
        study_dir: str | Path,
        threshold: float | None = None,
    ) -> dict:
        """
        Send all images in a study folder to the endpoint and aggregate.

        Args:
            study_dir:  Path to a study folder
                        e.g. MURA-v1.1/valid/XR_WRIST/patient00042/study1_positive
            threshold:  Override instance threshold for this call

        Returns:
            {
                "study_dir":  str,
                "body_part":  str,
                "mean_prob":  float,
                "pred":       int,    # 0 = normal, 1 = abnormal
                "label":      str,    # "normal" | "abnormal"
                "ground_truth": int | None,  # None if folder name has no label
                "n_images":   int,
                "image_probs": list[float],
            }
        """
        study_dir = Path(study_dir)
        threshold = threshold if threshold is not None else self.threshold

        images = sorted(
            p for p in study_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No images found in {study_dir}")

        study_id = study_dir.name
        probs = [self._invoke_image(img, study_id) for img in images]

        mean_prob = sum(probs) / len(probs)
        pred = int(mean_prob >= threshold)

        return {
            "study_dir":    str(study_dir),
            "body_part":    _body_part_from_path(study_dir),
            "mean_prob":    mean_prob,
            "pred":         pred,
            "label":        "abnormal" if pred == 1 else "normal",
            "ground_truth": _label_from_path(study_dir),
            "n_images":     len(images),
            "image_probs":  probs,
        }

    # ── Batch over a directory ─────────────────────────────────────────────────

    def predict_directory(
        self,
        root_dir: str | Path,
        body_parts: list[str] | None = None,
        workers: int = 4,
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Run predict_study over every study folder under root_dir.

        Args:
            root_dir:    e.g. "MURA-v1.1/valid"
            body_parts:  filter to specific body parts; None = all
            workers:     number of concurrent threads
            threshold:   override threshold for all studies

        Returns:
            List of study result dicts, one per study.
        """
        root_dir    = Path(root_dir)
        body_parts  = set(body_parts) if body_parts else BODY_PARTS
        threshold   = threshold if threshold is not None else self.threshold

        # Collect all study dirs
        study_dirs = [
            p for p in sorted(root_dir.rglob("*"))
            if p.is_dir()
            and _body_part_from_path(p) in body_parts
            and any(f.suffix.lower() in SUPPORTED_EXTENSIONS for f in p.iterdir() if f.is_file())
        ]

        if not study_dirs:
            raise ValueError(f"No study folders found under {root_dir}")

        print(f"Found {len(study_dirs)} studies under {root_dir}")
        print(f"Running with {workers} concurrent workers...\n")

        results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.predict_study, sd, threshold): sd
                for sd in study_dirs
            }
            for future in as_completed(futures):
                study_dir = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % 50 == 0 or completed == len(study_dirs):
                        print(f"  Progress: {completed}/{len(study_dirs)} studies")
                except Exception as e:
                    print(f"  FAILED: {study_dir} — {e}")

        return results

    # ── Metrics ────────────────────────────────────────────────────────────────

    @staticmethod
    def summarise(results: list[dict], verbose: bool = True) -> dict:
        """
        Compute kappa + accuracy from a list of predict_study results.
        Only studies with a ground_truth label are included.
        """
        from sklearn.metrics import cohen_kappa_score, accuracy_score

        labelled = [r for r in results if r["ground_truth"] is not None]
        if not labelled:
            print("No ground-truth labels found — skipping metrics.")
            return {}

        labels = [r["ground_truth"] for r in labelled]
        preds  = [r["pred"]         for r in labelled]

        overall_kappa    = cohen_kappa_score(labels, preds)
        overall_accuracy = accuracy_score(labels, preds)

        by_part: dict[str, dict] = defaultdict(lambda: {"labels": [], "preds": []})
        for r in labelled:
            by_part[r["body_part"]]["labels"].append(r["ground_truth"])
            by_part[r["body_part"]]["preds"].append(r["pred"])

        per_part = {}
        for bp, data in sorted(by_part.items()):
            per_part[bp] = {
                "kappa":     cohen_kappa_score(data["labels"], data["preds"]),
                "accuracy":  accuracy_score(data["labels"], data["preds"]),
                "n_studies": len(data["labels"]),
            }

        if verbose:
            print(f"\n{'='*55}")
            print(f"  Client-Side Study Evaluation")
            print(f"{'='*55}")
            print(f"  Labelled studies : {len(labelled)}")
            print(f"  Overall kappa    : {overall_kappa:.4f}")
            print(f"  Overall accuracy : {overall_accuracy:.4f}")
            print(f"\n  Per body part:")
            print(f"  {'Body Part':<16} {'Kappa':>8} {'Accuracy':>10} {'Studies':>9}")
            print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*9}")
            for bp, m in per_part.items():
                print(
                    f"  {bp:<16} {m['kappa']:>8.4f}"
                    f" {m['accuracy']:>10.4f} {m['n_studies']:>9}"
                )
            print(f"{'='*55}\n")

        return {
            "overall_kappa":    overall_kappa,
            "overall_accuracy": overall_accuracy,
            "per_part":         per_part,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run study-level inference via SageMaker endpoint")
    parser.add_argument("--endpoint",   default=ENDPOINT_NAME)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--workers",    type=int,   default=4)

    subparsers = parser.add_subparsers(dest="command")

    # predict a single study
    single = subparsers.add_parser("study", help="Predict a single study folder")
    single.add_argument("study_dir")

    # predict a whole directory
    batch = subparsers.add_parser("directory", help="Predict all studies under a directory")
    batch.add_argument("root_dir")
    batch.add_argument("--body-parts", nargs="+", default=None)

    args = parser.parse_args()

    client = MURAClient(
        endpoint_name=args.endpoint,
        threshold=args.threshold,
    )

    if args.command == "study":
        result = client.predict_study(args.study_dir)
        print(json.dumps(result, indent=2))

    elif args.command == "directory":
        results = client.predict_directory(
            root_dir=args.root_dir,
            body_parts=args.body_parts,
            workers=args.workers,
        )
        client.summarise(results)

    else:
        parser.print_help()
