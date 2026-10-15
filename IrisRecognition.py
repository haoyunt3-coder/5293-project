"""Run the complete CASIA-IrisV1 identification and performance evaluation pipeline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from FeatureExtraction import FeatureConfig
from IrisMatching import MatchingConfig, load_model, prepare_features, run_matching
from PerformanceEvaluation import evaluate_performance


def run_recognition(data_dir: str | Path, output_dir: str | Path,
                    matching_config: MatchingConfig | None = None,
                    feature_config: FeatureConfig | None = None,
                    rebuild_cache: bool = False, evaluate: bool = True,
                    dimensions: list[int] | None = None,
                    fmr_targets: tuple[float, ...] = (0.01, 0.001, 0.0001, 0.0)) -> dict:
    """Reuse checked features, fit session 1, test session 2, and save requested outputs."""
    cfg, feature_cfg = matching_config or MatchingConfig(), feature_config or FeatureConfig()
    if not isinstance(cfg.n_components, int) or not 1 <= cfg.n_components <= 107:
        raise ValueError("Fisher components must be an integer from 1 to 107.")
    if not np.isfinite(cfg.regularization) or cfg.regularization <= 0:
        raise ValueError("Regularization must be positive and finite.")
    if not all(np.isfinite(f) and 0 < f < 0.5 for f in feature_cfg.channel_frequencies):
        raise ValueError("Filter frequencies must be finite and between 0 and 0.5.")
    if evaluate:
        if dimensions is not None and (not dimensions or any(
                not isinstance(d, int) or not 1 <= d <= cfg.n_components for d in dimensions)):
            raise ValueError("Evaluation dimensions must be integers within the fitted Fisher dimensions.")
        if not fmr_targets or any(not np.isfinite(t) or not 0 <= t <= 1 for t in fmr_targets):
            raise ValueError("FMR targets must be finite fractions between 0 and 1.")

    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    matching_dir, evaluation_dir = output_dir / "matching", output_dir / "evaluation"
    started = time.perf_counter()
    print("[1/3] Checking dataset and preparing preprocessing features...", flush=True)
    # prepare_features calls localization, normalization, enhancement, and extraction.
    cache = prepare_features(data_dir, matching_dir / "feature_cache.npz",
                             feature_cfg, cfg, rebuild_cache)
    prepared = time.perf_counter()
    print("[2/3] Fitting on session 1 and identifying session 2...", flush=True)
    matching = run_matching(cache, cfg, matching_dir)
    matched = time.perf_counter()
    for metric, result in matching["metrics"].items():
        print(f"{metric}: {result['correct']}/{result['total']} = {result['crr_percent']:.2f}%", flush=True)
    evaluation = None
    if evaluate:
        print("[3/3] Computing identification tables, dimension curves, and verification ROC...", flush=True)
        model = load_model(matching_dir / "fisher_model.npz")
        evaluation = evaluate_performance(cache, model, evaluation_dir, dimensions, fmr_targets)
    else:
        print("[3/3] Performance evaluation skipped by request.", flush=True)
    finished = time.perf_counter()

    # Record the configuration and provenance of this run, including skipped evaluation.
    module_dir = Path(__file__).resolve().parent
    source_names = ["IrisRecognition.py", "IrisLocalization.py", "IrisNormalization.py",
                    "ImageEnhancement.py", "FeatureExtraction.py", "IrisMatching.py",
                    "PerformanceEvaluation.py"]
    summary = {
        "status": "completed", "data_directory": str(data_dir), "output_directory": str(output_dir),
        "matching_configuration": asdict(cfg), "feature_configuration": asdict(feature_cfg),
        "matching": matching, "evaluation": evaluation,
        "evaluation_performed": evaluate,
        "evaluation_dimensions": (list(range(1, cfg.n_components + 1)) if dimensions is None
                                  else sorted(set(dimensions))) if evaluate else [],
        "fmr_targets": list(fmr_targets) if evaluate else [],
        "meets_80_percent_requirement": all(r["crr_percent"] >= 80 for r in matching["metrics"].values()),
        "seconds": {"features": prepared - started, "matching": matched - prepared,
                    "evaluation": finished - matched, "total": finished - started},
        "feature_signature_sha256": hashlib.sha256(str(cache["signature"]).encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((module_dir / name).read_bytes()).hexdigest()
                          for name in source_names},
        "split": "Session 1 fits mean, projection and templates; session 2 is used for evaluation only.",
        "output_note": "When evaluation is skipped, existing evaluation files may belong to an earlier run.",
    }
    (output_dir / "recognition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"All three CRRs >= 80%: {summary['meets_80_percent_requirement']}", flush=True)
    print(f"Finished in {summary['seconds']['total']:.1f} seconds. Results: {output_dir}", flush=True)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "CASIA Iris Image Database (version 1.0)")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--components", type=int, default=107)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--frequencies", type=float, nargs=2, default=(0.1, 0.1))
    parser.add_argument("--rebuild-cache", action="store_true", help="Recompute all image features")
    parser.add_argument("--skip-evaluation", action="store_true", help="Only train and identify")
    parser.add_argument("--dimensions", type=int, nargs="+", help="Default: every fitted Fisher dimension")
    parser.add_argument("--fmr-targets", type=float, nargs="+", default=(0.01, 0.001, 0.0001, 0.0),
                        help="Fractions, e.g. 0.001 means 0.1 percent")
    args = parser.parse_args()
    if args.skip_evaluation and (args.dimensions is not None or tuple(args.fmr_targets) != (0.01, 0.001, 0.0001, 0.0)):
        parser.error("Evaluation parameters cannot be combined with --skip-evaluation.")
    try:
        run_recognition(args.data_dir, args.output_dir,
                        MatchingConfig(n_components=args.components, regularization=args.regularization),
                        FeatureConfig(channel_frequencies=tuple(args.frequencies)),
                        args.rebuild_cache, not args.skip_evaluation, args.dimensions, tuple(args.fmr_targets))
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        parser.exit(1, f"Recognition failed: {error}\n")


if __name__ == "__main__":
    main()
