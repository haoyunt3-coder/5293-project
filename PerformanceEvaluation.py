"""Identification tables, dimension curves, and empirical verification ROC on session 2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from IrisMatching import load_model, match_features, template_distances

METRICS = ("L1", "L2", "Cosine")


def verification_curve(distances: np.ndarray, genuine: np.ndarray) -> dict[str, np.ndarray]:
    """Accept distances <= threshold; group ties to preserve that decision rule."""
    scores = np.asarray(distances, dtype=np.float64).ravel()
    mask = np.asarray(genuine)
    if mask.dtype.kind != "b" or mask.shape != np.asarray(distances).shape:
        raise ValueError("Genuine mask must be boolean and match the distance array.")
    mask = mask.ravel()
    if scores.size == 0 or not np.isfinite(scores).all() or not mask.any() or mask.all():
        raise ValueError("ROC requires finite scores and both genuine and impostor comparisons.")
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_mask = scores[order], mask[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), scores.size - 1]
    true_accepts = np.cumsum(sorted_mask)[ends]
    false_accepts = np.cumsum(~sorted_mask)[ends]
    return {
        "threshold": np.r_[-np.inf, sorted_scores[ends]],
        "fmr": np.r_[0.0, false_accepts / (~mask).sum()],
        "fnmr": np.r_[1.0, 1.0 - true_accepts / mask.sum()],
        "true_accepts": np.r_[0, true_accepts],
        "false_accepts": np.r_[0, false_accepts],
    }


def operating_point(curve: dict[str, np.ndarray], target_fmr: float) -> dict:
    if not np.isfinite(target_fmr) or not 0 <= target_fmr <= 1:
        raise ValueError("Target FMR must lie between 0 and 1.")
    # The last feasible threshold maximizes true accepts under the empirical FMR cap.
    index = np.flatnonzero(curve["fmr"] <= target_fmr)[-1]
    return {"target_fmr_percent": 100 * target_fmr,
            "threshold": float(curve["threshold"][index]),
            "fmr_percent": 100 * float(curve["fmr"][index]),
            "fnmr_percent": 100 * float(curve["fnmr"][index]),
            "false_accepts": int(curve["false_accepts"][index]),
            "true_accepts": int(curve["true_accepts"][index])}


def curve_statistics(curve: dict[str, np.ndarray]) -> dict:
    fmr, fnmr = curve["fmr"], curve["fnmr"]
    difference = fmr - fnmr
    right = int(np.flatnonzero(difference >= 0)[0])
    left = max(0, right - 1)
    weight = 0.0 if left == right else -difference[left] / (difference[right] - difference[left])
    eer = fmr[left] + weight * (fmr[right] - fmr[left])
    # Integrate the piecewise linear ROC, including tie groups and both endpoints.
    tpr = 1 - fnmr
    auc = np.sum(np.diff(fmr) * (tpr[1:] + tpr[:-1]) / 2)
    return {"roc_auc": float(auc), "interpolated_eer_percent": float(100 * eer)}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output_dir: Path, dimension_rows: list[dict], curves: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"L1": ("#235789", "-", "o"), "L2": ("#B66A18", "--", "s"),
              "Cosine": ("#7B4F86", "-.", "^")}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(8.2, 5.2), layout="constrained")
    for metric in METRICS:
        rows = [row for row in dimension_rows if row["metric"] == metric]
        color, line, marker = styles[metric]
        ax.plot([r["dimensions"] for r in rows], [r["crr_percent"] for r in rows],
                label=metric, color=color, linestyle=line, marker=marker, markersize=3)
    ax.set(xlabel="Fisher dimensions", ylabel="Correct recognition rate (%)",
           title="Fig. 10: Identification versus dimensions (432 test images)", ylim=(0, 100))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(loc="lower right")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"Fig10_dimensions.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5), layout="constrained")
    for metric in METRICS:
        curve = curves[metric]
        color, line, _ = styles[metric]
        axes[0].plot(100 * curve["fmr"], 100 * (1 - curve["fnmr"]),
                     color=color, linestyle=line, label=metric)
        axes[1].plot(100 * curve["fmr"], 100 * curve["fnmr"],
                     color=color, linestyle=line, label=metric)
    axes[0].set(title="Full verification ROC", xlabel="False match rate (%)",
                ylabel="True match rate (%)", xlim=(0, 100), ylim=(0, 100))
    axes[1].set(title="Error tradeoff: low-FMR detail", xlabel="False match rate (%)",
                ylabel="False non-match rate (%)", xlim=(0, 1), ylim=(0, 100))
    for ax in axes:
        ax.grid(color="#DDDDDD", linewidth=0.6)
        ax.legend(loc="lower right" if ax is axes[0] else "upper right")
    fig.suptitle("Fig. 11: Verification (432 genuine; 46,224 impostor comparisons)")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"Fig11_ROC.{suffix}", dpi=180)
    plt.close(fig)


def evaluate_performance(cache: dict, model, output_dir: str | Path,
                         dimensions=None, fmr_targets=(0.01, 0.001, 0.0001, 0.0)) -> dict:
    output_dir = Path(output_dir)
    features = np.asarray(cache["features"], dtype=np.float64)
    labels, sessions = cache["labels"], cache["sessions"]
    train, test = sessions == 1, sessions == 2
    rotations = np.asarray(cache["rotated_training_features"], dtype=np.float64)
    if (features.ndim != 2 or labels.shape != sessions.shape or labels.shape != (len(features),)
            or train.sum() != 324 or test.sum() != 432 or not np.all(train | test)
            or not np.isfinite(features).all() or not np.isfinite(rotations).all()
            or rotations.shape != (324, model.angles.size, features.shape[1])
            or not np.array_equal(cache["angles"], model.angles)
            or not np.array_equal(np.unique(labels[train]), model.classes)
            or not np.array_equal(np.unique(labels[test]), model.classes)):
        raise ValueError("Cache and model must describe the same complete assignment dataset.")
    for eye in model.classes:
        if ((labels[train] == eye).sum(), (labels[test] == eye).sum()) != (3, 4):
            raise ValueError("Expected three training and four test images per eye.")
    raw_centers = np.stack([rotations[labels[train] == eye].mean(axis=0) for eye in model.classes])
    if (not np.allclose(features[train].mean(axis=0), model.mean, rtol=1e-10, atol=1e-10)
            or not np.allclose((raw_centers - model.mean) @ model.projection,
                               model.centers, rtol=1e-8, atol=1e-8)):
        raise ValueError("Saved model was not fitted to this cache's training templates.")
    maximum = model.projection.shape[1]
    dimensions = list(range(1, maximum + 1)) if dimensions is None else sorted(set(dimensions))
    if not dimensions or any(not isinstance(d, (int, np.integer)) or not 1 <= d <= maximum for d in dimensions):
        raise ValueError(f"Dimensions must be integers from 1 to {maximum}.")
    if not fmr_targets or any(not np.isfinite(t) or not 0 <= t <= 1 for t in fmr_targets):
        raise ValueError("Provide at least one finite FMR target between 0 and 1.")
    output_dir.mkdir(parents=True, exist_ok=True)
    queries, true_labels = features[test], labels[test]
    genuine = true_labels[:, None] == model.classes[None, :]
    table3, table4, dimension_rows, curves, verification = [], [], [], {}, {}
    for metric in METRICS:
        original = template_distances(queries, raw_centers, metric).min(axis=2)
        reduced = match_features(queries, model, metric)
        for space, dimension, predictions in [
                ("Original", features.shape[1], model.classes[original.argmin(axis=1)]),
                ("Fisher", maximum, reduced.predictions)]:
            correct = int((predictions == true_labels).sum())
            table3.append({"metric": metric, "feature_space": space, "dimensions": dimension,
                           "correct": correct, "total": len(queries),
                           "crr_percent": 100 * correct / len(queries)})
        for dimension in dimensions:
            result = reduced if dimension == maximum else match_features(queries, model, metric, dimension)
            correct = int((result.predictions == true_labels).sum())
            dimension_rows.append({"metric": metric, "dimensions": dimension, "correct": correct,
                                   "total": len(queries), "crr_percent": 100 * correct / len(queries)})
        curves[metric] = verification_curve(reduced.class_distances, genuine)
        verification[metric] = curve_statistics(curves[metric])
        for target in fmr_targets:
            table4.append({"metric": metric, **operating_point(curves[metric], target)})
        write_csv(output_dir / f"ROC_{metric}.csv",
                  [{key: values[i].item() for key, values in curves[metric].items()}
                   for i in range(len(curves[metric]["threshold"]))])
    write_csv(output_dir / "Table3_identification.csv", table3)
    write_csv(output_dir / "Table4_verification.csv", table4)
    write_csv(output_dir / "dimension_curve.csv", dimension_rows)
    plot_curves(output_dir, dimension_rows, curves)
    summary = {"training_samples": int(train.sum()), "test_samples": int(test.sum()),
               "classes": model.classes.size, "fisher_dimensions": maximum,
               "genuine_comparisons": int(genuine.sum()), "impostor_comparisons": int((~genuine).sum()),
               "minimum_nonzero_fmr_percent": 100 / int((~genuine).sum()),
               "table3": table3, "table4": table4, "verification": verification,
               "accept_rule": "distance <= threshold; minimum over seven class templates",
               "threshold_policy": "Descriptive session-2 ROC operating points, not independent threshold calibration",
               "dimension_policy": "Prefixes of a fixed session-1 Fisher model; test curve does not select final dimension",
               "original_policy": "Raw class means at seven angles, without centering or Fisher projection",
               "figure_number_note": "Assignment Fig11 ROC corresponds to paper Fig13; paper Fig11 compares filters",
               "l2_note": "Squared Euclidean distance; thresholds use squared units"}
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-dir", type=Path, default=root / "outputs" / "matching")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "evaluation")
    parser.add_argument("--dimensions", type=int, nargs="+", help="Default: every available Fisher dimension")
    parser.add_argument("--fmr-targets", type=float, nargs="+", default=[0.01, 0.001, 0.0001, 0.0],
                        help="FMR fractions, e.g. 0.001 means 0.1 percent")
    args = parser.parse_args()
    try:
        with np.load(args.matching_dir / "feature_cache.npz", allow_pickle=False) as saved:
            cache = {key: saved[key] for key in saved.files}
        model = load_model(args.matching_dir / "fisher_model.npz")
        summary = evaluate_performance(cache, model, args.output_dir, args.dimensions, args.fmr_targets)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Evaluation failed: {error}\nRun IrisMatching.py first to create compatible cache and model files.\n")
    for row in summary["table3"]:
        print(f'{row["metric"]:6s} {row["feature_space"]:8s} {row["dimensions"]:4d} dimensions: '
              f'{row["correct"]}/{row["total"]} = {row["crr_percent"]:.2f}%')
    print(f"Evaluation results: {args.output_dir}")


if __name__ == "__main__":
    main()
