"""Fit Fisher discriminants on session 1 and match session 2 iris samples."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from FeatureExtraction import FeatureConfig, extract_features
from ImageEnhancement import EnhancementConfig, enhance_iris
from IrisLocalization import LocalizationConfig, load_image, localize_iris
from IrisNormalization import NormalizationConfig, normalize_iris


@dataclass(frozen=True)
class MatchingConfig:
    n_components: int = 107  # At most C-1 Fisher dimensions for C classes.
    regularization: float = 0.01  # Ridge relative to the average within-class variance.
    rotation_angles: tuple[float, ...] = (-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0)


@dataclass(frozen=True)
class FisherModel:
    mean: np.ndarray
    projection: np.ndarray
    classes: np.ndarray
    centers: np.ndarray
    angles: np.ndarray
    eigenvalues: np.ndarray
    pca_dimension: int
    regularization: float
    ridge_value: float


@dataclass(frozen=True)
class MatchResult:
    predictions: np.ndarray
    class_distances: np.ndarray
    best_angles: np.ndarray


def _feature_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty, finite two-dimensional feature matrix.")
    return values


def fit_fisher(train_features: np.ndarray, train_labels: np.ndarray,
               rotated_features: np.ndarray | None = None,
               config: MatchingConfig | None = None) -> FisherModel:
    """Fit PCA plus regularized Fisher LDA, then average training templates by class."""
    cfg = config or MatchingConfig()
    features = _feature_matrix(train_features)
    labels = np.asarray(train_labels)
    if labels.ndim != 1 or labels.size != features.shape[0] or labels.dtype.kind not in "iu":
        raise ValueError("Training labels must be a one-dimensional integer array matching the samples.")
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        raise ValueError("Fisher training requires at least two classes and two samples per class.")
    if not np.isfinite(cfg.regularization) or cfg.regularization <= 0:
        raise ValueError("Regularization must be a positive finite number.")
    if not isinstance(cfg.n_components, int) or cfg.n_components < 1:
        raise ValueError("The number of Fisher dimensions must be a positive integer.")

    mean = features.mean(axis=0)
    centered = features - mean
    # PCA limits the small-sample problem to at most N-C dimensions, using training data only.
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    rank = int((singular_values > singular_values[0] * 1e-10).sum())
    pca_dimension = min(features.shape[0] - classes.size, features.shape[1], rank)
    max_dimensions = min(classes.size - 1, pca_dimension)
    if cfg.n_components > max_dimensions:
        raise ValueError(f"Requested {cfg.n_components} Fisher dimensions; this training set supports at most {max_dimensions}.")
    pca_basis = right_vectors[:pca_dimension].T
    projected = centered @ pca_basis
    class_means = np.stack([projected[inverse == i].mean(axis=0) for i in range(classes.size)])
    residuals = projected - class_means[inverse]
    within = residuals.T @ residuals / features.shape[0]
    offsets = class_means - projected.mean(axis=0)
    between = offsets.T @ (offsets * counts[:, None]) / features.shape[0]

    # Whiten the within-class scatter before solving the symmetric Fisher eigenproblem.
    ridge = cfg.regularization * float(np.trace(within)) / pca_dimension
    ridge = max(ridge, np.finfo(np.float64).eps)
    within_values, within_vectors = np.linalg.eigh(within)
    whitening = within_vectors / np.sqrt(np.maximum(within_values, 0) + ridge)
    fisher_scatter = whitening.T @ between @ whitening
    fisher_scatter = (fisher_scatter + fisher_scatter.T) / 2
    eigenvalues, fisher_vectors = np.linalg.eigh(fisher_scatter)
    order = np.argsort(eigenvalues)[::-1][:cfg.n_components]
    projection = pca_basis @ whitening @ fisher_vectors[:, order]

    if rotated_features is None:
        rotations = features[:, None, :]
        angles = np.array([0.0])
    else:
        rotations = np.asarray(rotated_features, dtype=np.float64)
        angles = np.asarray(cfg.rotation_angles, dtype=np.float64)
        expected_shape = (features.shape[0], angles.size, features.shape[1])
        if (rotations.shape != expected_shape or angles.ndim != 1 or angles.size == 0
                or not np.isfinite(rotations).all() or not np.isfinite(angles).all()
                or np.unique(angles).size != angles.size):
            raise ValueError("Rotated training features and angles are inconsistent or invalid.")
    rotated_projected = (rotations - mean) @ projection
    centers = np.stack([rotated_projected[inverse == i].mean(axis=0) for i in range(classes.size)])
    return FisherModel(mean, projection, classes, centers, angles, eigenvalues[order],
                       pca_dimension, cfg.regularization, ridge)


def template_distances(queries: np.ndarray, centers: np.ndarray, metric: str) -> np.ndarray:
    """Compute distances to a C x A x D center bank; L2 uses Eq. (8)'s squared value."""
    queries = _feature_matrix(queries)
    centers = np.asarray(centers, dtype=np.float64)
    if (centers.ndim != 3 or min(centers.shape) == 0 or centers.shape[2] != queries.shape[1]
            or not np.isfinite(centers).all()):
        raise ValueError("Center bank must have shape (classes, angles, dimensions) and match the queries.")
    metric = metric.lower()
    flat_centers = centers.reshape(-1, centers.shape[-1])
    if metric == "l1":
        distances = np.empty((queries.shape[0], flat_centers.shape[0]))
        chunk_size = max(1, min(32, 8_000_000 // (flat_centers.shape[0] * queries.shape[1])))
        for start in range(0, queries.shape[0], chunk_size):
            chunk = queries[start:start + chunk_size]
            distances[start:start + chunk_size] = np.abs(chunk[:, None] - flat_centers[None]).sum(axis=2)
    elif metric == "l2":
        distances = np.maximum((queries**2).sum(axis=1)[:, None]
                               + (flat_centers**2).sum(axis=1)[None]
                               - 2 * queries @ flat_centers.T, 0)
    elif metric == "cosine":
        query_norms = np.linalg.norm(queries, axis=1)
        center_norms = np.linalg.norm(flat_centers, axis=1)
        q_unit = np.divide(queries, query_norms[:, None], out=np.zeros_like(queries),
                           where=query_norms[:, None] > 1e-12)
        c_unit = np.divide(flat_centers, center_norms[:, None], out=np.zeros_like(flat_centers),
                           where=center_norms[:, None] > 1e-12)
        distances = 1 - np.clip(q_unit @ c_unit.T, -1, 1)
        distances[(query_norms <= 1e-12)[:, None] & (center_norms <= 1e-12)[None]] = 0
    else:
        raise ValueError("Metric must be l1, l2, or cosine.")
    return distances.reshape(queries.shape[0], centers.shape[0], centers.shape[1])


def match_features(features: np.ndarray, model: FisherModel, metric: str = "cosine",
                    n_components: int | None = None) -> MatchResult:
    """Take each class's minimum over rotation templates, then choose its nearest center."""
    features = np.asarray(features)
    if features.ndim == 1:
        features = features[None]
    features = _feature_matrix(features)
    dimension = model.projection.shape[1] if n_components is None else n_components
    if (features.shape[1] != model.mean.size or not isinstance(dimension, int)
            or not 1 <= dimension <= model.projection.shape[1]):
        raise ValueError("Feature length or requested projection dimension does not match the model.")
    projected = (features - model.mean) @ model.projection[:, :dimension]
    distances = template_distances(projected, model.centers[:, :, :dimension], metric)
    best_indices = distances.argmin(axis=2)
    class_distances = distances.min(axis=2)
    predictions = model.classes[class_distances.argmin(axis=1)]
    return MatchResult(predictions, class_distances, model.angles[best_indices])


def save_model(model: FisherModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=model.mean, projection=model.projection, classes=model.classes,
                        centers=model.centers, angles=model.angles, eigenvalues=model.eigenvalues,
                        pca_dimension=model.pca_dimension, regularization=model.regularization,
                        ridge_value=model.ridge_value)


def load_model(path: str | Path) -> FisherModel:
    with np.load(path, allow_pickle=False) as saved:
        model = FisherModel(saved["mean"], saved["projection"], saved["classes"], saved["centers"],
                            saved["angles"], saved["eigenvalues"], int(saved["pca_dimension"]),
                            float(saved["regularization"]), float(saved["ridge_value"]))
    dimension = model.projection.shape[1] if model.projection.ndim == 2 else 0
    expected_centers = (model.classes.size, model.angles.size, dimension)
    if (model.mean.ndim != 1 or model.projection.shape != (model.mean.size, dimension)
            or dimension == 0 or model.centers.shape != expected_centers
            or model.eigenvalues.shape != (dimension,) or model.classes.ndim != 1
            or model.angles.ndim != 1 or model.classes.size < 2 or model.angles.size == 0
            or not all(np.isfinite(a).all() for a in [model.mean, model.projection, model.centers, model.angles])):
        raise ValueError("The saved Fisher model is inconsistent or invalid.")
    return model


def dataset_records(data_dir: str | Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Validate the assignment's 108 eyes, three training and four test images per eye."""
    files = sorted(Path(data_dir).rglob("*.bmp"))
    if not files:
        raise ValueError(f"No BMP images found in {data_dir}.")
    records = []
    pattern = re.compile(r"(\d{3})_([12])_([1-4])\.bmp", re.IGNORECASE)
    for path in files:
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected CASIA filename: {path.name}")
        records.append(tuple(int(value) for value in match.groups()))
    if len(set(records)) != len(records):
        raise ValueError("Duplicate eye/session/image records found in the dataset.")
    labels, sessions, indices = np.array(records, dtype=np.int64).T
    if np.unique(labels).size != 108 or len(files) != 756:
        raise ValueError("This assignment requires exactly 108 eyes and 756 BMP images.")
    for label in np.unique(labels):
        for session, expected in [(1, {1, 2, 3}), (2, {1, 2, 3, 4})]:
            if set(indices[(labels == label) & (sessions == session)]) != expected:
                raise ValueError(f"Eye {label:03d}, session {session} has an incomplete image set.")
    return files, labels, sessions


def prepare_features(data_dir: str | Path, cache_path: str | Path,
                      feature_config: FeatureConfig | None = None,
                      matching_config: MatchingConfig | None = None,
                      rebuild: bool = False) -> dict[str, np.ndarray]:
    """Create or reuse preprocessing features after checking data and settings signatures."""
    feature_cfg, matching_cfg = feature_config or FeatureConfig(), matching_config or MatchingConfig()
    files, labels, sessions = dataset_records(data_dir)
    root = Path(data_dir).resolve()
    relative_paths = np.array([str(path.resolve().relative_to(root)) for path in files])
    angles = np.array(matching_cfg.rotation_angles, dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0 or not np.isfinite(angles).all() or np.unique(angles).size != angles.size:
        raise ValueError("Rotation angles must be a nonempty sequence of distinct finite values.")
    module_dir = Path(__file__).resolve().parent
    signature_data = {
        "version": 1,
        "files": [(name, path.stat().st_size, path.stat().st_mtime_ns) for name, path in zip(relative_paths.tolist(), files)],
        "source_hashes": {name: hashlib.sha256((module_dir / name).read_bytes()).hexdigest()
                          for name in ["IrisLocalization.py", "IrisNormalization.py", "ImageEnhancement.py", "FeatureExtraction.py"]},
        "localization": asdict(LocalizationConfig()), "normalization": asdict(NormalizationConfig()),
        "enhancement": asdict(EnhancementConfig()), "features": asdict(feature_cfg),
        "angles": angles.tolist(),
    }
    signature = json.dumps(signature_data, sort_keys=True)
    cache_path = Path(cache_path)
    expected_length = len(feature_cfg.channel_sigmas) * (feature_cfg.roi_height // feature_cfg.block_size) * (512 // feature_cfg.block_size) * 2
    if cache_path.is_file() and not rebuild:
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                cache = {key: saved[key] for key in saved.files}
            if (str(cache["signature"]) == signature
                    and cache["features"].shape == (len(files), expected_length)
                    and cache["rotated_training_features"].shape == ((sessions == 1).sum(), angles.size, expected_length)
                    and np.array_equal(cache["labels"], labels) and np.array_equal(cache["sessions"], sessions)
                    and np.isfinite(cache["features"]).all() and np.isfinite(cache["rotated_training_features"]).all()):
                print(f"Using checked feature cache: {cache_path}", flush=True)
                return cache
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            pass
    features, training_rotations = [], []
    for index, (path, session) in enumerate(zip(files, sessions)):
        try:
            image = load_image(path)
            boundaries = localize_iris(image)
            norm = normalize_iris(image, boundaries)
            enhanced = enhance_iris(norm.normalized, norm.valid_mask)
            vector = extract_features(enhanced.enhanced, enhanced.valid_mask, feature_cfg).features
            features.append(vector)
            if session == 1:
                rotated = []
                for angle in angles:
                    if angle == 0:
                        rotated.append(vector)
                        continue
                    norm = normalize_iris(image, boundaries, NormalizationConfig(start_angle_degrees=float(angle)))
                    enhanced = enhance_iris(norm.normalized, norm.valid_mask)
                    rotated.append(extract_features(enhanced.enhanced, enhanced.valid_mask, feature_cfg).features)
                training_rotations.append(np.stack(rotated))
        except (ValueError, OSError) as error:
            raise ValueError(f"Preprocessing failed for {path.name}: {error}") from error
        if (index + 1) % 42 == 0 or index + 1 == len(files):
            print(f"Preprocessed {index + 1}/{len(files)} images", flush=True)
    cache = {"features": np.stack(features), "rotated_training_features": np.stack(training_rotations),
             "labels": labels, "sessions": sessions, "paths": relative_paths, "angles": angles,
             "signature": np.array(signature), "feature_config": np.array(json.dumps(asdict(feature_cfg)))}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    return cache


def run_matching(cache: dict[str, np.ndarray], config: MatchingConfig,
                  output_dir: str | Path) -> dict:
    """Train on session 1 and save the three identification results for session 2."""
    train, test = cache["sessions"] == 1, cache["sessions"] == 2
    model = fit_fisher(cache["features"][train], cache["labels"][train],
                       cache["rotated_training_features"], config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, output_dir / "fisher_model.npz")
    outputs = {metric: match_features(cache["features"][test], model, metric)
               for metric in ["L1", "L2", "Cosine"]}
    test_labels, test_paths = cache["labels"][test], cache["paths"][test]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["image", "true_eye"]
        for metric in outputs:
            header += [f"{metric}_prediction", f"{metric}_distance", f"{metric}_angle", f"{metric}_correct"]
        writer.writerow(header)
        for i, (path, label) in enumerate(zip(test_paths, test_labels)):
            row = [path, int(label)]
            for result in outputs.values():
                winner = int(result.class_distances[i].argmin())
                row += [int(result.predictions[i]), float(result.class_distances[i, winner]),
                        float(result.best_angles[i, winner]), bool(result.predictions[i] == label)]
            writer.writerow(row)
    scores = {metric: result.class_distances for metric, result in outputs.items()}
    np.savez_compressed(output_dir / "matching_scores.npz", classes=model.classes,
                        test_labels=test_labels, test_paths=test_paths, **scores)
    metrics = {}
    for metric, result in outputs.items():
        correct = int((result.predictions == test_labels).sum())
        metrics[metric] = {"correct": correct, "total": int(test_labels.size),
                           "crr_percent": 100.0 * correct / test_labels.size}
    summary = {"training_samples": int(train.sum()), "test_samples": int(test.sum()),
               "classes": int(model.classes.size), "original_dimension": int(model.mean.size),
               "pca_dimension": model.pca_dimension, "fisher_dimension": int(model.projection.shape[1]),
               "regularization": model.regularization, "ridge_value": model.ridge_value,
               "rotation_angles": model.angles.tolist(), "metrics": metrics,
               "split": "Session 1 for fitting and templates; session 2 for testing only.",
               "l2_convention": "Squared Euclidean distance, matching Eq. (8).",
               "centering": "Subtract the training-set mean before the PCA/Fisher projection.",
               "configuration": asdict(config)}
    (output_dir / "matching_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent / "CASIA Iris Image Database (version 1.0)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "matching")
    parser.add_argument("--components", type=int, default=107, help="Fisher dimensions; maximum 107 for 108 eyes")
    parser.add_argument("--regularization", type=float, default=0.01, help="Relative within-class scatter ridge")
    parser.add_argument("--frequencies", type=float, nargs=2, default=(0.1, 0.1))
    parser.add_argument("--rebuild-cache", action="store_true", help="Recompute all preprocessing features")
    args = parser.parse_args()
    cfg = MatchingConfig(n_components=args.components, regularization=args.regularization)
    if not 1 <= cfg.n_components <= 107 or not np.isfinite(cfg.regularization) or cfg.regularization <= 0:
        parser.error("Components must be 1-107 and regularization must be positive and finite.")
    if not all(np.isfinite(f) and 0 < f < 0.5 for f in args.frequencies):
        parser.error("Each frequency must be finite and between 0 and 0.5.")
    try:
        cache = prepare_features(args.data_dir, args.output_dir / "feature_cache.npz",
                                  FeatureConfig(channel_frequencies=tuple(args.frequencies)), cfg, args.rebuild_cache)
        summary = run_matching(cache, cfg, args.output_dir)
    except (ValueError, OSError, KeyError, np.linalg.LinAlgError) as error:
        parser.exit(1, f"Matching failed: {error}\n")
    print(f"Training/test samples: {summary['training_samples']}/{summary['test_samples']}")
    print(f"Fisher dimensions: {summary['fisher_dimension']}")
    for metric, result in summary["metrics"].items():
        print(f"{metric} CRR: {result['crr_percent']:.2f}% ({result['correct']}/{result['total']})")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
