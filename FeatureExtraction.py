"""Extract Ma et al. (2003) spatial-filter features from an enhanced iris."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from ImageEnhancement import enhance_iris
from IrisLocalization import load_image, localize_iris
from IrisNormalization import normalize_iris


@dataclass(frozen=True)
class FeatureConfig:
    roi_height: int = 48  # Use the rows closest to the pupil.
    block_size: int = 8  # Each block contributes mean magnitude and absolute deviation.
    channel_sigmas: tuple[tuple[float, float], ...] = ((3.0, 1.5), (4.5, 1.5))
    channel_frequencies: tuple[float, ...] = (0.1, 0.1)  # Implementation defaults; not specified in the paper.
    truncate: float = 3.0  # Sample each Gaussian envelope up to three standard deviations.


@dataclass(frozen=True)
class FeatureResult:
    features: np.ndarray
    responses: np.ndarray
    block_statistics: np.ndarray
    block_valid_fraction: np.ndarray
    kernels: tuple[np.ndarray, ...]


def make_spatial_filter(sigma_x: float, sigma_y: float, frequency: float,
                        truncate: float = 3.0) -> np.ndarray:
    """Sample Eq. (3): a Gaussian envelope times cos(2*pi*f*sqrt(x*x+y*y))."""
    if not np.isfinite([sigma_x, sigma_y, frequency, truncate]).all():
        raise ValueError("Filter parameters must be finite.")
    if min(sigma_x, sigma_y, truncate) <= 0 or not 0 < frequency < 0.5:
        raise ValueError("Sigma and truncation must be positive; frequency must be between 0 and 0.5.")
    half_x, half_y = int(np.ceil(truncate * sigma_x)), int(np.ceil(truncate * sigma_y))
    x, y = np.meshgrid(np.arange(-half_x, half_x + 1), np.arange(-half_y, half_y + 1))
    envelope = np.exp(-0.5 * ((x / sigma_x)**2 + (y / sigma_y)**2))
    modulation = np.cos(2 * np.pi * frequency * np.hypot(x, y))
    return (envelope * modulation / (2 * np.pi * sigma_x * sigma_y)).astype(np.float32)


def _fill_invalid_roi(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolate missing angular samples before convolution; keep the original mask."""
    filled = roi.copy()
    good_rows = np.flatnonzero(mask.any(axis=1))
    columns = np.arange(roi.shape[1])
    for row in range(roi.shape[0]):
        if mask[row].all():
            continue
        source_row = row if mask[row].any() else good_rows[np.argmin(abs(good_rows - row))]
        good_columns = np.flatnonzero(mask[source_row])
        interpolated = np.interp(columns, good_columns, roi[source_row, good_columns],
                                 period=roi.shape[1])
        filled[row, ~mask[row]] = interpolated[~mask[row]]
    return filled


def _filter_roi(roi: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    half_y, half_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    # Angle is periodic; the two radial ends use reflection without repeating the edge.
    padded = np.pad(roi, ((half_y, half_y), (0, 0)), mode="reflect")
    padded = np.pad(padded, ((0, 0), (half_x, half_x)), mode="wrap")
    response = cv2.filter2D(padded, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
    return response[half_y:half_y + roi.shape[0], half_x:half_x + roi.shape[1]].copy()


def _block_statistics(response: np.ndarray, mask: np.ndarray,
                       block_size: int) -> tuple[np.ndarray, np.ndarray]:
    grid_rows, grid_cols = response.shape[0] // block_size, response.shape[1] // block_size
    shape = (grid_rows, block_size, grid_cols, block_size)
    magnitude = np.abs(response).astype(np.float64).reshape(shape).transpose(0, 2, 1, 3)
    block_mask = mask.reshape(shape).transpose(0, 2, 1, 3)
    counts = block_mask.sum(axis=(2, 3))
    means = np.divide((magnitude * block_mask).sum(axis=(2, 3)), counts,
                      out=np.zeros((grid_rows, grid_cols)), where=counts > 0)
    # The second feature is mean absolute deviation of |F|, as in Eq. (5).
    deviations = np.divide((np.abs(magnitude - means[:, :, None, None]) * block_mask).sum(axis=(2, 3)),
                           counts, out=np.zeros_like(means), where=counts > 0)
    statistics = np.stack([means, deviations], axis=-1).astype(np.float32)
    valid_fraction = (counts / block_size**2).astype(np.float32)
    return statistics, valid_fraction


def extract_features(enhanced: np.ndarray, valid_mask: np.ndarray | None = None,
                      config: FeatureConfig | None = None) -> FeatureResult:
    """Return channel-major, row-major [mean, deviation] pairs; default length is 1536."""
    cfg = config or FeatureConfig()
    if enhanced.ndim != 2 or enhanced.size == 0:
        raise ValueError("Expected a nonempty two-dimensional enhanced image.")
    image = enhanced.astype(np.float32)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("Enhanced intensities must be finite and within 0-255.")
    if not isinstance(cfg.roi_height, int) or not 2 <= cfg.roi_height <= image.shape[0]:
        raise ValueError("ROI height must be an integer between 2 and the image height.")
    if not isinstance(cfg.block_size, int) or cfg.block_size < 1:
        raise ValueError("Block size must be a positive integer.")
    if cfg.roi_height % cfg.block_size or image.shape[1] % cfg.block_size:
        raise ValueError("ROI height and image width must be divisible by the block size.")
    if not cfg.channel_sigmas or len(cfg.channel_sigmas) != len(cfg.channel_frequencies):
        raise ValueError("Each filter channel must have a sigma pair and a frequency.")
    mask = np.ones(image.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if mask.shape != image.shape or mask.dtype != np.bool_:
        raise ValueError("The validity mask must be boolean and match the enhanced image.")
    roi, roi_mask = image[:cfg.roi_height], mask[:cfg.roi_height]
    if not roi_mask.any():
        raise ValueError("The feature ROI contains no valid sampling points.")
    filled = _fill_invalid_roi(roi, roi_mask)
    kernels, responses, statistics = [], [], []
    valid_fraction = None
    for (sigma_x, sigma_y), frequency in zip(cfg.channel_sigmas, cfg.channel_frequencies):
        kernel = make_spatial_filter(sigma_x, sigma_y, frequency, cfg.truncate)
        response = _filter_roi(filled, kernel)
        stats, valid_fraction = _block_statistics(response, roi_mask, cfg.block_size)
        kernels.append(kernel)
        responses.append(response)
        statistics.append(stats)
    block_stats = np.stack(statistics)
    # Flatten channel 1 first, then channel 2; each block keeps its mean/deviation pair.
    features = block_stats.reshape(-1).copy()
    return FeatureResult(features, np.stack(responses), block_stats,
                         valid_fraction, tuple(kernels))


def save_features(result: FeatureResult, output_dir: str | Path, stem: str) -> dict[str, Path]:
    """Save the raw vector and intermediate arrays, without training or classification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"vector": output_dir / f"{stem}_features.npy",
             "arrays": output_dir / f"{stem}_features.npz",
             "response_preview": output_dir / f"{stem}_filter_responses.png"}
    np.save(paths["vector"], result.features, allow_pickle=False)
    kernels = {f"kernel_{i + 1}": kernel for i, kernel in enumerate(result.kernels)}
    np.savez_compressed(paths["arrays"], features=result.features, responses=result.responses,
                        block_statistics=result.block_statistics,
                        block_valid_fraction=result.block_valid_fraction, **kernels)
    previews = []
    for response in result.responses:
        magnitude = np.abs(response)
        high = float(magnitude.max())
        preview = np.zeros(magnitude.shape, dtype=np.uint8) if high == 0 else (
            np.clip(np.rint(magnitude * (255.0 / high)), 0, 255).astype(np.uint8))
        previews.append(preview)
    ok, encoded = cv2.imencode(".png", np.vstack(previews))
    if not ok:
        raise ValueError("Cannot encode the filter response preview.")
    encoded.tofile(paths["response_preview"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=Path, help="Original iris image; preprocess automatically")
    inputs.add_argument("--enhanced", type=Path, help="NPZ saved by ImageEnhancement.py")
    parser.add_argument("--frequencies", type=float, nargs=2, default=(0.1, 0.1),
                        metavar=("F1", "F2"), help="Cycles per pixel; paper does not specify numeric f")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent / "CASIA Iris Image Database (version 1.0)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "features")
    args = parser.parse_args()
    cfg = FeatureConfig(channel_frequencies=tuple(args.frequencies))
    try:
        if args.enhanced is not None:
            with np.load(args.enhanced, allow_pickle=False) as saved:
                enhanced, mask = saved["enhanced"], saved["valid_mask"]
            stem = args.enhanced.stem.removesuffix("_enhanced")
            source = {"enhanced_file": str(args.enhanced.resolve())}
        else:
            image_path = args.image
            if image_path is None:
                image_path = next(iter(sorted(args.data_dir.rglob("*.bmp"))), None)
                if image_path is None:
                    parser.error("No BMP images found. Set --image, --enhanced, or --data-dir.")
            image = load_image(image_path)
            norm = normalize_iris(image, localize_iris(image))
            enhancement = enhance_iris(norm.normalized, norm.valid_mask)
            enhanced, mask, stem = enhancement.enhanced, enhancement.valid_mask, image_path.stem
            source = {"source_image": str(image_path.resolve())}
        result = extract_features(enhanced, mask, cfg)
        paths = save_features(result, args.output_dir, stem)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Feature extraction failed: {error}\n")
    metadata = {
        **source, "parameters": asdict(cfg), "feature_length": int(result.features.size),
        "response_shape": list(result.responses.shape),
        "kernel_shapes": [list(kernel.shape) for kernel in result.kernels],
        "feature_order": "Channel first, then block row, block column, then mean/magnitude absolute deviation.",
        "border_handling": "Periodic angular padding and reflected radial padding.",
        "invalid_samples": "Interpolate before filtering; exclude invalid locations from block statistics.",
        "empty_blocks": int((result.block_valid_fraction == 0).sum()),
        "minimum_block_valid_fraction": float(result.block_valid_fraction.min()),
        "frequency_note": "Numeric frequency and discrete-kernel support are implementation choices, not specified in the paper.",
        "response_preview": "Magnitude scaled separately in each channel for display only.",
    }
    metadata_path = args.output_dir / f"{stem}_features.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Feature length: {result.features.size}")
    print(f"Filter response shape: {result.responses.shape}")
    print(f"Empty blocks: {metadata['empty_blocks']}")
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
